import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import os

import rclpy
import rosbag2_py

from rclpy.node import Node
from rclpy.serialization import serialize_message

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from message_filters import Subscriber, ApproximateTimeSynchronizer

from rl_games.algos_torch import model_builder, torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStdObs
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from rl_games.algos_torch.players import PpoPlayerContinuous

from components.utils import *
from components.velocity_field import VelocityField
from components.networks import RMANetworkBuilder, MLPNetworkBuilder, MLPWithRNNNetworkBuilder, RMAWithTCNNetworkBuilder
from components.config import *

  
class TargetVelsPublisher(Node):

# 1. IP Robomaster: Am Vicon-PC checken -> http://10.65.80.1/
# 2. In der Liste nachschauen nach robomaster: ep01 ....
# 3. Name: robo password: robo

# Potential Debugs
# 1. Disable nav_deep on Robomaster -> sudo systemctl disable nav_deep_start.service
# Bug: random index out of bounds error (difficult to reproduce)

# ros2 param get /ep05/Robomaster_Node acceleration_linear_max
# ros2 param get /ep05/Robomaster_Node acceleration_angular_max
# ros2 param get /ep05/Robomaster_Node velocity_linear_max
# ros2 param get /ep05/Robomaster_Node velocity_angular_max

  def __init__(self, args):
    super().__init__('target_vels_publisher')
    with open(os.path.join(os.getcwd(), 'rm_publisher/config/rm_publisher.yaml'), 'r') as f:
      cfg = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), 'rm_publisher/config/rm_network.yaml'), 'r') as f:
      cfg_train = yaml.load(f, Loader=yaml.SafeLoader)['params']

    with open(os.path.join(os.getcwd(), 'rm_publisher/config/racetrack_network.yaml')) as f:
      cfg_racetrack = yaml.load(f, Loader=yaml.SafeLoader)['params']

    model_builder.register_network('rma', lambda **kwargs: RMANetworkBuilder())
    model_builder.register_network('mlp_net', lambda **kwargs: MLPNetworkBuilder())
    model_builder.register_network('mlp_rnn', lambda **kwargs: MLPWithRNNNetworkBuilder())
    model_builder.register_network('rma_tcn', lambda **kwargs: RMAWithTCNNetworkBuilder())
    builder = model_builder.ModelBuilder()
    
    self.all_actions = []
    self.step_counter = 0
    self.device = cfg_train['config'].get('device_type', 'cuda')
    self.num_rm = cfg['robomaster']['numRms'] # Number of Robomasters
    self.rm_ids = range(self.num_rm) # List of Robomaster ID's

    self.action_size = cfg['model']['actionSize'] # Number of actions
    self.obs_size = cfg['model']['obsSize'] # Number of observations
    self.history_size = cfg['model']['actionObsHistory'] # Number of saved observations

    self.racetrack_action_size = cfg['racetrack']['actionSize']
    self.racetrack_obs_size = cfg['racetrack']['obsSize']
    self.racetrack_history_size = cfg['racetrack']['actionObsHistory']

    # Allocate buffers
    self.rew_buf = torch.zeros(self.num_rm, device=self.device, dtype=torch.float)
    self.reset_buf = torch.zeros(self.num_rm, device=self.device, dtype=torch.long)
    self.progress_buf = torch.zeros(self.num_rm, device=self.device, dtype=torch.long)
    self.task_buf = torch.zeros(self.num_rm, device=self.device, dtype=torch.long)
    self.envs_aranged = torch.arange(0, self.num_rm, device=self.device, dtype=torch.long)

    self.map_size = args.get('map_size', cfg['map']['mapSize']) 
    self.cell_size = cfg['map']['cellSize'] # Size of cell in map
    self.boundary_size = cfg['map']['boundarySize']
    self.max_episode_length = cfg['map']['episodeLength'] # Max length of episode before velocity field resets

    self.use_interpolated = args.get('interpolated', cfg['map']['useInterpolated'])
    self.use_actions = cfg['robomaster']['use_actions']
    self.use_wheel_vel = cfg['robomaster']['wheel']
    self.max_linear_velocity = cfg['robomaster']['maxLinearVelocity']
    self.max_angular_velocity = cfg['robomaster']['maxAngularVelocity']
    self.min_velocity = cfg['robomaster']['minVelocity']
    self.lin_vel_x_scale = cfg["scales"]["linearVelocityXScale"]
    self.lin_vel_y_scale = cfg["scales"]["linearVelocityYScale"]
    self.ang_vel_scale = cfg["scales"]["angularVelocityScale"]
    self.command_scale = np.array([self.lin_vel_x_scale, self.lin_vel_y_scale, self.ang_vel_scale])

    self.initial_position_offset = np.array(args.get('offset', cfg['robomaster']['initialPosition'])) #  Offset for initial position of robomaster in vicon
    self.initial_rotation_offset = np.array(cfg['robomaster']['initialRotation']) # Quaternion for rotating 90 degrees

    self.visualize = cfg['map']['enableDebugVis']
    self.racetrack = cfg['map']['racetrack']
    self.record = cfg['data']['record_bag']
    self.recording_threshold = cfg['data']['steps']
    self.bag = cfg['data']['bag'] 
    self.publish = cfg['robomaster']['publish']

    self.robomaster = args.get('robomaster', cfg['robomaster']['name']) # Name of robomaster

    config = {
      'actions_num' : self.action_size,
      'input_shape' : (self.obs_size,),
      'num_seqs' : 1,
      'value_size' : 1,
      'normalize_value' : cfg_train['config'].get('normalize_value', True),
      'normalize_input' : cfg_train['config'].get('normalize_input', False),
    }

    # Build model and load nn
    self.model = builder.load(cfg_train).build(config)
    self.model.to(self.device)
    self.model.eval()
    
    checkpoint = torch_ext.load_checkpoint(os.path.join(os.getcwd(),  cfg['model']['path']))
    # self.model.load_state_dict(checkpoint['model'])
    missing, unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)

    racetrack_config = {
      'actions_num': self.racetrack_action_size,
      'input_shape' : (self.obs_size,),
      'num_seqs' : 1,
      'value_size' : 1,
      'normalize_value' : cfg_racetrack['config'].get('normalize_value', True),
      'normalize_input' : cfg_racetrack['config'].get('normalize_input', False),
    }

    self.racetrack_model = builder.load(cfg_racetrack).build(racetrack_config)
    self.racetrack_model.to(self.device)
    self.racetrack_model.eval()

    racetrack_checkpoint = torch_ext.load_checkpoint(os.path.join(os.getcwd(), cfg['racetrack']['path']))
    self.racetrack_model.load_state_dict(racetrack_checkpoint['model'])

    self.history_queue = torch.zeros((self.num_rm, self.history_size, self.action_size + self.obs_size)
                                     ,dtype=torch.float, device=self.device)
    self.racetrack_history_queue = torch.zeros((self.num_rm, self.racetrack_history_size, self.racetrack_action_size + self.racetrack_obs_size)
                                     ,dtype=torch.float, device=self.device)
    self.history_pointer = torch.zeros((self.num_rm,), dtype=torch.long, device=self.device)
    self.racetrack_history_pointer = torch.zeros((self.num_rm,), dtype=torch.long, device=self.device)

    self.actions = np.zeros((self.num_rm, self.action_size))
    self.obs = torch.zeros((self.num_rm, self.obs_size ), device=self.device)

    self.racetrack_actions = np.zeros((self.num_rm, self.racetrack_action_size))
    self.racetrack_obs = torch.zeros((self.num_rm, self.racetrack_obs_size ), device=self.device)

    # Initialize gravity and forward vectors
    self.gravity_vec = get_axis_params(-1, 2)
    self.gravity_vec_rm_proj = get_axis_params(-1., 1)
    self.forward_vec = np.array([1, 0, 0])

    self.current_orientation = np.zeros(4)
    self.current_position = np.zeros(3)
    self.linear_velocities = np.zeros(2)
    self.angular_velocity = np.zeros(1)
    self.clipped_velocities = np.zeros(4)
    self.start_map = -(self.map_size / 2.0) 
    self.end_map = self.map_size / 2.0 + self.cell_size + (self.boundary_size * self.cell_size)
    self.x_pos = np.arange(self.start_map, self.end_map, self.cell_size)
    self.y_pos = np.arange(self.start_map, self.end_map, self.cell_size)
    self.idx_offset = self.map_size / 2.0

    # Initialize Velocity fields
    self.velocity_field_functions = VelocityField(self.num_rm, self.visualize)
    
    self.rot_matrix_z = np.zeros((self.num_rm, 2, 2), dtype=float)

    self.map_y, self.map_x = np.meshgrid(self.y_pos, self.x_pos, indexing='ij')
    self.map_xy = np.concatenate((np.expand_dims(self.map_x, axis=-1), np.expand_dims(self.map_y, axis=-1)), axis=-1)
    self.x = self.map_xy[..., 0]
    self.y = self.map_xy[..., 1]

    self.xy_vels_grid = np.zeros_like(self.map_xy).reshape(1, *self.map_xy.shape).repeat(self.num_rm, axis=0)
    self.xy_vels_grid_reset = self.xy_vels_grid.copy()
    self.orientation_map = np.zeros_like(self.xy_vels_grid) 
    self.boundary_grid = self.create_boundary_grid(0, self.xy_vels_grid.shape[1])

    self.max_cells = self.xy_vels_grid.shape[1]

    self.velocity_field_functions.reset = True

    self.steps = 0

    self.wheel_seperation_width = 0.40
    self.wheel_seperation_length = 0.40
    self.wheel_radius = 0.05
    self.wheel_velocities_matrix = np.ones((4, 3))
    self.wheel_velocities_matrix[1, 1] = self.wheel_velocities_matrix[2, 1] = -1
    self.wheel_velocities_matrix[2, 2] = self.wheel_velocities_matrix[0, 2] = (self.wheel_seperation_width + self.wheel_seperation_length)
    self.wheel_velocities_matrix[3, 2] = self.wheel_velocities_matrix[1, 2] = -(self.wheel_seperation_width + self.wheel_seperation_length)
    self.wheel_velocities_matrix = (1 / self.wheel_radius) * self.wheel_velocities_matrix
    
    self.current_cells = np.zeros(2)
    self.current_y_cell = 0
    self.current_x_cell = 0
    self.base_linear_velocities = np.zeros(3)
    self.base_angular_velocities = np.zeros(3)
    self.target_velocities = np.zeros((self.num_rm, 2))
    self.joint_states = JointState()

    if self.visualize:
      self.fig, self.ax = plt.subplots()
      plt.ion()

    self.actions_history = np.load('actions_hist.npz', allow_pickle=True)['actions_history']
    if self.record:
      id = 0
      path = f'bags/rm_{self.robomaster}_bag_'
      while os.path.exists(f'{path}{id}'):
        id += 1
      self.writer = rosbag2_py.SequentialWriter()
      storage_options = rosbag2_py._storage.StorageOptions(uri=f'{path}{id}', storage_id='sqlite3')
      converter_options = rosbag2_py._storage.ConverterOptions('', '')
      self.writer.open(storage_options, converter_options)

      # Create TopicMetadata instances for each topic
      topics = [
        {'name': f'/{self.robomaster}/joint_states_bag', 'type': 'sensor_msgs/msg/JointState'},
        {'name': f'/{self.robomaster}/vicon_pose_bag', 'type': 'geometry_msgs/msg/PoseWithCovarianceStamped'},
        {'name': f'/{self.robomaster}/vicon_odom_bag', 'type': 'nav_msgs/msg/Odometry'},
        {'name': f'/{self.robomaster}/cmd_wheel_speed_bag', 'type': 'sensor_msgs/msg/JointState'},
      ]

      for topic in topics:
        topic_info = rosbag2_py._storage.TopicMetadata(
            name=topic['name'],
            type=topic['type'],
            serialization_format='cdr'
        )
        self.writer.create_topic(topic_info)

      self.record_timer = ApproximateTimeSynchronizer((Subscriber(self, PoseWithCovarianceStamped, f'/{self.robomaster}/vicon_pose'), Subscriber(self, Odometry, f'/{self.robomaster}/vicon_odom'), Subscriber(self, JointState, f'/{self.robomaster}/joint_states')), 10, 1)
      self.record_timer.registerCallback(self.recorder)

    if self.bag:
      self.timer = ApproximateTimeSynchronizer((Subscriber(self, PoseWithCovarianceStamped, f'/{self.robomaster}/vicon_pose_bag'), Subscriber(self, Odometry, f'/{self.robomaster}/vicon_odom_bag'), Subscriber(self, JointState, f'/{self.robomaster}/joint_states_bag')), 10, 1)
    else:
      self.timer = ApproximateTimeSynchronizer((Subscriber(self, PoseWithCovarianceStamped, f'/{self.robomaster}/vicon_pose'), Subscriber(self, Odometry, f'/{self.robomaster}/vicon_odom'), Subscriber(self, JointState, f'/{self.robomaster}/joint_states')), 10, 1)
    self.timer.registerCallback(self.update_states)

    # Adjust the desired publishing rate (e.g., 10 Hz)
    self.publishing_rate = 50.0

    # Create a timer to control the publishing rate
    self.publish_timer = self.create_timer(1.0 / self.publishing_rate, self.publish_callback)

    if self.use_wheel_vel or self.use_actions:
      self.velocity_publisher = self.create_publisher(JointState, f'/{self.robomaster}/cmd_wheel_speed', 10)
    else:
      self.velocity_publisher = self.create_publisher(Twist, f'/{self.robomaster}/cmd_vel', 10)

    self.visualization_publisher = self.create_publisher(MarkerArray, f'/{self.robomaster}/velocity_field', 10)
    self.visualization_timer = self.create_timer(1.0 / self.publishing_rate, self.visualization_callback)
    self.orientation_publisher = self.create_publisher(MarkerArray, f'/{self.robomaster}/orientation_map', 10)
    self.orientation_timer = self.create_timer(1.0 / self.publishing_rate, self.orientation_callback)

    self.reset_position = True
    self.reset_velocity_field([0])
    if self.racetrack:
      self.generate_racetrack()

  def publish_callback(self):
    # Publish velocities
    self.publish_velocities()
    self.post_publish()

  def recorder(self, pwcs:PoseWithCovarianceStamped, odom:Odometry, joint:JointState):
    if self.record:
      self.steps += 1
      print(self.steps)
      if self.steps <= self.recording_threshold:
        self.writer.write(f'/{self.robomaster}/joint_states_bag', serialize_message(joint), self.get_clock().now().nanoseconds)
        self.writer.write(f'/{self.robomaster}/vicon_pose_bag', serialize_message(pwcs), self.get_clock().now().nanoseconds)
        self.writer.write(f'/{self.robomaster}/vicon_odom_bag', serialize_message(odom), self.get_clock().now().nanoseconds)

  def visualization_callback(self):
    header = Header(stamp=self.get_clock().now().to_msg(), frame_id=f'map')
    velocity_field = self.get_velocity_field(header)
    robomaster = self.get_robomaster(header)
    velocity_field.markers.append(robomaster)
    self.visualization_publisher.publish(velocity_field)

  def orientation_callback(self):
    header = Header(stamp=self.get_clock().now().to_msg(), frame_id=f'map')
    orientation_map = self.get_orientation_map(header)
    self.orientation_publisher.publish(orientation_map)

  def get_orientation_map(self, header):
    markers = MarkerArray()
    direction_matrix = np.zeros((self.orientation_map.shape[1], self.orientation_map.shape[2]))
    quat_matrix = np.zeros_like(self.boundary_grid)
    for i in range(self.orientation_map.shape[1]):
      for j in range(self.orientation_map.shape[2]):
        marker = Marker(header=header)
        marker.ns = f'Velocity Field Arrow'
        marker.id = i * self.boundary_grid.shape[1] + j
        marker.action = marker.ADD
        
        
        marker.color.g = 1.0
        marker.color.r = 1.0
        marker.color.a = 0.5
        
        direction = np.array([self.orientation_map[0,i,j,0], self.orientation_map[0,i,j,1]])
        quaternion = self.direction_to_quaternion(direction)
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]

        # compute length of Arrow in x depending on velocity and cell size
        vel = abs(np.sqrt(self.orientation_map[0,i,j,0]**2+self.orientation_map[0,i,j,1]**2) / (2*self.max_linear_velocity)) * self.cell_size
        if vel <= 0.01:
          marker.type = marker.SPHERE
          marker.scale.x = marker.scale.z = marker.scale.y = self.cell_size / 5
          direction_matrix[i,j] = self.cell_size / 5
        else:
          marker.type = marker.ARROW
          marker.scale.x = vel
          direction_matrix[i,j] = abs(np.sqrt(self.orientation_map[0,i,j,0]**2+self.orientation_map[0,i,j,1]**2))
          marker.scale.z = marker.scale.y = 0.2 * self.cell_size
        
        quat_matrix[i,j] = quaternion[:1]
        
        marker.pose.position.x = self.map_xy[i,j,0]
        marker.pose.position.y = self.map_xy[i,j,1]
        marker.pose.position.z = 0.0 
        marker.frame_locked = True
        markers.markers.append(marker)
    return markers

  def get_robomaster(self, header):
    robomaster = Marker(header=header)
    robomaster.ns = f'Robomaster'
    robomaster.action = robomaster.ADD
    robomaster.type = robomaster.ARROW
    robomaster.color.r = 1.0
    robomaster.color.a = 1.0

    robomaster.pose.orientation.x = self.current_orientation[0]
    robomaster.pose.orientation.y = self.current_orientation[1]
    robomaster.pose.orientation.z = self.current_orientation[2]
    robomaster.pose.orientation.w = self.current_orientation[3]

    robomaster.pose.position.x = self.current_position[0]
    robomaster.pose.position.y = self.current_position[1]
    robomaster.pose.position.z = self.current_position[2]

    robomaster.scale.x = self.cell_size
    robomaster.scale.y = robomaster.scale.z = self.cell_size / 2
    return robomaster
    
  def get_velocity_field(self, header):
    markers = MarkerArray()
    direction_matrix = np.zeros((self.xy_vels_grid.shape[1], self.xy_vels_grid.shape[2]))
    quat_matrix = np.zeros_like(self.boundary_grid)
    for i in range(self.xy_vels_grid.shape[1]):
      for j in range(self.xy_vels_grid.shape[2]):
        marker = Marker(header=header)
        marker.ns = f'Velocity Field Arrow'
        marker.id = i * self.boundary_grid.shape[1] + j
        marker.action = marker.ADD
        
        if self.xy_vels_grid[0,i,j,0] == self.orientation_map[0,i,j,0] and self.xy_vels_grid[0,i,j,1] == self.orientation_map[0,i,j,1]:
          marker.color.g = 1.0
          marker.color.a = 0.2
        else:
          marker.color.b = 1.0
          marker.color.a = 0.8
        direction = np.array([self.xy_vels_grid[0,i,j,0], self.xy_vels_grid[0,i,j,1]])
        quaternion = self.direction_to_quaternion(direction)
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]

        # compute length of Arrow in x depending on velocity and cell size
        vel = abs(np.sqrt(self.xy_vels_grid[0,i,j,0]**2+self.xy_vels_grid[0,i,j,1]**2) / (2*self.max_linear_velocity)) * self.cell_size
        if vel <= 0.01:
          marker.type = marker.SPHERE
          marker.scale.x = marker.scale.z = marker.scale.y = self.cell_size / 5
          direction_matrix[i,j] = self.cell_size / 5
        else:
          marker.type = marker.ARROW
          marker.scale.x = vel
          direction_matrix[i,j] = abs(np.sqrt(self.xy_vels_grid[0,i,j,0]**2+self.xy_vels_grid[0,i,j,1]**2))
          marker.scale.z = marker.scale.y = 0.2 * self.cell_size
        
        quat_matrix[i,j] = quaternion[:1]
        
        marker.pose.position.x = self.map_xy[i,j,0]
        marker.pose.position.y = self.map_xy[i,j,1]
        marker.pose.position.z = 0.0 
        marker.frame_locked = True
        markers.markers.append(marker)
    return markers

  def direction_to_quaternion(self, direction):
    # Normalize the direction vector
    direction /= np.linalg.norm(direction)
    
    # Calculate angle theta
    theta = np.arctan2(direction[1], direction[0])
    
    # Compute quaternion components
    x = np.cos(theta / 2)
    y = np.sin(theta / 2)

    # filter nan values
    if np.isnan(x):
      x = 0.0
    if np.isnan(y):
      y = 0.0
    
    return np.array([x, y, 0.0, 0.0])
  
  def generate_racetrack(self):
    velocity = self.max_linear_velocity
    self.xy_vels_grid[0,:,:,1] = 0.0
    self.xy_vels_grid[0,:,:,0] = 0.0

    # Straight
    self.xy_vels_grid[0, 1:4, 4:18, 0] = -velocity
    self.xy_vels_grid[0, 1:5, 1:4, 1] = velocity
    self.xy_vels_grid[0, 5:8, 1:10, 0] = velocity
    self.xy_vels_grid[0, 5:9, 10:13, 1] = velocity
    self.xy_vels_grid[0, 9:12, 4:13, 0] = -velocity
    self.xy_vels_grid[0, 9:15, 1:4, 1] = velocity
    self.xy_vels_grid[0, 15:18, 1:15, 0] = velocity
    self.xy_vels_grid[0, 4:18, 15:18, 1] = -velocity
    # self.xy_vels_grid[0, 21:24, 2:23, 0] = velocity
    # self.xy_vels_grid[0, 10:23, 1:4, 1] = velocity
    # self.xy_vels_grid[0, 9:12, 2:8, 0] = -velocity
    # self.xy_vels_grid[0, 10:19, 6:9, 1] = -velocity
    # self.xy_vels_grid[0, 17:20, 7:19, 0] = -velocity
    # self.xy_vels_grid[0, 14:19, 17:20, 1] = velocity

    # Curve 
    # self.xy_vels_grid[0, 1:3, 2:5, 1] = velocity
    # self.xy_vels_grid[0, 4:7, 1:3, 0] = velocity
    # self.xy_vels_grid[0, 5:7, 9:12, 1] = velocity
    # self.xy_vels_grid[0, 8:11, 11:13, 0] = -velocity
    # self.xy_vels_grid[0, 8:11, 11:13, 0] = -velocity
    # self.xy_vels_grid[0, 9:11, 2:5, 1] = velocity
    # self.xy_vels_grid[0, 14:17, 1:3, ] = velocity
    # self.xy_vels_grid[0, 16:18, 14:17, 1] = -velocity
    # self.xy_vels_grid[0, 2:5, 16:18, 0] = -velocity
    # self.xy_vels_grid[0, 10:13, 7:9, 0] = -velocity
    # self.xy_vels_grid[0, 9:11, 2:5, 1] = velocity
    # self.xy_vels_grid[0, 20:23, 1:3, 0] = velocity
    # self.xy_vels_grid[0, 22:24, 20:23, 1] = -velocity
    # self.xy_vels_grid[0, 12, 21:24, :] = -velocity
    # self.xy_vels_grid[0, 11, 20:23, :] = -velocity
    # self.xy_vels_grid[0, 12, 21, 0] = 0
    # self.xy_vels_grid[0, 11, 20, 0] = 0
    # self.xy_vels_grid[0, 12, 23, 1] = 0
    # self.xy_vels_grid[0, 11, 22, 1] = 0

    # self.xy_vels_grid[0, 2:5, 20:22, 0] = -velocity

    # orientation
    self.orientation_map[0, 1:4, 2:17, 0] = -velocity
    self.orientation_map[0, 2:7, 1:4, 1] = velocity
    self.orientation_map[0, 5:8, 2:12, 0] = velocity
    self.orientation_map[0, 6:11, 10:13, 1] = velocity
    self.orientation_map[0, 9:12, 2:12, 0] = -velocity
    self.orientation_map[0, 10:17, 1:4, 1] = velocity
    self.orientation_map[0, 15:18, 2:17, 0] = velocity
    self.orientation_map[0, 2:17, 15:18, 1] = -velocity

    self.orientation_map[0, 1:3, 2:5, 1] = velocity
    self.orientation_map[0, 4:7, 1:3, 0] = velocity
    self.orientation_map[0, 5:7, 9:12, 1] = velocity
    self.orientation_map[0, 8:11, 11:13, 0] = -velocity
    self.orientation_map[0, 8:11, 11:13, 0] = -velocity
    self.orientation_map[0, 9:11, 2:5, 1] = velocity
    self.orientation_map[0, 14:17, 1:3, ] = velocity
    self.orientation_map[0, 16:18, 14:17, 1] = -velocity
    self.orientation_map[0, 2:5, 16:18, 0] = -velocity

    self.orientation_map[0, 15:18, 4:14, 0] = 0.
    self.orientation_map[0, 15:18, 4:14, 1] = -velocity

  def generate_racetrack1(self):
    velocity = self.max_linear_velocity
    self.xy_vels_grid[0,:,:,1] = 0.0
    self.xy_vels_grid[0,:,:,0] = 0.0

    # Straight
    self.xy_vels_grid[0, 1:4, 2:17, 0] = -velocity
    self.xy_vels_grid[0, 2:7, 1:4, 1] = velocity
    self.xy_vels_grid[0, 5:8, 2:12, 0] = velocity
    self.xy_vels_grid[0, 6:11, 10:13, 1] = velocity
    self.xy_vels_grid[0, 9:12, 2:12, 0] = -velocity
    self.xy_vels_grid[0, 10:17, 1:4, 1] = velocity
    self.xy_vels_grid[0, 15:18, 2:17, 0] = velocity
    self.xy_vels_grid[0, 2:17, 15:18, 1] = -velocity
    # self.xy_vels_grid[0, 21:24, 2:23, 0] = velocity
    # self.xy_vels_grid[0, 10:23, 1:4, 1] = velocity
    # self.xy_vels_grid[0, 9:12, 2:8, 0] = -velocity
    # self.xy_vels_grid[0, 10:19, 6:9, 1] = -velocity
    # self.xy_vels_grid[0, 17:20, 7:19, 0] = -velocity
    # self.xy_vels_grid[0, 14:19, 17:20, 1] = velocity

    # Curve 
    self.xy_vels_grid[0, 1:3, 2:5, 1] = velocity
    self.xy_vels_grid[0, 4:7, 1:3, 0] = velocity
    self.xy_vels_grid[0, 5:7, 9:12, 1] = velocity
    self.xy_vels_grid[0, 8:11, 11:13, 0] = -velocity
    self.xy_vels_grid[0, 8:11, 11:13, 0] = -velocity
    self.xy_vels_grid[0, 9:11, 2:5, 1] = velocity
    self.xy_vels_grid[0, 14:17, 1:3, ] = velocity
    self.xy_vels_grid[0, 16:18, 14:17, 1] = -velocity
    self.xy_vels_grid[0, 2:5, 16:18, 0] = -velocity
    # self.xy_vels_grid[0, 10:13, 7:9, 0] = -velocity
    # self.xy_vels_grid[0, 9:11, 2:5, 1] = velocity
    # self.xy_vels_grid[0, 20:23, 1:3, 0] = velocity
    # self.xy_vels_grid[0, 22:24, 20:23, 1] = -velocity
    # self.xy_vels_grid[0, 12, 21:24, :] = -velocity
    # self.xy_vels_grid[0, 11, 20:23, :] = -velocity
    # self.xy_vels_grid[0, 12, 21, 0] = 0
    # self.xy_vels_grid[0, 11, 20, 0] = 0
    # self.xy_vels_grid[0, 12, 23, 1] = 0
    # self.xy_vels_grid[0, 11, 22, 1] = 0

    # self.xy_vels_grid[0, 2:5, 20:22, 0] = -velocity

    # orientation
    self.orientation_map[0, 1:4, 2:17, 0] = -velocity
    self.orientation_map[0, 2:7, 1:4, 1] = velocity
    self.orientation_map[0, 5:8, 2:12, 0] = velocity
    self.orientation_map[0, 6:11, 10:13, 1] = velocity
    self.orientation_map[0, 9:12, 2:12, 0] = -velocity
    self.orientation_map[0, 10:17, 1:4, 1] = velocity
    self.orientation_map[0, 15:18, 2:17, 0] = velocity
    self.orientation_map[0, 2:17, 15:18, 1] = -velocity

    self.orientation_map[0, 1:3, 2:5, 1] = velocity
    self.orientation_map[0, 4:7, 1:3, 0] = velocity
    self.orientation_map[0, 5:7, 9:12, 1] = velocity
    self.orientation_map[0, 8:11, 11:13, 0] = -velocity
    self.orientation_map[0, 8:11, 11:13, 0] = -velocity
    self.orientation_map[0, 9:11, 2:5, 1] = velocity
    self.orientation_map[0, 14:17, 1:3, ] = velocity
    self.orientation_map[0, 16:18, 14:17, 1] = -velocity
    self.orientation_map[0, 2:5, 16:18, 0] = -velocity

    self.orientation_map[0, 15:18, 4:14, 0] = 0.
    self.orientation_map[0, 15:18, 4:14, 1] = -velocity


  def update_states(self, pwcs:PoseWithCovarianceStamped, odom: Odometry, joint: JointState):
    if not self.record:
      self.steps += 1

    # Get current robomaster position in vicon system
    current_position = np.array([pwcs.pose.pose.position.x, pwcs.pose.pose.position.y, pwcs.pose.pose.position.z])
    self.current_position = current_position - self.initial_position_offset  

    # Get current robomaster orientation in vicon system
    self.current_orientation = np.array([pwcs.pose.pose.orientation.x, pwcs.pose.pose.orientation.y, pwcs.pose.pose.orientation.z, pwcs.pose.pose.orientation.w])

    # Get velocities of robomaster in vicon system
    self.base_linear_velocities = np.array([odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z])
    self.base_angular_velocities = np.array([odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z])

    # Get gravity using vicon pose
    self.linear_acceleration = quat_rotate(self.current_orientation, self.gravity_vec)

    self.joint_states = joint

    # Place robomaster on map
    self.map_pos_x = self.current_position[0]
    self.map_pos_y = self.current_position[1]
    self.get_current_cells()

    linear_velocities, angular_velocity = self.get_target_velocity()
    # linear_velocities[0,0] = 0
    # linear_velocities[0,1] = 0
    if self.use_actions:
      self.wheel_velocities = self.get_actions(linear_velocities.flatten(), angular_velocity)
    else:
      target_velocity = np.stack((linear_velocities[0,0], linear_velocities[0,1], angular_velocity))
      # print(target_velocity)
      self.wheel_velocities = get_wheel_velocities(np.expand_dims(target_velocity, axis=-1), self.wheel_velocities_matrix).flatten() 
    self.clipped_velocities = self.wheel_velocities
    # print(self.clipped_velocities)
    
    # print(f'OBS: {np.hstack((self.base_linear_velocities[0], self.base_linear_velocities[1], self.base_angular_velocities[2]))}')
    
  def get_interpolated_velocities(self):
    # Interpolated velocities depending on positions.
    current_interpolated_pos_x_coeffs = np.abs(self.map_xy[self.current_y_cell, self.current_x_cell + 1, 0]
                                                  - self.map_pos_x) / (2 * self.cell_size)

    current_interpolated_pos_y_coeffs = np.abs(self.map_xy[self.current_y_cell + 1, self.current_x_cell, 1]
                                                  - self.map_pos_y) / (2 * self.cell_size)
    
    v_bottom_left = self.xy_vels_grid[self.envs_aranged, self.current_y_cell, self.current_x_cell, :] 
    v_bottom_right = self.xy_vels_grid[self.envs_aranged, self.current_y_cell, self.current_x_cell + 1, :]
    v_top_left = self.xy_vels_grid[self.envs_aranged, self.current_y_cell + 1, self.current_x_cell, :]
    v_top_right = self.xy_vels_grid[self.envs_aranged, self.current_y_cell + 1, self.current_x_cell + 1, :]   

    current_interpolated_vel = current_interpolated_pos_x_coeffs * current_interpolated_pos_y_coeffs * v_bottom_left \
                                + (1 - current_interpolated_pos_x_coeffs) * current_interpolated_pos_y_coeffs * v_bottom_right \
                                + current_interpolated_pos_x_coeffs * (1 - current_interpolated_pos_y_coeffs) * v_top_left \
                                + (1 - current_interpolated_pos_x_coeffs) * (1 - current_interpolated_pos_y_coeffs) * v_top_right
    
    return np.stack((current_interpolated_vel[0], current_interpolated_vel[1]), axis=0)
  
  def get_interpolated_orienations(self):
    # Interpolated velocities depending on positions.
    current_interpolated_pos_x_coeffs = np.abs(self.map_xy[self.current_y_cell, self.current_x_cell + 1, 0]
                                                  - self.map_pos_x) / (2 * self.cell_size)

    current_interpolated_pos_y_coeffs = np.abs(self.map_xy[self.current_y_cell + 1, self.current_x_cell, 1]
                                                  - self.map_pos_y) / (2 * self.cell_size)
    
    o_bottom_left = self.orientation_map[self.envs_aranged, self.current_y_cell, self.current_x_cell, :] 
    o_bottom_right = self.orientation_map[self.envs_aranged, self.current_y_cell, self.current_x_cell + 1, :]
    o_top_left = self.orientation_map[self.envs_aranged, self.current_y_cell + 1, self.current_x_cell, :]
    o_top_right = self.orientation_map[self.envs_aranged, self.current_y_cell + 1, self.current_x_cell + 1, :]   

    current_interpolated_ori = current_interpolated_pos_x_coeffs * current_interpolated_pos_y_coeffs * o_bottom_left \
                                + (1 - current_interpolated_pos_x_coeffs) * current_interpolated_pos_y_coeffs * o_bottom_right \
                                + current_interpolated_pos_x_coeffs * (1 - current_interpolated_pos_y_coeffs) * o_top_left \
                                + (1 - current_interpolated_pos_x_coeffs) * (1 - current_interpolated_pos_y_coeffs) * o_top_right
    
    return np.stack((current_interpolated_ori[0], current_interpolated_ori[1]), axis=0)

  def get_current_cells(self):
    # Reset when current cell == 1 or max_cells - 1. This prevents querying for fields out of velocity grid.
    self.current_x_cell = int((self.idx_offset + self.map_pos_x) / self.cell_size)
    self.current_y_cell = int((self.idx_offset + self.map_pos_y) / self.cell_size)
    self.current_cells = np.stack((self.current_x_cell, self.current_y_cell))

    is_out_of_map = (self.current_cells < 0).any() or (self.current_cells > self.xy_vels_grid.shape[1]).any()
    if is_out_of_map:

      raise IndexError(f'Robomaster is out of bounds! {(self.current_x_cell, self.current_y_cell)} is not accessable in map.')

  def get_angular_velocity(self, Vx, Vy):
    # Calculate the angle between the x-axis and the desired velocity vector

    heading = np.arctan2(Vy, Vx)

    # Calculate the overall angular velocity to align the robot with the desired velocity
    theta = np.clip(3*wrap_to_pi(heading), -self.max_angular_velocity, self.max_angular_velocity)
    return theta

  def get_target_velocity(self):
    if self.use_interpolated:
      target_velocity = self.get_interpolated_velocities()
      target_orientation = self.get_interpolated_orienations()
    else:
      target_velocity = self.xy_vels_grid[0, self.current_y_cell, self.current_x_cell]
      target_orientation = self.orientation_map[0, self.current_y_cell, self.current_x_cell]
    
    # print(f'Target Velocity0: {target_velocity}')
    _, _, orientation_z = get_euler_xyz(self.current_orientation[np.newaxis, ...])
    target_velocity = self.vel_vector_rotation_z(target_velocity[0].reshape(-1, 1),
                                                target_velocity[1].reshape(-1, 1),
                                                -orientation_z)
    # print(f'Target Velocity1: {target_velocity}')
    # print(f'Target Orientation1: {target_orientation}')
    target_orientation = self.vel_vector_rotation_z(target_orientation[0].reshape(-1, 1),
                                                    target_orientation[1].reshape(-1, 1),
                                                    -orientation_z)
      
    # print(f'Target Velocity2: {target_velocity}')
    # print(f'Target Orientation2: {target_orientation}')
    # angular_velocity = self.get_angular_velocity(target_velocity[0,0], target_velocity[0,1])
    angular_velocity = self.get_angular_velocity(target_orientation[0,0], target_orientation[0, 1])
    
    return target_velocity, angular_velocity
  
  def get_actions(self, linear_velocity, angular_velocity):
    # Push action and observations to history queue
    self.push_to_action_queue(self.actions, self.obs)
    self.push_to_racetrack_action_queue(self.racetrack_actions, self.racetrack_obs)
    # Build observation
    commands = np.hstack((linear_velocity[0], linear_velocity[1], angular_velocity))
    # print(np.flip(self.current_cells))

    
    self.racetrack_obs = np.hstack([self.base_linear_velocities[:2], self.base_angular_velocities[2], self.linear_acceleration, np.flip(self.current_cells.T), commands, self.racetrack_actions.squeeze()])
    self.racetrack_obs = torch.tensor(self.racetrack_obs, device=self.device, dtype=torch.float).unsqueeze(0)
    print(self.racetrack_obs)
    racetrack_obs = self.racetrack_obs
    racetrack_input_dict = {
      'is_train' : False,
      'prev_actions' : None,
      'obs' : racetrack_obs,
    }

    with torch.no_grad():
      res_dict = self.racetrack_model(racetrack_input_dict)
    self.racetrack_actions = res_dict['mus'].clamp(-1, 1).cpu().detach().numpy()
    commands = self.racetrack_actions * np.array([self.max_linear_velocity, self.max_linear_velocity, self.max_angular_velocity])

    print(commands)
    self.obs = np.hstack([self.base_linear_velocities[:2], self.base_angular_velocities[2], self.linear_acceleration, commands.squeeze(), self.actions.squeeze()])
    self.obs = torch.tensor(self.obs, device=self.device, dtype=torch.float).unsqueeze(0)

    obs = {'Obs': self.obs, 'Past_obs': self.history_queue}

    input_dict = {
      'is_train' : False,
      'prev_actions' : None,
      'obs' : obs,
    }

    with torch.no_grad():
      res_dict = self.model(input_dict)
    self.actions = res_dict['mus'].clamp(-1, 1).cpu().detach().numpy()
    scaled_actions = self.actions * 70
    return scaled_actions.squeeze()
    

  def post_publish(self):
    # Increment progress buffer (episode length counter)
    self.progress_buf += 1
    env_ids = self.reset_buf.nonzero(as_tuple=False)
    if len(env_ids) > 0:
        # Reset if out of bounds or episode complete
        self.reset(env_ids) 
    self.examine_task()
    task_env_ids = self.task_buf.nonzero(as_tuple=False)
    if len(task_env_ids) > 0:
        # Reset if Robomaster is stationary
        self.reset(task_env_ids)
        self.task_buf[task_env_ids] = 0.0
    self.compute_reward()


  def examine_task(self):
    # Check if Robomaster is stationary
    m = (np.abs(self.linear_velocities) < self.min_velocity).all()
    m = torch.tensor(m, dtype=torch.bool, device = self.device)
    self.task_buf[m] = 1.0


  def compute_reward(self): 
    projected_gravity = quat_rotate(self.current_orientation, self.gravity_vec_rm_proj)
    projected_gravity[[1, 2]] = projected_gravity[[2, 1]] 
    base_lin_velocity = quat_rotate_inverse(self.current_orientation, self.base_linear_velocities)
    current_cells = torch.tensor(self.current_cells, device=self.device)
    boundary_reached = torch.tensor(self.boundary_reached, dtype=torch.bool, device=self.device)

    self.rew_buf, self.reset_buf[:] = compute_robomaster_reward(self.linear_velocities, base_lin_velocity, current_cells, projected_gravity,
        self.gravity_vec, self.forward_vec,
        self.reset_buf,
        self.progress_buf,
        self.max_episode_length, 
        self.max_cells, self.boundary_size, boundary_reached)


  def reset(self, env_ids):
    # Reset velocity field and buffers
    self.reset_velocity_field(env_ids)
    self.history_pointer[env_ids] = 0
    self.history_queue[env_ids] = 0
    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0


  def reset_velocity_field(self, rm_idx):
    for idx in rm_idx:
      if self.reset_position:        
        start = 0
        end = self.xy_vels_grid.shape[1]
        self.velocity_field_functions.reset = True
        f = self.velocity_field_functions[idx]
        vel_x, vel_y = f(self.x, self.y)
        self.reset_position = False
        self.boundary_reached = True
      else:
        start = 0
        end = self.xy_vels_grid.shape[1]
        self.xy_vels_grid = self.xy_vels_grid_reset
        scale = np.random.uniform(0.5, 1.0, 2)
        self.velocity_field_functions.reset = False
        f = self.velocity_field_functions[idx]
        vel_x, vel_y = f(self.x[start:end, start:end], self.y[start:end, start:end], scale)
        self.reset_position = True
        self.boundary_reached = False
    
    self.xy_vels_grid[0, start:end, start:end, 0] = vel_x 
    self.xy_vels_grid[0, start:end, start:end, 1] = vel_y 
    self.xy_vels_grid = self.xy_vels_grid.clip(-self.max_linear_velocity, self.max_linear_velocity)
    self.set_boundary_velocities(start, end)
    if self.racetrack:
      self.generate_racetrack()

  def create_boundary_grid(self, start, end):
    # Get the center of the grid
    center_x = (start + end) // 2
    center_y = (start + end) // 2

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(start, end), np.arange(start, end))

    # Calculate direction towards the center
    direction_x = center_x - x
    direction_y = center_y - y

    # Convert direction to float64 and clip values in border
    direction_x = np.clip(direction_x.astype(np.float64), -center_x + self.boundary_size, center_x - self.boundary_size)
    direction_y = np.clip(direction_y.astype(np.float64), -center_y + self.boundary_size, center_y - self.boundary_size)

    # Calculate the distance from each point to the center and clip values in border
    distance = np.clip(np.sqrt(direction_x**2 + direction_y**2), center_x + self.boundary_size, center_x - self.boundary_size)

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    distance[distance < epsilon] = epsilon

    # Normalize the direction vector
    direction_x /= distance
    direction_y /= distance
    
    return np.concatenate((np.expand_dims(direction_x, axis=-1), np.expand_dims(direction_y, axis=-1)), axis=-1)

  def set_boundary_velocities(self, start, end):
    # Set velocities based on the direction
    self.xy_vels_grid[0, start:self.boundary_size+1, start:end, 1] = self.boundary_grid[start:self.boundary_size+1, start:end, 1] * self.max_linear_velocity 
    self.xy_vels_grid[0, end-(self.boundary_size+1):end, start:end, 1] = self.boundary_grid[end-(self.boundary_size+1):end, start:end, 1] * self.max_linear_velocity 
    self.xy_vels_grid[0, start:end, start:self.boundary_size+1 , 1] = self.boundary_grid[start:end, start:self.boundary_size+1, 1] * self.max_linear_velocity 
    self.xy_vels_grid[0, start:end, end-(self.boundary_size+1):end, 1] = self.boundary_grid[ start:end, end-(self.boundary_size+1):end, 1] * self.max_linear_velocity 

    self.xy_vels_grid[0, start:self.boundary_size+1, start:end, 0] = self.boundary_grid[start:self.boundary_size+1, start:end, 0] * self.max_linear_velocity 
    self.xy_vels_grid[0, end-(self.boundary_size+1):end, start:end, 0] = self.boundary_grid[end-(self.boundary_size+1):end, start:end, 0] * self.max_linear_velocity 
    self.xy_vels_grid[0, start:end, start:self.boundary_size+1, 0] = self.boundary_grid[start:end, start:self.boundary_size+1, 0] * self.max_linear_velocity 
    self.xy_vels_grid[0, start:end, end-(self.boundary_size+1):end, 0] = self.boundary_grid[start:end, end-(self.boundary_size+1):end, 0] * self.max_linear_velocity 

  def visualize_map(self, grid_velocity):
    # Plotting map of robomaster and velocity fields
    if self.visualize:  
      last_shuffle = self.velocity_field_functions.get_last_shuffle()
      pos_x = self.map_pos_x
      pos_y = self.map_pos_y

      map_x = self.map_xy[..., 0] + self.cell_size / 2.0
      map_y = self.map_xy[..., 1] + self.cell_size / 2.0
      vel_x = self.xy_vels_grid[0, :, :, 0]
      vel_y = self.xy_vels_grid[0, :, :, 1]
      # Plot robomaster position 
      self.ax.scatter(pos_x, pos_y)
      # Calculate orientation of robomaster in map
      _, _, orientation_z = get_euler_xyz(self.current_orientation[np.newaxis, ...])
      # Calculate x and y axis in robomaster frame
      robot_x = self.vel_vector_rotation_z(np.array([1]).reshape(-1, 1), np.array([0]).reshape(-1, 1), orientation_z)
      robot_y = self.vel_vector_rotation_z(np.array([0]).reshape(-1, 1), np.array([1]).reshape(-1, 1), orientation_z)
      # Plot arrows
      self.ax.quiver(pos_x, pos_y, robot_x[0,0], robot_x[0,1], color='r') # x-axis
      self.ax.quiver(pos_x, pos_y, robot_y[0,0], robot_y[0,1], color='g') # y-axis
      actions = self.actions.cpu().detach().numpy()
      rot_actions = self.vel_vector_rotation_z(actions[:, 1].reshape(-1, 1), actions[:, 0].reshape(-1, 1), -actions[:, 2])
      self.ax.quiver(pos_x, pos_y, rot_actions[..., 0], rot_actions[..., 1], color='b') # actions
      self.ax.quiver(pos_x, pos_y, grid_velocity[..., 0], grid_velocity[..., 1]) # grid-velocity
      rot_base_vel = self.vel_vector_rotation_z(self.base_linear_velocities[0].reshape(-1, 1), self.base_linear_velocities[1].reshape(-1, 1), -self.base_angular_velocities[2])
      self.ax.quiver(pos_x, pos_y, rot_base_vel[..., 0], rot_base_vel[..., 1], color='y')
      self.ax.quiver(map_x, map_y, vel_x, vel_y)
      self.ax.set_title("Field: " + last_shuffle[0])

      plt.show()
      plt.pause(0.001)
      plt.cla()   
  
  def publish_velocities(self):
    if self.use_wheel_vel or self.use_actions:
      msg = JointState()
      msg.header = self.joint_states.header
      msg.name = self.joint_states.name
      msg.position = self.joint_states.position

      # FR, FL, RR, RL
      msg.velocity = [float(self.clipped_velocities[0]), float(self.clipped_velocities[1]), float(self.clipped_velocities[2]), float(self.clipped_velocities[3])]
      
      if self.publish:
        self.velocity_publisher.publish(msg) 
      
      if self.record and self.steps <= self.recording_threshold:
        self.writer.write(f'/{self.robomaster}/cmd_wheel_speed_bag', serialize_message(msg), self.get_clock().now().nanoseconds)
    else:
      linear_vel, angular_vel = self.get_target_velocity()
      msg = Twist()
      msg.linear.x = linear_vel[0][0]
      msg.linear.y = linear_vel[0][1]
      msg.angular.z = angular_vel

      if self.publish:
        self.velocity_publisher.publish(msg)
    self.steps += 1

  def vel_vector_rotation_z(self, vel_x, vel_y, orientation_z):
    self.rot_matrix_z[:] = 0
    c = np.cos(orientation_z)
    s = np.sin(orientation_z)
    self.rot_matrix_z[:, 0, 0] = c
    self.rot_matrix_z[:, 0, 1] = -s
    self.rot_matrix_z[:, 1, 0] = s
    self.rot_matrix_z[:, 1, 1] = c
    vel_tensor = np.concatenate((vel_x, vel_y), axis=1).reshape(-1, 1)
    rotation = np.matmul(self.rot_matrix_z, vel_tensor) # batch matrix multiplication *(check in numpy)
    rotation = np.squeeze(rotation, axis=2)
    return rotation
  
  def push_to_action_queue(self, last_actions, last_observation):
    last_actions = torch.tensor(last_actions, device=self.device)
    obs_action = torch.cat((last_observation, last_actions), dim=1)
    overflow_mask = self.history_pointer >= self.history_size
    if overflow_mask.any():
        self.history_queue[overflow_mask, :-1] = self.history_queue[overflow_mask, 1:]
    self.history_pointer = self.history_pointer.clamp(0, self.history_size - 1)
    self.history_queue[self.envs_aranged.item(), self.history_pointer.item()] = obs_action
    self.history_pointer += 1

  def push_to_racetrack_action_queue(self, last_actions, last_observation):
    last_actions = torch.tensor(last_actions, device=self.device)
    obs_action = torch.cat((last_observation, last_actions), dim=1)
    overflow_mask = self.racetrack_history_pointer >= self.racetrack_history_size
    if overflow_mask.any():
        self.racetrack_history_queue[overflow_mask, :-1] = self.racetrack_history_queue[overflow_mask, 1:]
    self.racetrack_history_pointer = self.racetrack_history_pointer.clamp(0, self.history_size - 1)
    self.racetrack_history_queue[self.envs_aranged.item(), self.racetrack_history_pointer.item()] = obs_action
    self.racetrack_history_pointer += 1

def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

def get_wheel_velocities(current_velocities, wheel_velocities_matrix):
  command_wheel_velocities = np.matmul(wheel_velocities_matrix, current_velocities)
  return command_wheel_velocities.transpose(1,0)

def main(rm_args=None, args=None):  
  rclpy.init(args=args)
  
  target_vels_publisher = TargetVelsPublisher(args=rm_args)
  
  rclpy.spin(target_vels_publisher)
  
  target_vels_publisher.destroy_node()
  
  rclpy.shutdown()
  
if __name__ == '__main__':
  args = get_args()
  args = parse_args(args)
  main(rm_args=args)