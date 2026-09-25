import numpy as np
import torch
import yaml
import os

import rclpy
import rosbag2_py
import threading

from rclpy.node import Node
from rclpy.serialization import serialize_message, deserialize_message
from rosidl_runtime_py.utilities import get_message

from geometry_msgs.msg import  PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import  JointState
from message_filters import Subscriber, ApproximateTimeSynchronizer

from rl_games.algos_torch import model_builder, torch_ext

from components.utils import *
from components.velocity_field import VelocityField
from components.networks import RMANetworkBuilder, MLPNetworkBuilder, MLPWithRNNNetworkBuilder, RMAWithTCNNetworkBuilder
from components.config import *
from components.frame_stack import FrameStack

from pynput import keyboard
from data import get_rosbag_options

class Controller(Node):

# 1. IP Robomaster: Am Vicon-PC checken -> http://10.65.80.1/
# 2. In der Liste nachschauen nach robomaster: ep01 ....
# 3. Name: robo password: robo

# Potential Debugs
# 1. Disable nav_deep on Robomaster -> sudo systemctl disable nav_deep_start.service
# Bug: random index out of bounds error (difficult to reproduce)

  def __init__(self, args):
    super().__init__('controler')
    with open(os.path.join(os.getcwd(), 'rm_publisher/config/rm_publisher.yaml'), 'r') as f:
      cfg = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), 'rm_publisher/config/rm_network.yaml'), 'r') as f:
      cfg_train = yaml.load(f, Loader=yaml.SafeLoader)['params']
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
    self.single_obs_size = cfg['model']['obsSize'] # Number of observations
    self.frame_stack = cfg["model"]["frameStack"]
    self.history_size = cfg['model']['actionObsHistory'] # Number of saved observations
    self.obs_size = [self.single_obs_size * self.frame_stack, (self.history_size, self.single_obs_size + self.action_size)]

    # Allocate buffers
    self.rew_buf = torch.zeros(self.num_rm, device=self.device, dtype=torch.float)
    self.reset_buf = torch.zeros(self.num_rm, device=self.device, dtype=torch.long)
    self.progress_buf = torch.zeros(self.num_rm, device=self.device, dtype=torch.long)
    self.task_buf = torch.zeros(self.num_rm, device=self.device, dtype=torch.long)
    self.envs_aranged = torch.arange(0, self.num_rm, device=self.device, dtype=torch.long)

    
    self.max_episode_length = cfg['map']['episodeLength'] # Max length of episode before velocity field resets

    self.use_interpolated = args.get('interpolated', cfg['map']['useInterpolated'])
    self.use_actions = cfg['robomaster']['use_actions']
    self.playback_cmd = cfg['robomaster']['playback_cmd']
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
    self.record_normal = cfg['data']['record_normal']
    self.record_bag = cfg['data']['record_bag']
    self.recording_threshold = cfg['data']['steps']
    self.bag = cfg['data']['bag'] 
    self.publish = cfg['robomaster']['publish']

    self.robomaster = args.get('robomaster', cfg['robomaster']['name']) # Name of robomaster

    config = {
      'actions_num' : self.action_size,
      'input_shape' : {'Obs': (self.single_obs_size * self.frame_stack,), 'Past_obs': (self.history_size, self.single_obs_size + self.action_size)},
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

    self.history_queue = torch.zeros((self.num_rm, self.history_size, self.action_size + self.single_obs_size)
                                     ,dtype=torch.float, device=self.device)
    self.history_pointer = torch.zeros((self.num_rm,), dtype=torch.long, device=self.device)

    self.actions = np.zeros((self.num_rm, self.action_size ))
    self.obs = torch.zeros((self.num_rm, self.single_obs_size ), device=self.device)

    # Initialize gravity and forward vectors
    self.gravity_vec = get_axis_params(-1, 2)
    self.gravity_vec_rm_proj = get_axis_params(-1., 1)
    self.forward_vec = np.array([1, 0, 0])
    self.heading_direction = np.zeros(1)

    self.current_orientation = np.zeros(4)
    self.current_position = np.zeros(3)
    self.linear_velocities = np.zeros(2)
    self.angular_velocity = np.zeros(1)
    self.wheel_velocities = np.zeros(4)
    self.clipped_velocities = np.zeros(4)
    self.forward_vec = np.array([1., 0., 0.])
    # Initialize Velocity fields
    self.velocity_field_functions = VelocityField(self.num_rm, self.visualize)
    
    self.rot_matrix_z = np.zeros((self.num_rm, 2, 2), dtype=float)

    self.steps = 0

    self.wheel_seperation_width = 0.40
    self.wheel_seperation_length = 0.40
    self.wheel_radius = 0.05
    self.wheel_velocities_matrix = np.ones((4, 3))
    self.wheel_velocities_matrix[1, 1] = self.wheel_velocities_matrix[2, 1] = -1
    self.wheel_velocities_matrix[2, 2] = self.wheel_velocities_matrix[0, 2] = (self.wheel_seperation_width + self.wheel_seperation_length)
    self.wheel_velocities_matrix[3, 2] = self.wheel_velocities_matrix[1, 2] = -(self.wheel_seperation_width + self.wheel_seperation_length)
    self.wheel_velocities_matrix = (1 / self.wheel_radius) * self.wheel_velocities_matrix
    
    if self.frame_stack != 1:
        self.observation_stack = FrameStack(self.num_rm, self.frame_stack, self.single_obs_size, self.device)
    self.history = FrameStack(self.num_rm, self.history_size, self.single_obs_size + self.action_size, self.device)

    self.current_cells = np.zeros(2)
    self.base_linear_velocities = np.zeros(3)
    self.base_angular_velocities = np.zeros(3)
    self.target_velocities = np.zeros((self.num_rm, 2))
    self.commands = np.zeros(3)
    self.joint_states = JointState()

    if self.record_normal:
      self.actions_history = np.load('obs_action_2500_samples.npz', allow_pickle=True)['actions']
      self.observations_history = np.load('obs_action_500_samples.npz', allow_pickle=True)['observations']
    # self.test = np.load('real_obs_action.npz', allow_pickle=True)
      self.obs_history = []
    
    if self.record_bag:
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
        {'name': f'/{self.robomaster}/cmd_vel_bag', 'type': 'geometry_msgs/msg/Twist'},
      ]

      for topic in topics:
        topic_info = rosbag2_py._storage.TopicMetadata(
            name=topic['name'],
            type=topic['type'],
            serialization_format='cdr'
        )
        self.writer.create_topic(topic_info)

    if self.playback_cmd:
      self.cmd_velocities = np.load('data-10/data-0.npz', allow_pickle=True)['cmd_vels']

    if self.bag:
      self.record_timer = ApproximateTimeSynchronizer((Subscriber(self, PoseWithCovarianceStamped, f'/{self.robomaster}/vicon_pose_bag'), Subscriber(self, Odometry, f'/{self.robomaster}/vicon_odom_bag'), Subscriber(self, JointState, f'/{self.robomaster}/joint_states_bag')), 10, 0.1)
    else:
      self.record_timer = ApproximateTimeSynchronizer((Subscriber(self, PoseWithCovarianceStamped, f'/{self.robomaster}/vicon_pose'), Subscriber(self, Odometry, f'/{self.robomaster}/vicon_odom'), Subscriber(self, JointState, f'/{self.robomaster}/joint_states')), 10, 0.1)
    self.record_timer.registerCallback(self.recorder)

    # Adjust the desired publishing rate (e.g., 10 Hz)
    self.publishing_rate = 50.0

    self.velocity_publisher = self.create_publisher(JointState, f'/{self.robomaster}/cmd_wheel_speed', 10)

    self.reset_position = True

  def recorder(self, pwcs:PoseWithCovarianceStamped, odom:Odometry, joint:JointState):
    if self.steps == self.recording_threshold:
      raise NotImplementedError
    # print('record')
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

    if self.playback_cmd:
      self.commands = self.cmd_velocities[self.steps]
    else:
      forward = quat_apply(self.current_orientation, self.forward_vec)
      heading = np.arctan2(forward[1], forward[0])
      self.angular_velocity = np.clip(0.5*wrap_to_pi(self.heading_direction - heading), -1., 1.)

    if not self.record_normal and not self.playback_cmd:
      commands = np.hstack((self.linear_velocities.flatten()[0], self.linear_velocities[1], self.angular_velocity))
    else:
      commands = self.commands
    
    print(commands)

    if self.record_bag and self.steps <= self.recording_threshold:
        twist = Twist()
        twist.linear.x = commands[0]
        twist.linear.y = commands[1]
        twist.angular.z = commands[2]
        print(commands)
        self.writer.write(f'/{self.robomaster}/cmd_vel_bag', serialize_message(twist), self.get_clock().now().nanoseconds)

    if self.use_actions:
      self.wheel_velocities = self.get_actions(commands)
    else:
      self.wheel_velocities = get_wheel_velocities(np.expand_dims(commands, axis=-1), self.wheel_velocities_matrix).flatten()
    print(self.wheel_velocities)
    self.clipped_velocities = self.wheel_velocities
    
    if self.record_bag and self.steps <= self.recording_threshold:
        self.writer.write(f'/{self.robomaster}/joint_states_bag', serialize_message(joint), self.get_clock().now().nanoseconds)
        self.writer.write(f'/{self.robomaster}/vicon_pose_bag', serialize_message(pwcs), self.get_clock().now().nanoseconds)
        self.writer.write(f'/{self.robomaster}/vicon_odom_bag', serialize_message(odom), self.get_clock().now().nanoseconds)
    
    self.publish_velocities()

  def get_actions(self, commands):
    # Build observation
    if self.record_normal:
      current_obs = np.hstack([self.base_linear_velocities[:2], self.base_angular_velocities[2], self.linear_acceleration, commands, self.actions.squeeze()*70])
    else:
      current_obs = np.hstack([self.base_linear_velocities[:2], self.base_angular_velocities[2], self.linear_acceleration, commands, self.actions.squeeze()])

    if self.frame_stack != 1:
        self.observation_stack.add_obs_history(torch.tensor(current_obs, device=self.device, dtype=torch.float))
        self.obs = self.observation_stack.return_obs(flatten=True)
    else:
        self.obs = current_obs

    self.history_queue = self.history.return_obs()
    if self.steps >= 499 and self.record_normal:
      np.savez('real_obs_action.npz', obs=self.obs_history, actions=self.actions_history)
    
    self.obs = torch.tensor(self.obs, device=self.device, dtype=torch.float).unsqueeze(0)
    if self.record_normal:
      self.obs_history.append(self.obs.cpu().numpy())

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
    print(scaled_actions)
    return scaled_actions.squeeze()

  def publish_velocities(self):
    msg = JointState()
    msg.header = self.joint_states.header
    msg.name = self.joint_states.name
    msg.position = self.joint_states.position
    
    if self.record_normal:
      self.commands = self.observations_history[self.steps, 6:9]

    # FR, FL, RR, RL  
    msg.velocity = [float(self.clipped_velocities[0]), float(self.clipped_velocities[1]), float(self.clipped_velocities[2]), float(self.clipped_velocities[3])] 
    
    if self.publish:
      self.velocity_publisher.publish(msg) 
 
    if self.record_bag and self.steps <= self.recording_threshold:
      self.writer.write(f'/{self.robomaster}/cmd_wheel_speed_bag', serialize_message(msg), self.get_clock().now().nanoseconds)
    self.steps += 1
    print(self.steps)

  def on_press(self, key): 

    if key == keyboard.Key.esc:
        return False  # stop listener
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys
    if k == 'w':
      self.linear_velocities[0] = min(self.lin_vel_x_scale, self.linear_velocities[0] + 0.1)
    if k == 's':
      self.linear_velocities[0] = max(-self.lin_vel_x_scale, self.linear_velocities[0] - 0.1)
    if k == 'a':
      self.linear_velocities[1] = min(self.lin_vel_y_scale, self.linear_velocities[1] + 0.1)
    if k == 'd':
      self.linear_velocities[1] = max(-self.lin_vel_y_scale, self.linear_velocities[1] - 0.1)
    if k == 'q':
      self.heading_direction = ((self.heading_direction + np.pi / 8) % (np.sign(self.heading_direction + np.pi / 8 + 1e-8) * 2 * np.pi))
    if k == 'e':
      self.heading_direction = ((self.heading_direction - np.pi / 8) % (np.sign(self.heading_direction - np.pi / 8 - 1e-8) * 2 * np.pi))
  
  def on_release(self, key):
    try:
      if key.char == 'w':
        self.linear_velocities[0] = 0
      if key.char == 's':
        self.linear_velocities[0] = 0
      if key.char == 'a':
        self.linear_velocities[1] = 0
      if key.char == 'd':
        self.linear_velocities[1] = 0
    except AttributeError:
      pass

def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

def get_wheel_velocities(current_velocities, wheel_velocities_matrix):
  command_wheel_velocities = np.matmul(wheel_velocities_matrix, current_velocities)
  return command_wheel_velocities.transpose(1,0)
  

def main(rm_args=None, args=None):
    rclpy.init(args=args)

    # Initialize the controller node
    target_vels_publisher = Controller(args=rm_args)

    # Start the ROS2 spinning loop in a separate thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(target_vels_publisher,))
    ros_thread.start()

    # Start the keyboard listener in its own thread
    listener = keyboard.Listener(on_press=target_vels_publisher.on_press, on_release=target_vels_publisher.on_release)
    listener.start()

    try:
        # Keep the main thread alive while ROS2 and listener threads are running
        ros_thread.join()
    finally:
        # Ensure the listener is stopped before shutdown
        listener.stop()
        listener.join()  # Wait for listener thread to finish

        # Clean up ROS2 resources
        target_vels_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
  args = get_args()
  args = parse_args(args)
  main(rm_args=args)
