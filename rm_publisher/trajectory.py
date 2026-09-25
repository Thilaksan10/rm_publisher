import numpy as np
import torch
import yaml
import os
import math

import rclpy
# import rosbag2_py
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

import time

# from pynput import keyboard
# from data import get_rosbag_options


class Trajectory(Node):

# 1. IP Robomaster: Am Vicon-PC checken -> http://10.65.80.1/
# 2. In der Liste nachschauen nach robomaster: ep01 ....
# 3. Name: robo password: robo

# Potential Debugs
# 1. Disable nav_deep on Robomaster -> sudo systemctl disable nav_deep_start.service
# Bug: random index out of bounds error (difficult to reproduce)
# ros2 launch robomaster_bringup vicon.launch.py
# source ~/venvs/robomaster_gpu/bin/activate

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
        self.obs_size = self.single_obs_size * self.frame_stack

        self.rew_buf = torch.zeros(self.num_rm, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_rm, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_rm, device=self.device, dtype=torch.long)
        self.task_buf = torch.zeros(self.num_rm, device=self.device, dtype=torch.long)
        self.envs_aranged = torch.arange(0, self.num_rm, device=self.device, dtype=torch.long)

        self.max_episode_length = cfg['map']['episodeLength'] # Max length of episode before velocity field resets

        self.use_interpolated = args.get('interpolated', cfg['map']['useInterpolated'])
        self.use_actions = cfg['robomaster']['use_actions']
        self.is_offset = cfg['robomaster']['offset']
        self.follow_trajectory = cfg['robomaster']['trajectory']
        self.playback_cmd = cfg['robomaster']['playback_cmd']
        self.max_linear_velocity = cfg['robomaster']['maxLinearVelocity']
        self.max_angular_velocity = cfg['robomaster']['maxAngularVelocity']
        self.max_velocity = cfg['robomaster']['maxWheel']
        self.min_velocity = cfg['robomaster']['minVelocity']
        self.action_scale = cfg['robomaster']['actionScale']
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
            'input_shape' : (self.single_obs_size * self.frame_stack,),
            'num_seqs' : 1,
            'value_size' : 1,
            'normalize_value' : cfg_train['config'].get('normalize_value', True),
            'normalize_input' : cfg_train['config'].get('normalize_input', False),
        }

        self.model = builder.load(cfg_train).build(config)
        self.model.to(self.device)
        self.model.eval()

        checkpoint = torch_ext.load_checkpoint(os.path.join(os.getcwd(),  cfg['model']['path']))
        # self.model.load_state_dict(checkpoint['model'])
        missing, unexpected = self.model.load_state_dict(checkpoint["model"], strict=True)


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

        self.wheel_seperation_width = 0.1
        self.wheel_seperation_length = 0.1
        self.wheel_radius = 0.05
        self.wheel_velocities_matrix = np.ones((4, 3))
        self.wheel_velocities_matrix[1, 1] = self.wheel_velocities_matrix[2, 1] = -1
        self.wheel_velocities_matrix[2, 2] = self.wheel_velocities_matrix[0, 2] = (self.wheel_seperation_width + self.wheel_seperation_length)
        self.wheel_velocities_matrix[3, 2] = self.wheel_velocities_matrix[1, 2] = -(self.wheel_seperation_width + self.wheel_seperation_length)
        self.wheel_velocities_matrix = (1 / self.wheel_radius) * self.wheel_velocities_matrix

        if self.frame_stack != 1:
            self.observation_stack = FrameStack(self.num_rm, self.frame_stack, self.single_obs_size, self.device)

        if self.follow_trajectory:
            self.trajectory_paths = [
                'trajectories/changing_velocity_profiles.csv',
                'trajectories/circle_small_radius.csv',
                'trajectories/connected_u_curves_2.csv',
                'trajectories/connected_u_curves_5.csv',
                'trajectories/curve_left_180.csv',
                'trajectories/zigzag_lateral_motion.csv',
                'trajectories/forward_right_trajectory.csv',
            ]

            # Choose which trajectory to run.
            # Change this index for now, or connect it to your argument parser later.
            self.trajectory_index = 3

            self.trajectory_path = self.trajectory_paths[self.trajectory_index]

            self.trajectory_steps = 100

            self.traj_commands = np.loadtxt(
                self.trajectory_path,
                delimiter=',',
                skiprows=1
            )[:self.trajectory_steps, :3] * self.max_linear_velocity

            # Name without directory and extension
            self.trajectory_name = os.path.splitext(
                os.path.basename(self.trajectory_path)
            )[0]

            # Determine controller name
            if self.use_actions:
                self.controller_name = os.path.splitext(
                    os.path.basename(cfg['model']['path'])
                )[0]
            else:
                self.controller_name = "ik"

            # Create directory for this run
            self.run_directory = self.create_run_directory()

            print(f"Trajectory: {self.trajectory_name}")
            print(f"Controller: {self.controller_name}")
            print(f"Recording to: {self.run_directory}")


        self.current_cells = np.zeros(2)
        self.base_linear_velocities = np.zeros(3)
        self.base_angular_velocities = np.zeros(3)
        self.prev_base_linear_velocities = np.zeros(3)
        self.prev_base_angular_velocities = np.zeros(3)
        self.target_velocities = np.zeros((self.num_rm, 2))
        self.commands = np.zeros(3)
        self.joint_states = JointState()

        self.record_timer = ApproximateTimeSynchronizer((Subscriber(self, PoseWithCovarianceStamped, f'/{self.robomaster}/vicon_pose'), Subscriber(self, Odometry, f'/{self.robomaster}/vicon_odom'), Subscriber(self, JointState, f'/{self.robomaster}/joint_states')), 10, 0.1)
        self.record_timer.registerCallback(self.recorder)

        # Adjust the desired publishing rate (e.g., 10 Hz)
        self.publishing_rate = 50.0

        self.velocity_publisher = self.create_publisher(JointState, f'/{self.robomaster}/cmd_wheel_speed', 10)

        self.reset_position = True

        self.logged_positions = []        # actual positions from Vicon
        self.logged_velocities = []
        self.logged_cmds = []

        self.last_callback_time = None
        self.callback_dts = []

    def compute_reference_trajectory(self, start_position, start_heading, traj_commands, dt):
        """
        start_position : np.array([x, y, z])  # current Vicon position
        start_heading  : float                # current yaw in radians
        traj_commands  : np.array(N, 3)       # [vx, vy, wz] for each step
        dt             : float                # timestep per command
        """
        ref_positions = []
        ref_position = np.array(start_position, dtype=float)
        heading = start_heading

        for cmd in traj_commands:
            vx, vy, wz = cmd  # commanded velocities

            # Update heading
            heading += wz * dt

            # Rotate (vx, vy) into world frame
            dx = vx * np.cos(heading) - vy * np.sin(heading)
            dy = vx * np.sin(heading) + vy * np.cos(heading)

            # Integrate position
            ref_position = ref_position + np.array([dx, dy, 0.0]) * dt
            ref_positions.append(ref_position.copy())

        return np.array(ref_positions)
    

    def recorder(self, pwcs:PoseWithCovarianceStamped, odom:Odometry, joint:JointState):
        now = time.perf_counter()

        if self.last_callback_time is not None:
            dt_real = now - self.last_callback_time
            self.callback_dts.append(dt_real)

            if len(self.callback_dts) >= 50:
                mean_dt = np.mean(self.callback_dts[-50:])
                print(
                    f"CONTROL RATE: "
                    f"{1.0 / mean_dt:.2f} Hz "
                    f"(dt={mean_dt:.5f}s)"
                )

        self.last_callback_time = now

        t_pose = stamp_to_sec(pwcs)
        t_odom = stamp_to_sec(odom)
        t_joint = stamp_to_sec(joint)

        times = np.array([
            t_pose,
            t_odom,
            t_joint
        ])

        sync_diff = np.max(times) - np.min(times)

        print(
            f"SYNC DIFF: {sync_diff * 1000:.2f} ms "
            f"| pose-odom: {abs(t_pose - t_odom) * 1000:.2f} ms "
            f"| pose-joint: {abs(t_pose - t_joint) * 1000:.2f} ms"
        )
        if self.steps == len(self.traj_commands):

            np.savetxt(
                os.path.join(self.run_directory, "actual_positions.csv"),
                np.array(self.logged_positions),
                delimiter=","
            )

            np.savetxt(
                os.path.join(self.run_directory, "actual_velocities.csv"),
                np.array(self.logged_velocities),
                delimiter=","
            )

            np.savetxt(
                os.path.join(self.run_directory, "reference_velocities.csv"),
                self.traj_commands,
                delimiter=","
            )

            print("\n====================================")
            print("Trajectory finished")
            print(f"Trajectory : {self.trajectory_name}")
            print(f"Controller : {self.controller_name}")
            print(f"Saved to   : {self.run_directory}")
            print("====================================\n")

            raise NotImplementedError
        # print('record')
        
        current_position = np.array([pwcs.pose.pose.position.x, pwcs.pose.pose.position.y, pwcs.pose.pose.position.z])
        self.current_position = current_position - self.initial_position_offset  
        base_velocity = np.hstack((self.base_linear_velocities[:2], self.base_angular_velocities[2]))

        self.logged_positions.append(self.current_position)
        self.logged_velocities.append(base_velocity)
        
        # Get current robomaster orientation in vicon system
        self.current_orientation = np.array([pwcs.pose.pose.orientation.x, pwcs.pose.pose.orientation.y, pwcs.pose.pose.orientation.z, pwcs.pose.pose.orientation.w])
        if self.steps == 0:
            ref_positions = self.compute_reference_trajectory(self.current_position, 
                                            quat_to_yaw(self.current_orientation), 
                                            self.traj_commands,
                                            (1/self.publishing_rate))
            
            np.savetxt(
                os.path.join(
                    self.run_directory,
                    "reference_positions.csv"
                ),
                ref_positions,
                delimiter=","
            )


        # Get velocities of robomaster in vicon system
        self.prev_base_linear_velocities = self.base_linear_velocities
        self.prev_base_angular_velocities = self.base_angular_velocities
        self.base_linear_velocities = np.array([odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z])
        # print(f'Twist Vel: {self.base_linear_velocities}')
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

        if self.follow_trajectory:
            commands = self.traj_commands[self.steps]
        elif not self.record_normal and not self.playback_cmd:
            commands = np.hstack((self.linear_velocities.flatten()[0], self.linear_velocities[1], self.angular_velocity))
        else:
            commands = self.commands
        
        # print(f'Commands: {commands}')

        if self.record_bag and self.steps <= self.recording_threshold:
            twist = Twist()
            twist.linear.x = commands[0]
            twist.linear.y = commands[1]
            twist.angular.z = commands[2]
            # print(commands)
            self.writer.write(f'/{self.robomaster}/cmd_vel_bag', serialize_message(twist), self.get_clock().now().nanoseconds)

        if self.use_actions:
            if self.is_offset:
                offset = self.get_actions(commands)[[3, 0, 2, 1]] *  (self.max_velocity * self.action_scale)
                # print(f'Offset: {offset}')
                self.wheel_velocities = get_wheel_velocities(np.expand_dims(commands, axis=-1), 
                                    self.wheel_velocities_matrix).clip(-self.max_velocity, self.max_velocity).flatten() + offset
                # self.wheel_velocities = np.ones_like(self.wheel_velocities) *10
                # self.wheel_velocities[3] = 20.
            else:
                self.wheel_velocities = self.get_actions(commands)[[3, 0, 2, 1]]
        else:
            self.wheel_velocities = get_wheel_velocities(np.expand_dims(commands, axis=-1), self.wheel_velocities_matrix).flatten()
        # print(self.wheel_velocities)
        self.clipped_velocities = self.wheel_velocities.clip(-self.max_velocity, self.max_velocity)
        
        if self.record_bag and self.steps <= self.recording_threshold:
            self.writer.write(f'/{self.robomaster}/joint_states_bag', serialize_message(joint), self.get_clock().now().nanoseconds)
            self.writer.write(f'/{self.robomaster}/vicon_pose_bag', serialize_message(pwcs), self.get_clock().now().nanoseconds)
            self.writer.write(f'/{self.robomaster}/vicon_odom_bag', serialize_message(odom), self.get_clock().now().nanoseconds)
        
        self.publish_velocities()

    def get_actions(self, commands):
        base_velocity = np.hstack((self.base_linear_velocities[:2], self.base_angular_velocities[2]))
        base_linear_acc = self.base_linear_velocities - self.prev_base_linear_velocities
        base_angular_acc = self.base_angular_velocities - self.prev_base_angular_velocities
        base_acceleration = np.hstack((base_linear_acc[:2], base_angular_acc[2]))
        tracking_error = commands - base_velocity
        # Build observation        
        current_obs = np.hstack([base_velocity, base_acceleration, self.linear_acceleration, commands, tracking_error, self.actions.squeeze()])
        # print(current_obs)

        if self.frame_stack != 1:
            self.observation_stack.add_obs_history(torch.tensor(current_obs, device=self.device, dtype=torch.float))
            self.obs = self.observation_stack.return_obs(flatten=True)
        else:
            self.obs = current_obs
        
        self.obs = torch.tensor(self.obs, device=self.device, dtype=torch.float).unsqueeze(0)

        obs = self.obs

        input_dict = {
        'is_train' : False,
        'prev_actions' : None,
        'obs' : obs,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
        self.actions = res_dict['mus'].clamp(-1, 1).cpu().detach().numpy()
        scaled_actions = self.actions if self.is_offset else self.actions * self.max_velocity
        return scaled_actions.squeeze()


    def publish_velocities(self):
        msg = JointState()
        msg.header = self.joint_states.header

        # command_order = [0, 1, 3, 2]

        # msg.name = [
        #     self.joint_states.name[i]
        #     for i in command_order
        # ]

        # msg.position = [
        #     self.joint_states.position[i]
        #     for i in command_order
        # ]
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

    def create_run_directory(self):
        """
        Creates:

        records/
            trajectory_name/
                controller_name/
                    run_001/
                    run_002/
                    ...
        """

        base_directory = os.path.join(
            os.getcwd(),
            "rm_publisher",
            "records",
            self.trajectory_name,
            self.controller_name
        )

        os.makedirs(base_directory, exist_ok=True)

        # Find next available run number
        run_number = 1

        while True:
            run_directory = os.path.join(
                base_directory,
                f"run_{run_number:03d}"
            )

            if not os.path.exists(run_directory):
                break

            run_number += 1

        os.makedirs(run_directory)

        return run_directory

    # def on_press(self, key): 

    #     if key == keyboard.Key.esc:
    #         return False  # stop listener
    #     try:
    #         k = key.char  # single-char keys
    #     except:
    #         k = key.name  # other keys
    #     if k == 'w':
    #         self.linear_velocities[0] = min(self.lin_vel_x_scale, self.linear_velocities[0] + 0.1)
    #     if k == 's':
    #         self.linear_velocities[0] = max(-self.lin_vel_x_scale, self.linear_velocities[0] - 0.1)
    #     if k == 'a':
    #         self.linear_velocities[1] = min(self.lin_vel_y_scale, self.linear_velocities[1] + 0.1)
    #     if k == 'd':
    #         self.linear_velocities[1] = max(-self.lin_vel_y_scale, self.linear_velocities[1] - 0.1)
    #     if k == 'q':
    #         self.heading_direction = ((self.heading_direction + np.pi / 8) % (np.sign(self.heading_direction + np.pi / 8 + 1e-8) * 2 * np.pi))
    #     if k == 'e':
    #         self.heading_direction = ((self.heading_direction - np.pi / 8) % (np.sign(self.heading_direction - np.pi / 8 - 1e-8) * 2 * np.pi))
  
    # def on_release(self, key):
    #     try:
    #         if key.char == 'w':
    #             self.linear_velocities[0] = 0
    #         if key.char == 's':
    #             self.linear_velocities[0] = 0
    #         if key.char == 'a':
    #             self.linear_velocities[1] = 0
    #         if key.char == 'd':
    #             self.linear_velocities[1] = 0
    #     except AttributeError:
    #         pass
def stamp_to_sec(msg):
        return (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

def quat_to_yaw(q):
    x, y, z, w = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

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
    target_vels_publisher = Trajectory(args=rm_args)

    # Start the ROS2 spinning loop in a separate thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(target_vels_publisher,))
    ros_thread.start()

    # Start the keyboard listener in its own thread
    # listener = keyboard.Listener(on_press=target_vels_publisher.on_press, on_release=target_vels_publisher.on_release)
    # listener.start()

    try:
        # Keep the main thread alive while ROS2 and listener threads are running
        ros_thread.join()
    finally:
        # Ensure the listener is stopped before shutdown
        # listener.stop()
        # listener.join()  # Wait for listener thread to finish

        # Clean up ROS2 resources
        target_vels_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
  args = get_args()
  args = parse_args(args)
  main(rm_args=args)