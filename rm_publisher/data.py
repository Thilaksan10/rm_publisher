import os
import rosbag2_py
import numpy as np
from rcl_interfaces.msg import Log
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import  JointState
from nav_msgs.msg import Odometry
from components.utils import quat_rotate, get_axis_params

def get_rosbag_options(path, storage_id, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(
        uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options

if __name__ == '__main__':

    bags_directory = 'bags'
    storage_id = 'sqlite3'

    gravity_vec = get_axis_params(-1, 2)

    # Loop through all files in the bags directory
    for idx, bag_uri in enumerate(sorted(os.listdir(bags_directory))):
        bag_path = f'{bags_directory}/{bag_uri}'
        print(bag_uri)
        print(idx)
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)

        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

        storage_options, converter_options = get_rosbag_options(bag_path, storage_id)


        # Open the storage object
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()

        # Create a map for quicker lookup
        type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

        msg_counter = 0
        msg_group_start = 0
        
        cmd_vels = []
        positions = []
        orientations = []
        base_linear_velocities = []
        base_angular_velocities = []
        cmd_wheels = []
        wheel_vels = []
        observations = []
        actions = []
        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            # print(msg)
            # print(topic)
            if isinstance(msg, Twist().__class__):
                # print(msg)
                cmd_vels.append(np.array([msg.linear.x, msg.linear.y, msg.angular.z]))
            elif isinstance(msg, PoseWithCovarianceStamped().__class__):
                positions.append(np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]))
                orientations.append(np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]))
            elif isinstance(msg, Odometry().__class__):
                base_linear_velocities.append(np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]))
                base_angular_velocities.append(np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]))
            elif isinstance(msg, JointState().__class__):
                if topic == '/ep05/cmd_wheel_speed_bag':
                    cmd_wheels.append(msg.velocity)
                else:
                    wheel_vels.append(msg.velocity)
                  
            msg_counter += 1
            # print(msg_counter)

            # print(msg)
        # print(msg_counter)
        # # input()

        for i in range(len(cmd_vels)):
            base_lin_vel = base_linear_velocities[i]
            base_ang_vel = base_angular_velocities[i]
            orientation = orientations[i]
            linear_acceleration = quat_rotate(orientation, gravity_vec)
            commands = cmd_vels[i]
            action = wheel_vels[i]
            observation = np.hstack([base_lin_vel[:2], base_ang_vel[2], linear_acceleration, commands, action])
            # print(observation)
            observations.append(observation[np.newaxis, :])
            actions.append(action)

        print(f'CMD VELS: {len(cmd_vels)}')
        print(f'POSITIONS: {len(positions)}')
        print(f'ORIENTATIONS: {len(orientations)}')
        print(f'BASE LIN: {len(base_linear_velocities)}')
        print(f'BASE ANG: {len(base_angular_velocities)}')
        print(f'CMD WHEELS: {len(cmd_wheels)}')
        print(f'WHEEL VELS: {len(wheel_vels)}')#
        print(f'OBS: {len(observations)}')
        print(f'ACTIONS: {len(actions)}')

        if idx == 0:
            name = 'data-10/data-controller-recording.npz'
        else:
            name = 'data-10/data-ik-recording.npz'
        np.savez(name, 
                 cmd_vels=cmd_vels, 
                 positions=positions, 
                 orientations=orientations, 
                 base_linear_velocities=base_linear_velocities, 
                 base_angular_velocities=base_angular_velocities,
                 cmd_wheels=cmd_wheels,
                 wheel_vels=wheel_vels,
                 obs=observations,
                 actions=actions)
        print(f'Saved recordings in {name}')

    # print(len())
    # print(len(poses))
    # print(len(msgs))