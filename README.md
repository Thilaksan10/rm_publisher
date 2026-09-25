# RM Publisher

### Prerequisites:

- **Ros2 Humble:**  
Ros can be installed from [here](https://docs.ros.org/en/humble/index.html)

- **Vicon Subscriber ROS2:**  
    Visit the [Vicon Subscriber ROS2](https://gitlab.cc-asp.fraunhofer.de/iml/oe130/software/pace-lab/vicon/vicon-subscriber-ros2) repositiory for installation instructions

- **Trained model from the Robomaster_rl repository:**  
In order to recieve action offsets for the robomaster's angular and linear velocities being published, a trained model must be available. This repository comes with a pretrained model using the [Robomaster RL](https://gitlab.cc-asp.fraunhofer.de/robomaster_rl/robomaster_rl/) repository.

- **Install the following libraries:**  
    `numpy`  
    `pytorch`  
    `rl_games` (installation instructions can be found [here](https://gitlab.cc-asp.fraunhofer.de/nbach/isaacgym_rlgames))  
    `matplotlib`


### Installation: 

1. Clone the repository to the `src` directory in your ros2 Workspace:  
`git clone git@gitlab.cc-asp.fraunhofer.de:robomaster_rl/rm_publisher.git`
2. Return to the ros2 workspace root directory and run:  
`colcon build --packages-select rm_publisher`

### Usage:
This project uses the Vicon System to obtain information regarding the Robomaster's current position and orientation and requires 3 prerequisites before it can run:

1. **Active Vicon Tracker:**  
Make sure that Vicon Tracker is running on the Vicon-PC (Windows) and the Robomaster in use is enabled.

2. **Vicon Bridge:**  
Run `vicon_bridge` on the Vicon-PC (Linux) terminal. 

3. **Vicon Subscriber ROS2:**  
Make sure that the robomaster in use is added to the Vicon Subscriber config. Run the ROS2 Vicon Subscriber node using `ros2 launch vicon_subscriber mqtt.launch.py` from your ros2 workspace root directory. 

The code can be executed from inside your IDE or by running:
- `ros2 run rm_publisher publisher` from the ros2 workspace root directory
- `python3 target_vels_publisher.py` from inside the project's root directory

### Configuration:

This project contains a number of configurable options that may be tweaked for testing purposes. These may be edited in the `rm_publisher.yaml` config file or provided as arguments when running from the terminal.

| Parameter         | Argument      | Value (example)                   | Description 
|---                |---            |---                                |---
| **Robomaster**
|numRms             |               | 1                                 |Number of Robomasters in use
|name               | --robomaster  | 'ep01'                            |Name of the Robomaster(s) in use
|maxVelocity        |               | 1                                 |Maximum Velocity the robomaster is allowed to reach
|minVelocity        |               | 0.07                              |Minimum Velocity before the the Robomaster resets       
|initialPosition    | --offset      | [9,3,0]                           |An offset provided to the Robomaster's initial position
|initialRotation    |               | [0,0,-0.7071068, 0.7071068]       |An offset provided to the Robomaster's initial rotation
| **Map**
|mapSize            | --map_size    |10                                 | Size of the generated Velocity Field
|cellSize           |               |0.1                                | Size of each generated cell
|boundarySize       |               |5                                  | The map/Robomaster resets when the Robomaster reaches the boundary
|useInterpolated    | --interpolated| False                             | Whether to use interpolated velocities instead of regular velocities or not
|enableDebugVis     |               | True                              | Visualizes the map in matplotlib for debugging purposes
|episodeLength      |               | 250                               | Max episode length before which the robomaster/map resets
| **Model**
|actionObsHistory   |               | 50                                | History size
|actionSize         |               | 3                                 | Number of actions recieved from the model
|obsSize            |               | 8                                 | Number of observations sent to the model
|path               |               | 'rm_publisher/nn/Robomaster.pth'  | Path to the trained model in use