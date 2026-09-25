import numpy as np


        

if __name__ == '__main__':
    data_base_path = "./data-9/"
    start = 0
    sim_steps = 5000
    history_vel = np.zeros((start+sim_steps, 3))
    history_orientation = np.zeros((start+sim_steps, 4))
    history_pos = np.zeros((start+sim_steps, 3))
    base_linear_velocities = np.zeros((start+sim_steps, 3))
    base_angular_velocities = np.zeros((start+sim_steps, 3))
    data_path = f'{data_base_path}data-13.npz' 
    history_vel = np.load(data_path)['cmd_vel'][start:start+sim_steps]
    history_orientation = np.load(data_path)['orientations'][start:start+sim_steps]
    history_pos = np.load(data_path)['positions'][start:start+sim_steps]
    base_linear_velocities = np.load(data_path)['base_linear_velocities'][start:start+sim_steps]
    base_angular_velocities = np.load(data_path)['base_angular_velocities'][start:start+sim_steps]

    pos_mean = np.std(history_pos, axis=0)
    ori_mean = np.std(history_orientation, axis=0)
    bl_mean = np.std(base_linear_velocities, axis=0)
    ba_mean = np.std(base_angular_velocities, axis=0)

    print(pos_mean)
    print(ori_mean)
    print(bl_mean)
    print(ba_mean)