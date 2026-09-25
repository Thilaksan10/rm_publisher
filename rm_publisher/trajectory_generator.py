import numpy as np
import matplotlib.pyplot as plt
import csv

def generate_forward_then_right_curve_path(
    forward_duration=200,
    curve_duration=162,
    vx_forward=2.0,
    omega_right=-1.0,
    sim_dt=1/50,
    decimation=1,
    save_path='trajectories/forward_right_trajectory.csv'
):
    """
    Generate a trajectory that first drives forward, then curves to the right.
    Velocities are clipped to [-1, 1].

    - forward_duration: number of timesteps driving straight
    - curve_duration: number of timesteps driving a right curve
    - vx_forward: constant forward velocity (max 1.0)
    - omega_right: angular velocity during the right curve (min -1.0)
    """

    # Step 1: Create velocity commands
    forward_traj = np.tile([vx_forward, 0.0, 0.0], (forward_duration, 1))
    curve_traj = np.tile([vx_forward, 0.0, omega_right], (curve_duration, 1))

    trajectory = np.concatenate([forward_traj, curve_traj], axis=0)

    # Step 2: Save to CSV
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['vx', 'vy', 'omega'])
        writer.writerows(trajectory)

    print(f"Saved trajectory to {save_path}")

    # Step 3: Optional preview: simulate path
    pose = np.array([0.0, 0.0, 0.0])  # x, y, heading
    positions = [pose[:2].copy()]
    for vx, vy, omega in trajectory:
        for _ in range(decimation):
            theta = pose[2]
            dx = (np.cos(theta) * vx - np.sin(theta) * vy) * sim_dt
            dy = (np.sin(theta) * vx + np.cos(theta) * vy) * sim_dt
            dtheta = omega * sim_dt

            pose[0] += dx
            pose[1] += dy
            pose[2] += dtheta
            positions.append(pose[:2].copy())

    positions = np.array(positions)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(positions[:, 0], positions[:, 1], label='Simulated Path')
    plt.quiver(positions[::20, 0], positions[::20, 1],
               np.cos(pose[2]), np.sin(pose[2]), scale=20, color='r')
    plt.axis('equal')
    plt.grid()
    plt.title("Forward → Right Curve Trajectory")
    plt.xlabel("x (forward)")
    plt.ylabel("y (sideways)")
    plt.legend()
    plt.show()

def generate_forward_then_right_curve_path_omni(
    forward_duration=0,
    curve_duration=162,
    vx_forward=2.5,
    vy_right=-2.5,
    sim_dt=1/50,
    decimation=1,
    save_path='trajectories/forward_right_trajectory_omni.csv'
):
    """
    Generate an omni-directional trajectory:
    - First drives forward,
    - Then transitions smoothly into a rightward motion.
    Angular velocity is always 0.

    - forward_duration: timesteps driving straight forward
    - curve_duration: timesteps transitioning from forward to right
    - vx_forward: forward velocity at the start of the curve
    - vy_right: rightward velocity at the end of the curve
    """

    # Step 1: Forward phase
    forward_traj = np.tile([vx_forward, 0.0, 0.0], (forward_duration, 1))

    # Step 2: Smooth transition phase (from forward -> right)
    vx_curve = np.linspace(vx_forward, 0.0, curve_duration)
    vy_curve = np.linspace(0.0, vy_right, curve_duration)
    omega_curve = np.zeros(curve_duration)
    curve_traj = np.stack([vx_curve, vy_curve, omega_curve], axis=1)

    # Combine trajectory
    trajectory = np.concatenate([forward_traj, curve_traj], axis=0)

    # Step 3: Save to CSV
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['vx', 'vy', 'omega'])
        writer.writerows(trajectory)

    print(f"Saved trajectory to {save_path}")

    # Step 4: Simulate path (no rotation)
    pose = np.array([0.0, 0.0, 0.0])  # x, y, heading
    positions = [pose[:2].copy()]
    for vx, vy, omega in trajectory:
        for _ in range(decimation):
            theta = pose[2]  # heading (constant since omega=0)
            dx = (np.cos(theta) * vx - np.sin(theta) * vy) * sim_dt
            dy = (np.sin(theta) * vx + np.cos(theta) * vy) * sim_dt

            pose[0] += dx
            pose[1] += dy
            positions.append(pose[:2].copy())

    positions = np.array(positions)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(positions[:, 0], positions[:, 1], label='Simulated Path')
    plt.quiver(positions[::20, 0], positions[::20, 1],
               np.ones_like(positions[::20, 0]), np.zeros_like(positions[::20, 0]),
               scale=20, color='r')
    plt.axis('equal')
    plt.grid()
    plt.title("Forward → Smooth Right Curve (Omni, No Rotation)")
    plt.xlabel("x (forward)")
    plt.ylabel("y (sideways)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    generate_forward_then_right_curve_path_omni(sim_dt=1/50)