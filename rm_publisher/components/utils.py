import numpy as np
import torch

def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    a = np.array(np.pi / 2.0).repeat(sinp.shape[0])
    pitch = np.where(np.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), np.arcsin(sinp))
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.sign(roll) * (np.abs(roll) % (2*np.pi)), np.sign(pitch) * (np.abs(pitch) % (2*np.pi)), np.sign(yaw) * (np.abs(yaw) % (2*np.pi))

def copysign(a, b):   
  a = np.array(a, dtype=float).repeat(b.shape[0])
  return np.abs(a) * np.sign(b)

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([x, y, z, w], axis=-1).flatten()

    return quat

def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = np.cross(xyz, b) * 2
    return (b + a[:, 3:] * t + np.cross(xyz, t)).reshape(shape)

def quat_rotate(q, v):
    shape = q.shape
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)[..., np.newaxis]
    b = np.cross(q_vec, v, axis=-1) * q_w[..., np.newaxis] * 2.0
    c = q_vec * \
        np.matmul(q_vec.reshape(1, 3), v.reshape(3, 1)).squeeze(-1) * 2.0
    return a + b + c


def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)[..., np.newaxis]
    b = np.cross(q_vec, v, axis=-1) * q_w[..., np.newaxis] * 2.0
    c = q_vec * \
        np.matmul(q_vec.reshape(1, 3), v.reshape(3, 1)).squeeze(-1) * 2.0
    return a - b + c

def get_axis_params(value, axis_idx, x_value=0., dtype=np.float64, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return params.astype(dtype)


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def compute_robomaster_reward(target_vel, root_lin_vels, current_cells, imu,  gravity_vec, forward_vec, reset_buf, progress_buf, max_episode_length,
                              max_cells, min_cells, boundary_reached):

    # To world coordinate system
    root_lin_vels = np.stack([root_lin_vels[2], root_lin_vels[0], root_lin_vels[1]], axis=0)

    err = np.sum(np.square(target_vel[:2] - root_lin_vels[:2]), axis=0)
    reward = np.exp(-err/0.25)

    # Penalty for not driving forward
    normalized_root_vels = root_lin_vels / (np.sqrt((np.power(root_lin_vels, 2)).sum()).reshape(-1, 1) + 1e-5)
    # normalized_root_vels = torch.nn.functional.normalize(root_lin_vels)

    forward_err = -1 + (normalized_root_vels * forward_vec).sum()

    err = (imu * gravity_vec).sum(axis=0)
    death_mask = np.abs(err + 1.0) > 0.2

    reward += 0.25 * forward_err
    # reward[death_mask] -= 50.0
    reward -= 50.0 * death_mask

    # Resets
    
    m1 = ((current_cells <= min_cells)).any()
    m2 = ((current_cells >= max_cells)).any()
    m = torch.stack((m1, m2), dim=0).any()

    # Reset if out if bounds
    reset_buf = torch.where(m, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(boundary_reached, torch.zeros_like(reset_buf), reset_buf)
    # reset_buf = torch.where(death_mask, torch.ones_like(reset_buf), reset_buf)
    # Reset if episode complete
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf).squeeze()

    return reward.flatten(), reset_buf.flatten()

def vel_to_rad_s(x_in, y_in, a_in):
    
    rm_wheel_separation_width = 0.55
    rm_wheel_spparation_length = 0.8
    rm_wheel_radius = 0.07

    wheel_front_left = (1 / rm_wheel_radius) * (
            x_in + a_in - (rm_wheel_separation_width + rm_wheel_spparation_length) * -y_in)
    wheel_front_right = (1 / rm_wheel_radius) * (
            x_in - a_in + (rm_wheel_separation_width + rm_wheel_spparation_length) * -y_in)
    wheel_rear_left = (1 / rm_wheel_radius) * (
            x_in - a_in - (rm_wheel_separation_width + rm_wheel_spparation_length) * -y_in)
    wheel_rear_right = (1 / rm_wheel_radius) * (
            x_in + a_in + (rm_wheel_separation_width + rm_wheel_spparation_length) * -y_in)

    return wheel_front_left, wheel_front_right, wheel_rear_left, wheel_rear_right
