import numpy as np
import matplotlib.pyplot as plt

cmd_velocities_act = np.load('data-10/data-0.npz', allow_pickle=True)['cmd_vels']
base_linear_velocities_act = np.load('data-10/data-0.npz', allow_pickle=True)['base_linear_velocities']
base_angular_velocities_act = np.load('data-10/data-0.npz', allow_pickle=True)['base_angular_velocities']
cmd_velocities_mat = np.load('data-10/data-1.npz', allow_pickle=True)['cmd_vels']
base_linear_velocities_mat = np.load('data-10/data-1.npz', allow_pickle=True)['base_linear_velocities']
base_angular_velocities_mat = np.load('data-10/data-1.npz', allow_pickle=True)['base_angular_velocities']

# Create a figure and axis
fig, ax = plt.subplots(3, 1, figsize=(15, 10))

# Plotting each component (x, y, z) for each type of velocity
labels = ['x', 'y', 'z']

# Plotting cmd_velocities_act
for i in range(2):
    ax[i].plot(cmd_velocities_act[:, i], label=f'cmd_velocities_act {labels[i]}')
    ax[i].plot(base_linear_velocities_mat[:, i], label=f'base_linear_velocities_mat {labels[i]}', linestyle='--')
    ax[i].plot(base_linear_velocities_act[:, i], label=f'base_linear_velocities_act {labels[i]}', linestyle='dotted')
    ax[i].set_title(f'{labels[i]}')
    ax[i].legend()

# Plotting base_velocities
# for i in range(2):  # x, y
#     ax[i, 1].plot(cmd_velocities_act[:, i], label=f'cmd_velocities_mat {labels[i]}')
#     ax[i, 1].plot(base_linear_velocities_act[:, i], label=f'base_linear_velocities_act {labels[i]}', linestyle='--')
#     ax[i, 1].set_title(f'policy {labels[i]}')
#     ax[i, 1].legend()

# Plotting base_angular_velocities (z component) in the third row
ax[2].plot(cmd_velocities_act[:, 2], label=f'cmd_velocities_act z')
ax[2].plot(base_angular_velocities_mat[:, 2], label='base_angular_velocities_mat z', linestyle='--')
ax[2].set_title('z')
ax[2].legend()
# ax[2, 1].plot(cmd_velocities_act[:, 2], label=f'cmd_velocities_mat {labels[i]}')
ax[2].plot(base_angular_velocities_act[:, 2], label='base_angular_velocities_act z', linestyle='dotted')
# ax[2, 1].set_title('base_angular_velocities z')
# ax[2, 1].legend()


# Add common labels
fig.text(0.5, 0.04, 'Sample Index', ha='center')
fig.text(0.04, 0.5, 'Velocity', va='center', rotation='vertical')

# Calculate differences for linear velocities (x, y) and angular velocity (z)
linear_diff_x = cmd_velocities_act[:, 0] - base_linear_velocities_act[:, 0]
linear_diff_y = cmd_velocities_act[:, 1] - base_linear_velocities_act[:, 1]
angular_diff_z = cmd_velocities_act[:, 2] - base_angular_velocities_act[:, 2]

# Calculate standard deviations
std_dev_linear_x = np.std(linear_diff_x)
std_dev_linear_y = np.std(linear_diff_y)
std_dev_angular_z = np.std(angular_diff_z)

# Print the standard deviations
print(f"Standard Deviation between cmd_velocities and base_linear_velocities (x): {std_dev_linear_x}")
print(f"Standard Deviation between cmd_velocities and base_linear_velocities (y): {std_dev_linear_y}")
print(f"Standard Deviation between cmd_velocities and base_angular_velocities (z): {std_dev_angular_z}")

# Calculate differences for linear velocities (x, y) and angular velocity (z)
linear_diff_x = cmd_velocities_act[:, 0] - base_linear_velocities_mat[:, 0]
linear_diff_y = cmd_velocities_act[:, 1] - base_linear_velocities_mat[:, 1]
angular_diff_z = cmd_velocities_act[:, 2] - base_angular_velocities_mat[:, 2]

# Calculate standard deviations
std_dev_linear_x = np.std(linear_diff_x)
std_dev_linear_y = np.std(linear_diff_y)
std_dev_angular_z = np.std(angular_diff_z)

# Print the standard deviations
print(f"Standard Deviation between cmd_velocities and base_linear_velocities (x): {std_dev_linear_x}")
print(f"Standard Deviation between cmd_velocities and base_linear_velocities (y): {std_dev_linear_y}")
print(f"Standard Deviation between cmd_velocities and base_angular_velocities (z): {std_dev_angular_z}")

plt.tight_layout()
plt.show()