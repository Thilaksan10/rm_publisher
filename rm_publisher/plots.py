import numpy as np
import matplotlib.pyplot as plt

history_2 = np.load('./rm_publisher/plots/data4.npz')

pose_2 = history_2['poses'].squeeze()
fig, ax = plt.subplots()


plt.scatter(pose_2[0, 0], pose_2[0, 1], color='r')
plt.plot(pose_2[1:, 0], pose_2[1:, 1], color='r')

ax.set_xlim([-4.5, 4.5])
ax.set_ylim([-4.5, 4.5])

plt.grid()
plt.show()
