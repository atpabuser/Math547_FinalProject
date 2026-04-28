import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# define the root joint and scaling of the values
R = 1000
# define the connections between the joints (skeleton)
I = np.array(
	[1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 16, 21, 22, 23, 25, 26, 24, 28, 16, 30, 31,
	 32, 33, 34, 35, 33, 37])-1
J = np.array(
	[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32,
	 33, 34, 35, 36, 37, 38])-1


def plot_action(vec, file_name, nodes_to_highlight=[]):
	xyz = vec.reshape(38, 3, -1)
	xroot, yroot, zroot = xyz[0, 0, 0], xyz[0, 1, 0], xyz[0, 2, 0]

	for time_idx in range(1, xyz.shape[2]):
		plt.figure(figsize=(10, 10))
		ax = plt.axes(projection='3d')
		for ijind in range(0, I.shape[0]):
			xline = np.array([xyz[I[ijind], 0, time_idx], xyz[J[ijind], 0, time_idx]])
			yline = np.array([xyz[I[ijind], 1, time_idx], xyz[J[ijind], 1, time_idx]])
			zline = np.array([xyz[I[ijind], 2, time_idx], xyz[J[ijind], 2, time_idx]])
			# use plot if you'd like to plot skeleton with lines
			ax.plot(xline, yline, zline, c='k', marker='o')
			if len(nodes_to_highlight) > 0:
				ax.scatter(
					xyz[nodes_to_highlight, 0, time_idx], xyz[nodes_to_highlight, 1, time_idx], xyz[nodes_to_highlight, 2, time_idx],
					c='yellow', s=50, alpha=0.5
				)

		# use scatter if you'd like to plot all points without lines
		# ax.scatter(xyz[:,0,time_idx],xyz[:,1,time_idx],xyz[:,2,time_idx], c = 'r', s = 50)

		ax.set_xlim((-R+xroot, R+xroot))
		ax.set_ylim((-R+yroot, R+yroot))
		ax.set_zlim((-R+zroot, R+zroot))

		plt.savefig(f'./anim/{time_idx}.png')
		plt.close()

	images = [Image.open(f'./anim/{n}.png') for n in range(1, xyz.shape[2])]
	images[0].save(f'./anim/{file_name}.gif', save_all=True, append_images=images[1:], duration=30, loop=0)

	# remove plotted png images
	for n in range(1, xyz.shape[2]):
		os.remove(f'./anim/{n}.png')