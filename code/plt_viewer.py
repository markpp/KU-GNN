import numpy as np
import matplotlib.pylab  as plt
from mpl_toolkits.mplot3d import Axes3D

def show_points(points, point_colors, p0 = [0,0,0], nx = None, ny = None, nz = None, filename = "none"):
    # Create the figure
    fig = plt.figure(figsize=(6,6))

    # Add an axes
    ax = fig.add_subplot(111,projection='3d')

    # and plot the point
    ax.scatter(points[:,0] , points[:,1] , points[:,2],  color=point_colors, alpha=0.5)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_axis_off()

    lim_val = 0.10
    ax.set_xlim3d(-lim_val, lim_val)
    ax.set_ylim3d(-lim_val, lim_val)
    ax.set_zlim3d(-lim_val, lim_val)

    ax.quiver(*p0, *nx, length=0.15, normalize=False)

    plt.show()
