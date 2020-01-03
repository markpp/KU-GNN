import sys, math, pygame
import pandas as pd
from pyntcloud import PyntCloud
import argparse
import os
import vg
import numpy as np
from tools import load_point_normal, load_pose, vec2azel, generate_axis_lines, generate_point, generate_normal


def rotateX(point, cosa, sina):
    """ Rotates the point around the X axis. """
    y = point[1] * cosa - point[2] * sina
    z = point[1] * sina + point[2] * cosa
    return np.array([point[0], y, z])

def rotateY(point, cosa, sina):
    """ Rotates the point around the Y axis. """
    z = point[2] * cosa - point[0] * sina
    x = point[2] * sina + point[0] * cosa
    return np.array([x, point[1], z])

def rotateZ(point, cosa, sina):
    """ Rotates the point around the Z axis. """
    x = point[0] * cosa - point[1] * sina
    y = point[0] * sina + point[1] * cosa
    return np.array([x, y, point[2]])

def project(point, win_width, win_height, fov, viewer_distance):
    """ Transforms this 3D point to 2D using a perspective projection. """
    factor = fov / (viewer_distance + point[2])
    x = point[0] * factor + win_width / 2
    y = -point[1] * factor + win_height / 2
    return int(x), int(y)

def show_points(points, point_colors, pose = None, pose_colors = None, p0 = [0,0,0], nx = None, ny = None, nz = None, filename = "none", show=True):

    if p0 is not None and nx is not None:
        p_norm, c_norm = generate_normal(p0, nx)
        p_point, c_point = generate_point(p0)
        points = np.concatenate(([points, p_norm, p_point]), axis=0)
        point_colors = np.concatenate(([point_colors, c_norm, c_point]), axis=0)

    viewer = PointViewer()
    angleX, angleY, angleZ = 0, 0, 0
    
    rot_points = points
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        viewer.clock.tick(50)
        viewer.screen.fill((255, 255, 255))

        d2r = math.pi / 180
        radx, rady, radz = angleX * d2r,  angleY * d2r,  angleZ * d2r
        cosa = [math.cos(radx), math.cos(rady), math.cos(radz)]
        sina = [math.sin(radx), math.sin(rady), math.sin(radz)]
        for rot_p, color_p in zip(rot_points,point_colors):
            # Rotate the point around X axis, then around Y axis, and finally around Z axis.
            if angleX != 0:
                rot_p = rotateX(rot_p,cosa[0],sina[0])
            if angleY != 0:
                rot_p = rotateY(rot_p,cosa[1],sina[1])
            if angleZ != 0:
                rot_p = rotateZ(rot_p,cosa[2],sina[2])

            # Transform the point from 3D to 2D
            color_p = tuple(int(i*255) for i in color_p)
            x, y = project(rot_p, viewer.screen.get_width(), viewer.screen.get_height(), fov = 512, viewer_distance = 0.3)
            viewer.screen.fill(color_p,(x,y,4,4))
            #pygame.draw.circle(viewer.screen, color_p, (x,y), 5, 1)

        angleX += 1
        angleY += 1
        #angleZ += 1

        pygame.display.flip()
        if not show:
            pygame.image.save(viewer.screen,"figs/pyg/{}.png".format(filename))
            break

class PointViewer:
    def __init__(self, win_width = 1280, win_height = 960):
        pygame.init()

        self.screen = pygame.display.set_mode((win_width, win_height))
        pygame.display.set_caption("Pygame based Point Cloud visualization and rotation")

        self.clock = pygame.time.Clock()


if __name__ == '__main__':
    """
    Main function for executing the .py script.
    Command:
        -p path/<filename>.npy
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--points", type=str,
                    help="point cloud")
    ap.add_argument("-s", "--show", type=bool,
                    default=True, help="show else plot")
    args = vars(ap.parse_args())

    cloud = PyntCloud.from_file(args["points"])

    df = cloud.points.dropna()
    p_cloud = df.values[:,:3]

    n_points, n_dim = df.shape
    if n_dim == 6:
        c_cloud = df.values[:,3:6] # use color from point cloud if any
    else:
        c_cloud = np.full((len(p_cloud),3),[1.0, 0.7, 0.75]) # fixed color

    # check if json file exists and load pose
    pose_path = args["points"][:-4]+".json"
    use_pose = os.path.exists(pose_path)
    if use_pose:
        p0, nx = load_point_normal(pose_path) # get ear vector from point + normal
        #p0, nx = load_pose(pose_path) # get ear vector from ros pose
        nz = np.cross(nx,[0,1,0])
        ny = np.cross(nz,nx)

    if use_pose:
        show_points(p_cloud,c_cloud,p0=p0,nx=nx,ny=ny,nz=nz,show=args["show"])
    else:
        show_points(p_cloud,c_cloud,show=args["show"])
