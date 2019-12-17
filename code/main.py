import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pygsp import graphs, filters, plotting, utils
from pygsp.graphs import Graph
import networkx as nx
import dgl
import random


def sample_n_points(points, n_points=1024):
    candidate_ids = [i for i in range(points.shape[0])]
    sel = []
    for _ in range(n_points):
        # select idx for closest point and add to id_selections
        idx = random.randint(0,len(candidate_ids)-1)
        sel.append(candidate_ids[idx])
        # remove that idx from point_idx_options
        del candidate_ids[idx]
    return points[sel]

def plot_points_with_arrow(x, y):
  p_cloud = x
  c_cloud = np.full((len(p_cloud),3),[1, 0.7, 0.75]) # fixed color

  p0, nx = y[:3], y[3:]
  nz = np.cross(nx,[0,1,0])
  ny = np.cross(nz,nx)

  from plt_viewer import show_points
  show_points(p_cloud,c_cloud,p0=p0,nx=nx,ny=ny,nz=nz)

def plot_graph(G):
  from pygsp import plotting
  #plotting.plot(G,show_edges=True,vertex_size=30)
  plotting.plot(G,show_edges=True,vertex_size=30,backend='pyqtgraph')

  plt.axis('off')
  plt.show()

if __name__ == '__main__':
  X = np.load("../data/val_X.npy",allow_pickle=True)
  print("{} point clouds, cloud 0 has shape {}".format(X.shape,X[0].shape))
  Y = np.load("../data/val_Y.npy",allow_pickle=True)
  print("{} point+normals, p0=Y[0,:3], pn=Y[0,3:]".format(Y.shape))

  for idx in range(2,4):
    x = sample_n_points(X[idx],n_points=256*4)
    #plot_points_with_arrow(x, Y[idx])

    pygsp_G = graphs.NNGraph(x,use_flann=True, center=True, k=8)
    plot_graph(pygsp_G)

    '''
    #g_dgl = dgl.DGLGraph(G)
    nx_G = Graph.to_networkx(pygsp_G)

    fig, ax = plt.subplots()
    #nx.draw(g_dgl.to_networkx(), ax=ax)
    nx.draw_networkx(nx_G)
    #ax.set_title('Class: {:d}'.format(0))
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.show()
    '''
