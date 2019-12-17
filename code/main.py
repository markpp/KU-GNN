import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from optimizer import Lookahead, RAdam

#import torch.onnx
#from torch.utils.tensorboard import SummaryWriter

from earnet import EarNet
from model import Model
from dgl.data.utils import download, get_download_dir

from functools import partial
import tqdm
import urllib
import os
import argparse
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
    rep_selection = ['p','n','pn'][1]

    with open('/home/datasets/train_docker.txt') as f:
        num_train = len(f.read().splitlines())

    variables = [num_train//64, num_train//32, num_train//16, num_train//8, num_train//4, num_train//2, num_train][:2]
    print("variables: {}".format(variables))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")##

    modelnet = EarNet(path='/home/datasets')
    val_loader = TestDataLoader(modelnet.val())
    test_loader = TestDataLoader(modelnet.test())

    for var in variables:
        print(var)
        train_loader = TrainDataLoader(modelnet.train(split=var))

        model = Model(10, [64, 64, 128, 256], [512, 512, 256], output_dims=3, rep=rep_selection)
        model = model.to(dev)

        opt = Lookahead(base_optimizer=RAdam(model.parameters(), lr=0.01),k=5,alpha=0.5)
        #opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.num_epochs, eta_min=0.0001)

        best_val_err = 999.9
        best_test_err = 999.9

        experiment_dir = os.path.join(local_path,"{}".format(var))
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        plot_file = open("{}/err_{}.txt".format(experiment_dir,rep_selection),'w')
        plot_file.write("val_err:test_err\n")

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
