import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
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


def predict(data, label, model, dev, rep_selection):
    if rep_selection == 'p':
        p = model(data)
        pred = p.data.numpy()
        err = nn.L1Loss()(p, label[:,:3])
    elif rep_selection == 'n':
        n = model(data)
        pred = n.data.numpy()
        err = nn.L1Loss()(n, label[:,3:])
    else:
        p, n = model(data)
        p_err = nn.L1Loss()(p, label[:,:3])
        n_err = nn.L1Loss()(n, label[:,3:])
        err = (p_err + n_err) / 2
        pred = np.concatenate((p.data.numpy(), n.data.numpy()), axis=1)
    print(err)
    print(pred.shape)
    return pred

if __name__ == '__main__':
    dev = torch.device("cpu")

    model_p = torch.load("models/model_p.pkl")


    w = list(model_p.parameters())
    print(w)

    '''
    model_n = torch.load("models/model_n.pkl")

    model_p.eval()
    model_n.eval()

    data = torch.from_numpy(np.load('data/testx.npy',allow_pickle=True)[:])
    data = data.to(dev)
    label = torch.from_numpy(np.load('data/testy.npy',allow_pickle=True)[:].astype('float32'))
    label = label.to(dev)

    model_p = model_p.to(dev)
    pred_p = predict(data, label, model_p, dev, rep_selection="p")
    print(pred_p.shape)

    model_n = model_n.to(dev)
    pred_n = predict(data, label, model_n, dev, rep_selection="n")
    print(pred_n.shape)

    pred_gnn = np.concatenate((pred_p, pred_n), axis=1)
    print(pred_gnn.shape)
    np.save("pred_gnn.npy",pred_gnn)
    '''