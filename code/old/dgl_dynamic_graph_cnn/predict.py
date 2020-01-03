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
from in_out import load_x_y, sample_xs

def predict(data, label, model, dev, rep_selection):
    if rep_selection == 'p' or rep_selection == 'n':
        pred = model(data)
        #err = nn.L1Loss()(pred, label[:,:3]) p
        #err = nn.L1Loss()(pred, label[:,3:]) n
        if dev.type == 'cuda':
            pred = pred.cpu()
        pred = pred.data.numpy()
    else:
        pred = model(data)
        #p_err = nn.L1Loss()(p, label[:,:3])
        #n_err = nn.L1Loss()(n, label[:,3:])
        #err = (p_err + n_err) / 2
        #pred = np.concatenate((p.data.numpy(), n.data.numpy()), axis=1)
    if dev.type == 'cuda':
        pred = pred.cpu()
    pred = pred.data.numpy()
    #print(err)
    #print(pred.shape)
    return pred

from itertools import islice

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def print_model_info(model):
    print("Model's parameters:")
    print("total parms: {}".format(sum(p.numel() for p in model.parameters())))
    print("trainable parms: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

if __name__ == '__main__':

    model_p = torch.load("models/model_p.pkl")


    w = list(model_p.parameters())
    print(w)

    '''
    model_n = torch.load("models/model_n.pkl")

    rep_selection = ['p_n','pn'][1]

    model_dir = 'output'
    data_dir = '/home/datasets'

    #dev = torch.device("cpu")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")##


    folder_list = ['255','510','1020','2040','4080','8160','16320']


    for dataset in ['test','val','train'][:]:
        print(dataset)
        with open('{}/{}_docker.txt'.format(data_dir,dataset)) as f:
            if dataset == 'train':
                pc_files = f.read().splitlines()[::5]
            else:
                pc_files = f.read().splitlines()

        print("predicting {} files".format(len(pc_files)))
        data, label = load_x_y(pc_files)
        #np.save("{}/gt_{}.npy".format(model_dir,dataset),data)
        np.save("{}/gt_{}.npy".format(model_dir,dataset),label)

        data = torch.from_numpy(sample_xs(data))
        label = torch.from_numpy(label)

        for folder in folder_list:

            if rep_selection == 'p_n':
                model_p = torch.load("{}/{}/model_p.pkl".format(model_dir,folder))
                #print_model_info(model_p)
                model_n = torch.load("{}/{}/model_n.pkl".format(model_dir,folder))
                #print_model_info(model_n)

                # disable DO
                model_p.eval()
                model_n.eval()
                model_p = model_p.to(dev)
                model_n = model_n.to(dev)
            else:
                model_pn = torch.load("{}/{}/model_pn.pkl".format(model_dir,folder))
                model_pn.eval()
                model_pn = model_pn.to(dev)


            pred_gnn = np.empty((0,6), float)
            for batch_idx in list(chunk(range(len(data)), 64)):
                batch_idx = list(batch_idx)
                #data = torch.from_numpy(np.load('data/testx.npy',allow_pickle=True)[:])
                x = data[batch_idx].to(dev)
                #label = torch.from_numpy(np.load('data/testy.npy',allow_pickle=True)[:].astype('float32'))
                y = label[batch_idx].to(dev)

                if rep_selection == 'p_n':
                    pred_p = predict(x, y, model_p, dev, rep_selection="p")
                    pred_n = predict(x, y, model_n, dev, rep_selection="n")

                    tmp_pred = np.concatenate((pred_p, pred_n), axis=1)
                else:
                    pred_pn = predict(x, y, model_pn, dev, rep_selection="pn")
                    tmp_pred = pred_pn
                pred_gnn = np.append(pred_gnn, tmp_pred, axis=0)
                print(pred_gnn.shape)

    pred_gnn = np.concatenate((pred_p, pred_n), axis=1)
    print(pred_gnn.shape)
    np.save("pred_gnn.npy",pred_gnn)
    '''
