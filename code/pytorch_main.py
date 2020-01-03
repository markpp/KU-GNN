import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from optimizer import Lookahead, RAdam

from models import NoSplit, HalfSplit, CompleteSplit
from dgl.data.utils import download, get_download_dir

from functools import partial
import tqdm
import urllib
import os
import argparse
import numpy as np
import json

import plot

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='')
parser.add_argument('--num-epochs', type=int, default=101)
parser.add_argument('--num-workers', type=int, default=6)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size
local_path = args.dataset_path or os.path.join(os.getcwd(), "experiments")

def train(model, opt, dev, crit, X, Y):
    model.train()
    total_loss = 0
    num_batches = 0
    alpha = 0.95

    permutation = np.random.permutation(X.shape[0])
    for i in range(0,X.shape[0], batch_size):
        indices = permutation[i:i+batch_size]
        data, label = X[indices].to(dev), Y[indices].to(dev)

        p, _, n, _ = model(data)
        p_loss = crit(p, label[:,:3])
        n_loss = crit(n, label[:,3:])
        #loss = p_loss + n_loss
        loss = alpha*p_loss + (1-alpha)*n_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        num_batches += 1
        epoch_loss = total_loss / num_batches

    return epoch_loss

def evaluate(model, dev, crit, X, Y):
    model.eval()
    p_total_error = 0
    n_total_error = 0
    num_batches = 0

    permutation = np.random.permutation(X.shape[0])
    for i in range(0,X.shape[0], batch_size):
        indices = permutation[i:i+batch_size]
        data, label = X[indices].to(dev), Y[indices].to(dev)

        p, _, n, _ = model(data)
        p_err = crit(p, label[:,:3])
        n_err = crit(n, label[:,3:])

        num_batches += 1

        p_total_error += p_err
        p_epoch_error = p_total_error / num_batches

        n_total_error += n_err
        n_epoch_error = n_total_error / num_batches

    return p_epoch_error, n_epoch_error


def train_loop(conf, dir, dev, train_X, train_Y, val_X, val_Y):
    print("Training {}, with architecture {}".format(conf['net'],conf['arch']))

    if conf['data'] == "xyz":
        dims = 3
    else:
        dims = 6

    if conf['arch'] == 'CompleteSplit':
        model = CompleteSplit(conf['net'], input_dims=dims)
    elif conf['arch'] == 'HalfSplit':
        model = HalfSplit(conf['net'], input_dims=dims)
    else:
        model = NoSplit(conf['net'], input_dims=dims)

    model = model.to(dev)

    opt = Lookahead(base_optimizer=RAdam(model.parameters(), lr=conf['lr']),k=5,alpha=0.5)
    #opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.num_epochs, eta_min=0.0001)

    plot_file = open("{}/err.txt".format(dir),'w')
    plot_file.write("p_train_err:n_train_err:p_val_err:n_val_err\n")

    best_val_err = 999.9

    train_crit = nn.MSELoss()
    eval_crit = nn.L1Loss()
    for epoch in range(args.num_epochs):
        train_loss = train(model, opt, dev, train_crit, train_X, train_Y)
        print('Epoch #{}, training loss {:.5f}'.format(epoch,train_loss))
        #del train_loss
        if epoch % 5 == 0:
            with torch.no_grad():
                p_train_err, n_train_err = evaluate(model, dev, eval_crit, train_X, train_Y)
                p_val_err, n_val_err = evaluate(model, dev, eval_crit, val_X, val_Y)
                val_err = p_val_err + n_val_err
                train_err = p_train_err + n_train_err
                plot_file.write("{:.5f}:{:.5f}:{:.5f}:{:.5f}\n".format(p_train_err, n_train_err, p_val_err, n_val_err))

                if val_err < best_val_err:
                    best_val_err = val_err
                    torch.save(model.state_dict(),'{}/model.pth'.format(dir))
                    torch.save(model,'{}/model.pkl'.format(dir))
                    print('Epoch #{}, validation err: {:.5f} (best: {:.5f}), train err: {:.5f}'.format(epoch, val_err, best_val_err, train_err))
                #del train_err
                #del val_err
    plot_file.close()


if __name__ == '__main__':
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data #
    data_source = "ct"
    val_X = np.load('{}/{}/{}_x_1024.npy'.format("input",data_source,"val"),allow_pickle=True)
    val_Y = np.load('{}/{}/{}_y.npy'.format("input",data_source,"val"),allow_pickle=True)[:,:].astype('float32')


    # load parameters
    experiment_name = 'ct_sampling'
    iterations = list(range(8))
    with open('experiments/{}.json'.format(experiment_name)) as f:
      confs = json.load(f)
      for iteration in iterations[:]:
          print("iteration {}".format(iteration))
          for conf in confs[:]:
            print(conf)

            experiment_dir = os.path.join(local_path,"{}/id-{}_it-{}".format(experiment_name,conf['id'],iteration))
            if not os.path.exists(experiment_dir):
              os.makedirs(experiment_dir)

            train_X = np.load('{}/{}/{}_x_1024.npy'.format("input",data_source,"train"),allow_pickle=True)
            train_Y = np.load('{}/{}/{}_y.npy'.format("input",data_source,"train"),allow_pickle=True)[:,:].astype('float32')

            if conf['samp'] == "furthest":
                print("using furthest sampling")
                perm = np.load('{}/{}/furthest_point.npy'.format("input",data_source))
                train_X, train_Y = train_X[perm], train_Y[perm]

            #print("{}, {}, {}, {}".format(val_X.shape, val_Y.shape, train_X.shape, train_Y.shape))

            # train #
            if conf['data'] == "xyz":
                train_x, val_x = torch.from_numpy(train_X[:,:,:3]), torch.from_numpy(val_X[:,:,:3])
            elif conf['data'] == "rgb":
                train_x, val_x = torch.from_numpy(train_X[:,:,:]), torch.from_numpy(val_X[:,:,:])
                train_x[:,:,3:], val_x[:,:,3:] = train_x[:,:,3:]/255.0, val_x[:,:,3:]/255.0

            print("# training samples {}".format(int(len(train_x)*conf['share'])))
            train_x, train_y = train_x[:int(len(train_x)*conf['share'])], torch.from_numpy(train_Y[:int(len(train_Y)*conf['share'])]),
            val_x, val_y = val_x[:int(len(val_x))], torch.from_numpy(val_Y[:int(len(val_Y))])

            train_loop(conf, experiment_dir, dev, train_x, train_y, val_x, val_y)

            # plot #
            train_errs_p = []
            train_errs_n = []
            val_errs_p = []
            val_errs_n = []
            f = open("{}/err.txt".format(experiment_dir), "r")
            for line in f.readlines()[1:]:
                train_err_p, train_err_n, val_err_p, val_err_n = line.rstrip().split(':')
                train_errs_p.append(float(train_err_p))
                train_errs_n.append(float(train_err_n))
                val_errs_p.append(float(val_err_p))
                val_errs_n.append(float(val_err_n))

            plot.train_val_err(train_errs_p, val_errs_p, backend='sns', path=os.path.join(experiment_dir,"err_p.png"))
            plot.train_val_err(train_errs_n, val_errs_n, backend='sns', path=os.path.join(experiment_dir,"err_n.png"))
