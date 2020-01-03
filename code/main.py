import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse
import numpy as np
import json

from src.common.plot import plot_loss
from src.common.models import NoSplit, HalfSplit, FullSplit
from src.common.optimizer import Lookahead, RAdam, Ralamb

parser = argparse.ArgumentParser()
parser.add_argument('--output-path', type=str, default='')
parser.add_argument('--num-epochs', type=int, default=201)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

local_path = args.output_path or os.path.join(os.getcwd(), "experiments/gnn")
num_epochs = args.num_epochs
batch_size = args.batch_size

def train(model, opt, dev, crit, X, Y, scheduler, alpha=0.95):
    scheduler.step()
    model.train()
    total_loss = 0
    num_batches = 0

    permutation = np.random.permutation(X.shape[0])
    for i in range(0,X.shape[0], batch_size):
        indices = permutation[i:i+batch_size]
        data, label = X[indices].to(dev), Y[indices].to(dev)

        p, _, n, _ = model(data)
        p_loss = crit(p, label[:,:3])
        n_loss = crit(n, label[:,3:])
        loss = alpha*p_loss + (1-alpha)*n_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        num_batches += 1
        epoch_loss = total_loss / num_batches

    #for param_group in opt.param_groups:
        #lr = param_group['lr']

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
    print("Training {}, with architecture {}".format(conf['ne'],conf['ar']))

    dims = 3 if conf['da'] == "xyz" else 6
    if conf['ar'] == 'FullSplit':
        model = FullSplit(conf['ne'], input_dims=dims)
    elif conf['ar'] == 'HalfSplit':
        model = HalfSplit(conf['ne'], input_dims=dims)
    else:
        model = NoSplit(conf['ne'], input_dims=dims)

    model = model.to(dev)

    if conf['op'] == 'ranger':
        opt = Lookahead(base_optimizer=RAdam(model.parameters(), lr=conf['lr']),k=5,alpha=0.5)
    elif conf['op'] == 'ralamb':
        opt = Lookahead(base_optimizer=Ralamb(model.parameters(), lr=conf['lr']),k=5,alpha=0.5)
    elif conf['op'] == 'adam':
        opt = optim.Adam(model.parameters(), lr=conf['lr'], weight_decay=1e-4)
    else:
        opt = optim.SGD(model.parameters(), lr=conf['lr'], momentum=0.9, weight_decay=1e-4)

    #scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, (train_X).shape[0], eta_min=conf['lr']*0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[int(num_epochs*0.65),int(num_epochs*0.85)], gamma=0.05)

    plot_file = open("{}/err.txt".format(dir),'w')
    plot_file.write("p_train_err:n_train_err:p_val_err:n_val_err\n")

    p_best_val_err = 999.9
    n_best_val_err = 999.9

    train_crit = nn.MSELoss()
    eval_crit = nn.L1Loss()
    for epoch in range(num_epochs):
        train_loss = train(model, opt, dev, train_crit, train_X, train_Y, scheduler, conf['lw'])
        print('Epoch #{}, training loss {:.5f}'.format(epoch,train_loss))
        if epoch % 5 == 0:
            with torch.no_grad():
                p_train_err, n_train_err = evaluate(model, dev, eval_crit, train_X, train_Y)
                p_val_err, n_val_err = evaluate(model, dev, eval_crit, val_X, val_Y)
                #val_err = p_val_err + n_val_err
                #train_err = p_train_err + n_train_err
                plot_file.write("{:.5f}:{:.5f}:{:.5f}:{:.5f}\n".format(p_train_err, n_train_err, p_val_err, n_val_err))

                #if val_err < best_val_err:
                if p_val_err < p_best_val_err and n_val_err < n_best_val_err:
                    p_best_val_err, n_best_val_err = p_val_err, n_val_err
                    #torch.save(model.state_dict(),'{}/model.pth'.format(dir))
                    torch.save(model,'{}/best_model.pkl'.format(dir))
                    print('Winner!')
                print('Epoch #{}. Train err: p: {:.5f}, n: {:.5f}'.format(epoch, p_train_err, n_train_err))
                print('Validation err: p: {:.5f}(best: {:.5f}), n: {:.5f}(best: {:.5f})'.format(p_val_err, p_best_val_err, n_val_err, n_best_val_err))

    torch.save(model,'{}/final_model.pkl'.format(dir))
    plot_file.close()


if __name__ == '__main__':
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data #
    data_source = "kin"
    val_X = np.load('{}/{}/new/{}_x_1024_ra.npy'.format("input",data_source,"val"),allow_pickle=True).astype('float32')
    val_Y = np.load('{}/{}/new/{}_y.npy'.format("input",data_source,"val"),allow_pickle=True)[:,:].astype('float32')

    # load parameters
    experiment_name = 'all_sa'
    iterations = list(range(10))
    with open('experiments/{}.json'.format(experiment_name)) as f:
        confs = json.load(f)
        for iteration in iterations[:]:
            for conf_idx, conf in enumerate(confs[:]):
                print("ITERATION {}, CONF IDX {}".format(iteration,conf_idx))
                print(conf)
                experiment_dir = os.path.join(local_path,"ne-{}_sa-{}_sh-{}_da-{}_ar-{}_lw-{}_op-{}_lr-{}_nn-{}_it-{}"
                                              .format(conf['ne'],conf['sa'],conf['sh'],conf['da'],conf['ar'],
                                                      conf['lw'],conf['op'],conf['lr'],conf['nn'],iteration))
                if not os.path.exists(experiment_dir):
                    os.makedirs(experiment_dir)
                # check if experiment has been done before
                if os.path.exists(os.path.join(experiment_dir,"err_p.png")):
                    print("ALREADY DONE: {}".format(experiment_dir))
                    continue

                train_X = np.load('{}/{}/new/{}_x_1024_ra.npy'.format("input",data_source,"train"),allow_pickle=True).astype('float32')
                train_Y = np.load('{}/{}/new/{}_y.npy'.format("input",data_source,"train"),allow_pickle=True)[:,:].astype('float32')

                if conf['sa'] == "furthest":
                    print("using furthest sampling")
                    perm = np.load('{}/{}/new/ae/furthest_point.npy'.format("input",data_source))
                    train_X, train_Y = train_X[perm], train_Y[perm]

                #print("{}, {}, {}, {}".format(val_X.shape, val_Y.shape, train_X.shape, train_Y.shape))

                # train #
                if conf['da'] == "xyz":
                    train_x, val_x = torch.from_numpy(train_X[:,:,:3]), torch.from_numpy(val_X[:,:,:3])
                elif conf['da'] == "rgb":
                    indices = [0,1,2,6,7,8]
                    train_x, val_x = torch.from_numpy(train_X[:,:,indices]), torch.from_numpy(val_X[:,:,indices])
                    #train_x[:,:,3:], val_x[:,:,3:] = train_x[:,:,3:]/255.0, val_x[:,:,3:]/255.0
                elif conf['da'] == "nxyz":
                    train_x, val_x = torch.from_numpy(train_X[:,:,:6]), torch.from_numpy(val_X[:,:,:6])

                print("# training samples {}".format(int(len(train_x)*conf['sh'])))
                train_x, train_y = train_x[:int(len(train_x)*conf['sh'])], torch.from_numpy(train_Y[:int(len(train_Y)*conf['sh'])]),
                val_x, val_y = val_x[:int(len(val_x))], torch.from_numpy(val_Y[:int(len(val_Y))])

                train_loop(conf, experiment_dir, dev, train_x, train_y, val_x, val_y)

                # plot #
                plot_loss(experiment_dir)
