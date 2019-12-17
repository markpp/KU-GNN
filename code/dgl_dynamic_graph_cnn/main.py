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


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='')
parser.add_argument('--num-epochs', type=int, default=96)
parser.add_argument('--num-workers', type=int, default=6)
parser.add_argument('--batch-size', type=int, default=64)
args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size
local_path = args.dataset_path or os.path.join(os.getcwd(), "output")

TrainDataLoader = partial(
        DataLoader,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

TestDataLoader = partial(
        DataLoader,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False)

def train(model, opt, train_loader, dev):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0
    num_batches = 0
    with tqdm.tqdm(train_loader, ascii=True) as tq:
        for data, label in tq:
            num_examples = label.shape[0]
            data, label = data.to(dev), label.to(dev)
            opt.zero_grad()
            if rep_selection == 'p':
                p = model(data)
                p_loss = criterion(p, label[:,:3])
                loss = p_loss
            elif rep_selection == 'n':
                n = model(data)
                n_loss = criterion(n, label[:,3:])
                loss = n_loss
            else:
                pn = model(data)
                p_loss = criterion(pn[:,:3], label[:,:3])
                n_loss = criterion(pn[:,3:], label[:,3:])
                loss = p_loss + n_loss
            loss.backward()
            opt.step()

            loss_ = loss.item()
            total_loss += loss_
            num_batches += 1
            epoch_loss = total_loss / num_batches

            for param_group in opt.param_groups:
                lr = param_group['lr']
            tq.set_postfix({
                'LR': '%.5f' % lr,
                'BatchLoss': '%.5f' % loss_,
                'EpochLoss': '%.5f' % epoch_loss})
    return epoch_loss

def evaluate(model, eval_loader, dev):
    model.eval()
    criterion = nn.L1Loss()
    total_error = 0
    num_batches = 0

    with torch.no_grad():
        with tqdm.tqdm(eval_loader, ascii=True) as tq:
            for data, label in tq:
                num_examples = label.shape[0]
                data, label = data.to(dev), label.to(dev)

                if rep_selection == 'p':
                    p = model(data)
                    err = criterion(p, label[:,:3])
                elif rep_selection == 'n':
                    n = model(data)
                    err = criterion(n, label[:,3:])
                else:
                    pn = model(data)
                    p_err = criterion(pn[:,:3], label[:,:3])
                    n_err = criterion(pn[:,3:], label[:,3:])
                    err = p_err + n_err

                total_error += err
                num_batches += 1

                epoch_error = total_error / num_batches
                tq.set_postfix({
                    'BatchErr': '%.5f' % err,
                    'EpochErr': '%.5f' % (epoch_error)})
    return epoch_error

if __name__ == '__main__':
    rep_selection = ['p','n','pn'][1]

    with open('/home/datasets/train_docker.txt') as f:
        num_train = len(f.read().splitlines())

    variables = [num_train//64, num_train//32, num_train//16, num_train//8, num_train//4, num_train//2, num_train][2:]
    print("variables: {}".format(variables))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")##

    modelnet = EarNet(path='/home/datasets')
    val_loader = TestDataLoader(modelnet.val())
    test_loader = TestDataLoader(modelnet.test())

    for var in variables:
        print(var)
        train_loader = TrainDataLoader(modelnet.train(split=var))

        model = Model(10, [64, 64, 128, 256], [512, 512, 256], output_dims=6, rep=rep_selection)
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

        for epoch in range(args.num_epochs):
            print('Epoch #%d Training' % epoch)
            train_loss = train(model, opt, train_loader, dev)
            if epoch % 5 == 0:
                print('Epoch #%d Validating' % epoch)
                #train_err = evaluate(model, train_loader, dev)
                val_err = evaluate(model, val_loader, dev)
                test_err = evaluate(model, test_loader, dev)
                plot_file.write("{:.5f}:{:.5f}\n".format(val_err,test_err))
                if val_err < best_val_err:
                    best_val_err = val_err
                    best_test_err = test_err
                    torch.save(model.state_dict(),'{}/model_{}.pth'.format(experiment_dir,rep_selection))
                    torch.save(model,'{}/model_{}.pkl'.format(experiment_dir,rep_selection))
                print('Current validation err: %.5f (best: %.5f), test err: %.5f (best: %.5f)' % (val_err, best_val_err, test_err, best_test_err))
        plot_file.close()

        '''
        print("Model's parameters:")
        print("total parms: {}".format(sum(p.numel() for p in model.parameters())))
        print("trainable parms: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in opt.state_dict():
            print(var_name, "\t", opt.state_dict()[var_name])
        '''
