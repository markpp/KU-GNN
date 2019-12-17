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

TRAIN = 0

rep_selection = ['p','n','pn'][1]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='')
parser.add_argument('--load-model-path', type=str, default='models/model_{}.pth'.format(rep_selection))
parser.add_argument('--save-model-path', type=str, default='models/model_{}.pth'.format(rep_selection))
parser.add_argument('--num-epochs', type=int, default=15)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=48)
args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size
local_path = args.dataset_path or os.path.join(os.getcwd(), "lists")

CustomDataLoader = partial(
        DataLoader,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

def train(model, opt, scheduler, train_loader, dev):
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
                p, n = model(data)
                p_loss = criterion(p, label[:,:3])
                n_loss = criterion(n, label[:,3:])
                loss = p_loss + n_loss
            loss.backward()
            opt.step()
            scheduler.step()

            loss_ = loss.item()
            total_loss += loss_
            num_batches += 1

            for param_group in opt.param_groups:
                lr = param_group['lr']
            tq.set_postfix({
                'LR': '%.5f' % lr,
                'BatchLoss': '%.5f' % loss_,
                'EpochLoss': '%.5f' % (total_loss / num_batches)})

    return (total_loss / num_batches)

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
                    p, n = model(data)
                    p_err = criterion(p, label[:,:3])
                    n_err = criterion(n, label[:,3:])
                    err = p_err + n_err

                total_error += err
                num_batches += 1

                tq.set_postfix({
                    'BatchErr': '%.5f' % err,
                    'EpochErr': '%.5f' % (total_error / num_batches)})

    return (total_error / num_batches)

def predict(model, dev):
    model.eval()
    data = torch.from_numpy(np.load('data/testx.npy',allow_pickle=True)[:])
    data = data.to(dev)
    label = torch.from_numpy(np.load('data/testy.npy',allow_pickle=True)[:].astype('float32'))
    label = label.to(dev)
    #writer = SummaryWriter('runs/experiment_1')
    #writer.add_graph(model, data)
    #writer.close()

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
    np.save("pred_{}.npy".format(rep_selection),pred)


model = Model(10, [64, 64, 128, 256], [512, 512, 256], output_dims=48, rep=rep_selection)

modelnet = EarNet(local_path, 1024)

if TRAIN:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")##
    model = model.to(dev)

    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.num_epochs, eta_min=0.0001)

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
    train_loader = CustomDataLoader(modelnet.train())
    valid_loader = CustomDataLoader(modelnet.valid())
    test_loader = CustomDataLoader(modelnet.test())

    best_valid_err = 999.9
    best_test_err = 999.9

    plot_file = open("{}_err.txt".format(rep_selection),'w')
    plot_file.write("valid_err:test_err\n")

    for epoch in range(args.num_epochs):
        print('Epoch #%d Training' % epoch)
        _ = train(model, opt, scheduler, train_loader, dev)
        if epoch % 1 == 0:
            print('Epoch #%d Validating' % epoch)
            #train_err = evaluate(model, train_loader, dev)
            valid_err = evaluate(model, valid_loader, dev)
            test_err = evaluate(model, test_loader, dev)

            plot_file.write("{:.5f}:{:.5f}\n".format(valid_err,test_err))
            print(valid_err)
            if valid_err < best_valid_err:
                best_valid_err = valid_err
                best_test_err = test_err
                if args.save_model_path:
                    torch.save(model.state_dict(), args.save_model_path)
                    torch.save(model, 'models/model_{}.pkl'.format(rep_selection))
            print('Current validation err: %.5f (best: %.5f), test err: %.5f (best: %.5f)' % (valid_err, best_valid_err, test_err, best_test_err))

    plot_file.close()
else:
    '''
    # extract weights
    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))
    
    l1_w = model.l1.weight.detach().numpy()
    l2_w = model.l2.weight.detach().numpy()
    l3_w = model.l3.weight.detach().numpy()

    print(l1_w.shape)
    print(l2_w.shape)
    print(l3_w.shape)
    print(l3_w)
    np.save("l1_w.npy",l1_w)
    np.save("l2_w.npy",l2_w)
    np.save("l3_w.npy",l3_w)

    '''
    dev = torch.device("cpu")
    model = model.to(dev)

    #test_loader = CustomDataLoader(modelnet.test())

    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path, map_location=dev))

    predict(model, dev)
