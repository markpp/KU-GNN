import torch
import json
import argparse
import numpy as np
import os
from itertools import islice

local_path = os.path.join(os.getcwd(), "experiments")


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def print_model_info(model, path):
    file = open(path,'w')
    file.write("Model's parameters:\n")
    file.write("total parms: {}\n".format(sum(p.numel() for p in model.parameters())))
    file.write("trainable parms: {}\n".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    file.write("Model's state_dict:\n")
    for param_tensor in model.state_dict():
        file.write("{}, \t {}".format(param_tensor,model.state_dict()[param_tensor].size()))
    file.close()

if __name__ == '__main__':

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data #
    data_source = "kin"
    test_X = np.load('{}/{}/{}_x_1024.npy'.format("input",data_source,"test"),allow_pickle=True)
    test_Y = np.load('{}/{}/{}_y.npy'.format("input",data_source,"test"),allow_pickle=True)[:,:].astype('float32')
    val_X = np.load('{}/{}/{}_x_1024.npy'.format("input",data_source,"val"),allow_pickle=True)
    val_Y = np.load('{}/{}/{}_y.npy'.format("input",data_source,"val"),allow_pickle=True)[:,:].astype('float32')
    train_X = np.load('{}/{}/{}_x_1024.npy'.format("input",data_source,"train"),allow_pickle=True)
    train_Y = np.load('{}/{}/{}_y.npy'.format("input",data_source,"train"),allow_pickle=True)[:,:].astype('float32')

    # load parameters
    experiment_name = 'efficiency'
    iterations = list(range(6))

    with torch.no_grad():

        with open('experiments/{}.json'.format(experiment_name)) as f:
            confs = json.load(f)
            for iteration in iterations[:3]:
                print("iteration {}".format(iteration))
                for conf in confs[:]:
                    print(conf)
                    experiment_dir = os.path.join(local_path,"{}/id-{}_it-{}".format(experiment_name,conf['id'],iteration))
                    if not os.path.exists(experiment_dir):
                        print("dir: {} does not exist".format(experiment_dir))

                    # predict #
                    if conf['data'] == "xyz":
                        train_x, val_x, test_x = torch.from_numpy(train_X[:,:,:3]), torch.from_numpy(val_X[:,:,:3]), torch.from_numpy(test_X[:,:,:3])

                    if conf['data'] == "rgb":
                        train_x, val_x, test_x = torch.from_numpy(train_X[:,:,:]), torch.from_numpy(val_X[:,:,:]), torch.from_numpy(test_X[:,:,:])
                        train_x[:,:,3:], val_x[:,:,3:], test_x[:,:,3:] = train_x[:,:,3:]/255.0, val_x[:,:,3:]/255.0, test_x[:,:,3:]/255.0

                    np.save("{}/gt_train.npy".format(experiment_dir),train_Y)
                    np.save("{}/gt_val.npy".format(experiment_dir),val_Y)
                    np.save("{}/gt_test.npy".format(experiment_dir),test_Y)
                    #train_y, val_y, test_y = torch.from_numpy(train_Y), torch.from_numpy(val_Y), torch.from_numpy(test_Y)

                    datasets = []
                    datasets.append(["train",train_x])
                    datasets.append(["val",val_x])
                    datasets.append(["test",test_x])

                    model = torch.load("{}/model.pkl".format(experiment_dir))
                    model.eval()
                    model = model.to(dev)

                    print_model_info(model, "{}/model.txt".format(experiment_dir))

                    for data in datasets:
                        #print("{}, x {}".format(data[0],data[1].shape))
                        preds = np.empty((0,6), float)
                        feats = np.empty((0,1024), float)

                        for batch_idx in list(chunk(range(len(data[1])), 64)):
                            batch_idx = list(batch_idx)
                            x = data[1][batch_idx].to(dev)
                            pred_p, feat_p, pred_n, feat = model(x)
                            pred = torch.cat([pred_p, pred_n], 1)
                            if dev.type == 'cuda':
                                pred = pred.cpu()
                                feat = feat.cpu()
                            pred = pred.data.numpy()
                            feat = feat.data.numpy()
                            preds = np.append(preds, pred, axis=0)
                            feats = np.append(feats, feat, axis=0)

                        np.save("{}/pred_{}.npy".format(experiment_dir,data[0]),preds)
                        np.save("{}/feat_{}.npy".format(experiment_dir,data[0]),feats)
