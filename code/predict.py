import torch
import json
import argparse
import numpy as np
import os
from itertools import islice

local_path = os.path.join(os.getcwd(), "experiments/gnn")


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
    test_X = np.load('{}/{}/new/{}_x_1024_ra.npy'.format("input",data_source,"test"),allow_pickle=True).astype('float32')
    test_Y = np.load('{}/{}/new/{}_y.npy'.format("input",data_source,"test"),allow_pickle=True)[:,:].astype('float32')
    val_X = np.load('{}/{}/new/{}_x_1024_ra.npy'.format("input",data_source,"val"),allow_pickle=True).astype('float32')
    val_Y = np.load('{}/{}/new/{}_y.npy'.format("input",data_source,"val"),allow_pickle=True)[:,:].astype('float32')
    train_X = np.load('{}/{}/new/{}_x_1024_ra.npy'.format("input",data_source,"train"),allow_pickle=True).astype('float32')
    train_Y = np.load('{}/{}/new/{}_y.npy'.format("input",data_source,"train"),allow_pickle=True)[:,:].astype('float32')

    # load parameters
    experiment_name = 'pn_sa'
    iterations = list(range(8))

    with torch.no_grad():
        done = False
        with open('experiments/{}.json'.format(experiment_name)) as f:
            confs = json.load(f)
            for iteration in iterations[:]:
                if done:
                    break
                print("iteration {}".format(iteration))
                for conf in confs[:]:
                    print(conf)
                    experiment_dir = os.path.join(local_path,"ne-{}_sa-{}_sh-{}_da-{}_ar-{}_lw-{}_op-{}_lr-{}_nn-{}_it-{}"
                                                             .format(conf['ne'],conf['sa'],conf['sh'],conf['da'],conf['ar'],
                                                             conf['lw'],conf['op'],conf['lr'],conf['nn'],iteration))
                    model_path = os.path.join(experiment_dir,"final_model.pkl")
                    if not os.path.exists(model_path):
                        print("model has not finished training: {}".format(experiment_dir))
                        done = True
                        break
                        #continue
                    # check if predictions have been done before
                    if os.path.exists(os.path.join(experiment_dir,"model.txt")):
                        print("ALREADY DONE: {}".format(experiment_dir))
                        continue

                    # predict #
                    if conf['da'] == "xyz":
                        train_x, val_x, test_x = torch.from_numpy(train_X[:,:,:3]), torch.from_numpy(val_X[:,:,:3]), torch.from_numpy(test_X[:,:,:3])
                    elif conf['da'] == "rgb":
                        indices = [0,1,2,6,7,8]
                        train_x, val_x, test_x = torch.from_numpy(train_X[:,:,indices]), torch.from_numpy(val_X[:,:,indices]), torch.from_numpy(test_X[:,:,indices])
                        #train_x[:,:,3:], val_x[:,:,3:], test_x[:,:,3:] = train_x[:,:,3:]/255.0, val_x[:,:,3:]/255.0, test_x[:,:,3:]/255.0
                    elif conf['da'] == "nxyz":
                        train_x, val_x, test_x = torch.from_numpy(train_X[:,:,:6]), torch.from_numpy(val_X[:,:,:6]), torch.from_numpy(test_X[:,:,:6])
                    else:
                        print("{} is not acceptable".format(conf['da']))

                    np.save("{}/gt_train.npy".format(experiment_dir),train_Y)
                    np.save("{}/gt_val.npy".format(experiment_dir),val_Y)
                    np.save("{}/gt_test.npy".format(experiment_dir),test_Y)
                    #train_y, val_y, test_y = torch.from_numpy(train_Y), torch.from_numpy(val_Y), torch.from_numpy(test_Y)

                    datasets = []
                    datasets.append(["train",train_x])
                    datasets.append(["val",val_x])
                    datasets.append(["test",test_x])

                    model = torch.load("{}/best_model.pkl".format(experiment_dir))
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
