from tensorflow.keras import models

import os
import argparse
import numpy as np
from in_out import load_x_y, sample_xs


from itertools import islice
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


if __name__ == '__main__':

    model_dir = 'output'
    data_dir = '/home/dmri/datasets/supervised/'

    folder_list = ['255','510','1020','2040','4080','8160','16320','paper'][6:]


    for dataset in ['test','val','train'][:]:
        print(dataset)

        with open('{}/{}.txt'.format(data_dir,dataset)) as f:
            if dataset == 'train':
                pc_files = f.read().splitlines()[::5]
            else:
                pc_files = f.read().splitlines()

        print("predicting {} files".format(len(pc_files)))
        data, label = load_x_y(pc_files)
        #np.save("{}/gt_{}.npy".format(model_dir,dataset),data)
        np.save("{}/gt_{}.npy".format(model_dir,dataset),label)

        data = sample_xs(data)

        for folder in folder_list:

            model_p = models.load_model(os.path.join(model_dir,"{}/pointnet_p.h5".format(folder)), compile=False)
            model_n = models.load_model(os.path.join(model_dir,"{}/pointnet_n.h5".format(folder)), compile=False)
            model_p._make_predict_function()
            model_n._make_predict_function()

            pred = np.empty((0,6), float)
            for batch_idx in list(chunk(range(len(data)), 64)):
                batch_idx = list(batch_idx)
                x = data[batch_idx]

                pred_p = model_p.predict(x)
                pred_n = model_n.predict(x)

                tmp_pred = np.concatenate((pred_p, pred_n), axis=1)
                pred = np.append(pred, tmp_pred, axis=0)
                print(pred.shape)

            np.save("{}/{}/pred_{}.npy".format(model_dir,folder,dataset),pred)
