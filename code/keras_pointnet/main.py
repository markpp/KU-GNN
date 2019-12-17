
from model import Point2Pose
import numpy as np

import matplotlib.pyplot as plt

import os
os.environ["TF_KERAS"] = "1"
from keras_lookahead import Lookahead


from schedules import onetenth_75_95

from tensorflow.keras.optimizers import Adam

import os, sys
import argparse
import numpy as np
from in_out import DataGenerator, load_x_y, sample_x
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import math


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='')
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=64)
args = parser.parse_args()

batch_size = args.batch_size
local_path = args.dataset_path or os.path.join(os.getcwd(), "output")

rep_selection = ['p','n','pn'][1]


def plot_history(histories, result_dir, exp_name, key='loss'):
    plt.figure(figsize=(6,4))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       color='blue', linestyle='--', label=name.title()+' Val')
        val_min = min(history.history['val_'+key])
        val_max = max(history.history['val_'+key])

        tra = plt.plot(history.epoch, history.history[key],
                       color=val[0].get_color(), label=name.title()+' Train')
        train_min = min(history.history[key])
        train_max = max(history.history[key])

    #plt.title('loss')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.yscale('log')
    plt.xlim([0,max(history.epoch)])

    # find order of magnitude for min value on y axis
    min_ofm = math.floor(math.log10(min([val_min, train_min])))
    max_ofm = math.ceil(math.log10(max([val_max, train_max])))
    plt.ylim([10**min_ofm,10**max_ofm])

    plt.grid()
    plt.savefig(os.path.join(result_dir, exp_name+'_loss_{}.png'.format(rep_selection)))
    plt.close()


if __name__ == '__main__':

    with open('/home/dmri/datasets/supervised/train.txt') as f:
        num_train = len(f.read().splitlines())

    variables = [num_train//64, num_train//32, num_train//16, num_train//8, num_train//4, num_train//2, num_train][-1:]
    print("variables: {}".format(variables))

    with open('/home/dmri/datasets/supervised/val.txt') as f:
            val = f.read().splitlines()
    val_X, val_Y = load_x_y(val)

    for var in variables:
        experiment_dir = os.path.join(local_path,"{}".format(var))
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        print(experiment_dir)

        with open('/home/dmri/datasets/supervised/train.txt') as f:
            train = f.read().splitlines()[:var]

        train_X, train_Y = load_x_y(train)

        train_gen = DataGenerator(train_X, train_Y, augment=True)
        val_gen = DataGenerator(val_X, val_Y, augment=False)

        model = Point2Pose(cloud_shape = (1024, 3))
        #print(model.summary())
        lr = 0.01

        #opt = Lookahead(Adam(lr=lr),sync_period=5,slow_step=0.5)
        opt = Adam(lr=lr)

        model.compile(optimizer=opt,
                      loss='mean_squared_error',
                      metrics=['mae'])

        lr_callback = onetenth_75_95(lr)

        checkpoint = ModelCheckpoint(os.path.join(experiment_dir,'pointnet_{}.h5'.format(rep_selection)), monitor='val_loss',
                                     save_weights_only=False, save_best_only=True, verbose=1)

        history = model.fit_generator(train_gen.generateXY(mode=rep_selection),
                                      steps_per_epoch=train_X.shape[0] // batch_size,
                                      epochs=100,
                                      validation_data=val_gen.generateXY(mode=rep_selection),
                                      validation_steps=val_X.shape[0] // batch_size,
                                      callbacks=[checkpoint,lr_callback],
                                      verbose=1)

        plot_history([('', history)], experiment_dir, 'end2end')
        #model.save(os.path.join(experiment_dir,"end2end.h5"))
