import numpy as np

import os, sys
sys.path.insert(1, os.path.realpath(os.path.pardir))

import autoencoder, enc2pose, end2end
from in_out import DataGenerator, load_x_y, sample_x
from model import Point2Pose

import matplotlib
import matplotlib.pyplot as plt

#n_aug = 10

with open('../lists/train.txt') as f:
    num_train = len(f.read().splitlines())

# number of training samples
variables = [num_train//64, num_train//32, num_train//16, num_train//8, num_train//4, num_train//2, num_train][:] #8288
print("variables: {}".format(variables))

def run(METHOD, TRAIN=1, experiment_name="noname"):
    if not os.path.exists("results"):
        os.makedirs("results")

    experiment_dir = os.path.join("results",experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    method_dir = os.path.join(experiment_dir,METHOD)
    if not os.path.exists(method_dir):
        os.makedirs(method_dir)

    val_X, val_Y = load_x_y('../lists/val.txt')

    for var in variables:
        result_dir = os.path.join(method_dir,str(var))
        print(result_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        train_X, train_Y = load_x_y('../lists/train.txt',n_samples=var)
        print("train samples {}, val samples {}".format(train_X.shape[0],val_X.shape[0]))

        if TRAIN:
            train_gen = DataGenerator(train_X, train_Y, augment=True)
            val_gen = DataGenerator(val_X, val_Y, augment=False)

            if METHOD == "end2end":
                model, encoder = Point2Pose(cloud_shape = (1024, 3), n_latent = 16)#, split="complete_split")
                end2end.fit_gen(train_gen, val_gen, 5000, val_X.shape[0], model=model, output_path=result_dir)#, split="complete_split")

        val_X, val_Y = load_x_y('../lists/val.txt')
        test_X, test_Y = load_x_y('../lists/test.txt')

        val_x = sample_x(val_X, 1024)
        test_x = sample_x(test_X, 1024)
        train_x = sample_x(train_X, 1024)

        if METHOD == "end2end":
            print("end2end evaluated on train set")
            res_names, train_res = end2end.evaluate(train_x, train_Y, model_path=result_dir)
            print(train_res)

            print("end2end evaluated on validation sets")
            res_names, val_res = end2end.evaluate(val_x, val_Y, model_path=result_dir)
            print(val_res)
            res_names, test_res = end2end.evaluate(test_x, test_Y, model_path=result_dir)
            print(test_res)
            #val_pred = end2end.predict(val_x, model_path=result_dir)

        res_file = open('{}/res.txt'.format(result_dir),"w")
        res_file.write("train_{}:{}:{}\n".format(var,res_names,train_res))
        res_file.write("val:{}:{}\n".format(res_names,val_res))
        res_file.write("test:{}:{}\n".format(res_names,test_res))
        res_file.close()

def plot_results(METHOD,metric_names,experiment_name="noname"):
    experiment_dir = os.path.join("results",experiment_name)
    method_dir = os.path.join(experiment_dir,METHOD)

    for idx, metric in enumerate(metric_names):
        train, val, test = [], [], []
        names = []
        for var in variables:
            print(var)
            name = var
            names.append(name)
            with open(os.path.join(method_dir,"{}/res.txt".format(name)), 'r') as f:
                lines = f.readlines()
                train_res = lines[0].rstrip("\n").replace('[', '').replace(']', '').split(':')[2].split(',')
                print(train_res)
                train.append(float(train_res[idx]))
                val_res = lines[1].rstrip("\n").replace('[', '').replace(']', '').split(':')[2].split(',')
                val.append(float(val_res[idx]))
                test_res = lines[2].rstrip("\n").replace('[', '').replace(']', '').split(':')[2].split(',')
                test.append(float(test_res[idx]))

        plt.plot(names, train, linestyle='--', marker='o')
        plt.plot(names, val, linestyle='--', marker='o')
        plt.plot(names, test, linestyle='--', marker='o')

        plt.title('dataset size impact on {}'.format(metric))
        plt.xlabel('# samples')
        plt.ylabel('{}'.format(metric))

        plt.grid()
        plt.legend(['train', 'val', 'test'], loc='upper right')
        plt.savefig(os.path.join(experiment_dir, 'dataset_size_vs_{}.png'.format(metric)))
        plt.close()

if __name__ == '__main__':
    TRAIN = 1
    METHOD = ["end2end","self"][0]

    #run(METHOD, TRAIN, "dataset_size")


    if METHOD == "end2end":
        #metric_names = ['loss', 'mean_absolute_error']
        metric_names = ['loss', 'center_loss', 'norm_loss', 'center_mean_absolute_error', 'norm_mean_absolute_error']
        plot_results(METHOD,metric_names,"dataset_size")
    elif METHOD == "self":
        metric_names = ['rec_loss']
        plot_results(METHOD,metric_names,"dataset_size")
