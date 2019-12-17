import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import numpy as np
import pandas as pd
import json


def read_results(res_txt, mode='other'):
    train_errs_p = []
    train_errs_n = []
    val_errs_p = []
    val_errs_n = []

    f = open(res_txt, "r")
    for line in f.readlines()[1:]:
        train_err_p, train_err_n, val_err_p, val_err_n = line.rstrip().split(':')
        train_errs_p.append(float(train_err_p))
        train_errs_n.append(float(train_err_n))
        val_errs_p.append(float(val_err_p))
        val_errs_n.append(float(val_err_n))

    if mode == 'efficiency':
        p_idx, n_idx = np.argmin(np.array(val_errs_p)), np.argmin(np.array(val_errs_n))
        return [train_errs_p[p_idx]], [train_errs_n[n_idx]], [val_errs_p[p_idx]], [val_errs_n[n_idx]]
    else:
        return train_errs_p, train_errs_n, val_errs_p, val_errs_n

def datatypes(train_p, train_n, val_p, val_n, name):
    tmp = pd.DataFrame()
    epochs = list(range(0,len(train_p)*5,5))
    tmp['p_err'] = np.concatenate((train_p,val_p),axis=0)
    tmp['n_err'] = np.concatenate((train_n,val_n),axis=0)
    tmp['name'] = np.concatenate(([name]*len(train_p),[name]*len(val_p)),axis=0)
    tmp['dataset'] = np.concatenate((['train']*len(train_p),['val']*len(val_p)),axis=0)
    tmp['epochs'] = np.concatenate((epochs,epochs),axis=0)
    return tmp

def architecture(train_p, train_n, val_p, val_n, name):
    tmp = pd.DataFrame()
    epochs = list(range(0,len(train_p)*5,5))
    tmp['p_err'] = np.concatenate((train_p,val_p),axis=0)
    tmp['n_err'] = np.concatenate((train_n,val_n),axis=0)
    tmp['name'] = np.concatenate(([name]*len(train_p),[name]*len(val_p)),axis=0)
    tmp['dataset'] = np.concatenate((['train']*len(train_p),['val']*len(val_p)),axis=0)
    tmp['epochs'] = np.concatenate((epochs,epochs),axis=0)
    return tmp



if __name__ == '__main__':

    experiment_name = ['architecture', 'sampling', 'datatype'][1]
    iterations = list(range(2))

    df = pd.DataFrame()

    # load json
    with open('{}.json'.format(experiment_name)) as f:
      confs = json.load(f)
      for conf in confs[:]:

        for iteration in iterations[:]:
          experiment_dir = "{}/id-{}_it-{}".format(experiment_name,conf['id'],iteration)
          train_p, train_n, val_p, val_n = read_results(os.path.join(experiment_dir,"err.txt"),mode=experiment_name)
          #tmp = architecture(train_p, train_n, val_p, val_n, name = conf['arch'] + "-" + conf['net'])
          #tmp = datatypes(train_p, train_n, val_p, val_n, name = conf['data'] + "-" + conf['net'])
          df = df.append(tmp)


    print(df.describe())


    #sns.pointplot(x="epochs", y="train", data=df)
    #sns.lineplot(x="epochs", y="p_err", hue="name", style="dataset", data=df)
    sns.lineplot(x=df['epochs'].values, y=df['p_err'].values, hue=df['name'].values, style=df['dataset'].values)

    #plt.yscale('log')
    #plt.ylim(0.0,0.02)
    plt.savefig("{}_p_err.png".format(experiment_name))
    plt.close()

    sns.lineplot(x="epochs", y="n_err", hue="name", style="dataset", data=df)
    #plt.ylim(0.0,0.16)
    plt.savefig("{}_n_err.png".format(experiment_name))

    #plt.show()
    #plt.close()
