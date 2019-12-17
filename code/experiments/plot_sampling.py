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

    p_idx, n_idx = np.argmin(np.array(val_errs_p)), np.argmin(np.array(val_errs_n))
    return train_errs_p[p_idx], train_errs_n[n_idx], val_errs_p[p_idx], val_errs_n[n_idx]


def efficiency(train_p, train_n, val_p, val_n, name):
    '''
    tmp = pd.DataFrame({'name':[0,1],
                        'dataset':['train','val'],
                        'p_err':[train_p,val_p],
                        'n_err':[train_n,val_n],
                        'epochs':[float(name.split('-')[0]),float(name.split('-')[0])]})
    '''
    tmp = pd.DataFrame()
    tmp['p_err'] = np.concatenate((train_p,val_p),axis=0)
    tmp['n_err'] = np.concatenate((train_n,val_n),axis=0)
    tmp['name'] = np.concatenate(([name]*1,[name]*1),axis=0)
    tmp['dataset'] = np.concatenate((['train']*1,['val']*1),axis=0)
    tmp['epochs'] = np.concatenate(([float(name.split('-')[0])],[float(name.split('-')[0])]),axis=0)

    return tmp

if __name__ == '__main__':

    experiment_name = 'sampling'
    iterations = list(range(2))

    df = pd.DataFrame()

    names = []
    datasets = []
    p_errs = []
    n_errs = []
    shares = []

    # load json
    with open('{}.json'.format(experiment_name)) as f:
      confs = json.load(f)
      for conf in confs[:]:

        for iteration in iterations[:]:
          experiment_dir = "{}/id-{}_it-{}".format(experiment_name,conf['id'],iteration)
          train_p, train_n, val_p, val_n = read_results(os.path.join(experiment_dir,"err.txt"),mode=experiment_name)
          #print("best: val_p {}, val_n {}, share {}".format(val_p,val_n,conf['share']))
          names.append(conf['net']+"-"+conf['samp'])
          datasets.append('val')
          p_errs.append(val_p)
          n_errs.append(val_n)
          shares.append(conf['share'])

          names.append(conf['net']+"-"+conf['samp'])
          datasets.append('train')
          p_errs.append(train_p)
          n_errs.append(train_n)
          shares.append(conf['share'])

    #print(df.describe())


    sns.lineplot(x=shares, y=p_errs, hue=names, style=datasets)

    #plt.yscale('log')
    plt.xlim(0.25,1.0)
    #plt.ylim(0.005,0.01)
    plt.savefig("{}_p_err.png".format(experiment_name))
    plt.close()

    sns.lineplot(x=shares, y=n_errs, hue=names, style=datasets)
    plt.xlim(0.25,1.0)
    #plt.ylim(0.02,0.08)
    plt.savefig("{}_n_err.png".format(experiment_name))

    #plt.show()
    #plt.close()
