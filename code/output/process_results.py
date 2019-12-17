import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import numpy as np
import pandas as pd
import json

def plot_loss(histories, name, key='loss'):
    plt.figure(figsize=(8,4))

    for label, history in histories:
        plt.plot(history.epoch, history.history['val_'+key], color='blue', label=label.title()+' Val')
        plt.plot(history.epoch, history.history[key], color='green', label=label.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.ylim([0,0.1])
    #plt.yscale('log')
    plt.savefig('plots/pose_loss_{}.png'.format(name))
    #plt.show()
    plt.close()

def train_val_err(train, val, step_size=5, backend='sns', path='err.png'):
    epochs = list(range(0,len(train)*step_size,step_size))
    plt.figure(figsize=(8,4))

    if backend == 'sns':
        df = pd.DataFrame()
        df['err'] = np.concatenate((train,val),axis=0)
        df['name'] = np.concatenate((['train']*len(train),['val']*len(val)),axis=0)
        #print(df['name'])
        df['epochs'] = np.concatenate((epochs,epochs),axis=0)
        lp = sns.lineplot(x="epochs", y="err", hue="name", data=df)
        #lp.set(xlim=(0,epochs[-1]))
    else:
        plt.plot(epochs, train, color='blue', label='Val')
        plt.plot(epochs, val, color='green', label='Train')
        plt.xlabel('Epochs')
        plt.ylabel('Err')
        plt.legend()
        #plt.xlim([0,epochs[-1]])

    #plt.ylim([0,0.1])
    #plt.yscale('log')
    plt.savefig(path)
    #plt.show()
    #plt.close()

def read_results(res_txt):
    train = []
    val = []

    f = open(res_txt, "r")
    for line in f.readlines()[1:]:
        train_err, val_err = line.rstrip().split(':')
        train.append(float(train_err))
        val.append(float(val_err))

    return np.array(train), np.array(val)

if __name__ == '__main__':

    experiment_name = 'consistency'
    iterations = list(range(9))

    
    df = pd.DataFrame(columns = ['name','err','epochs'])

    # load json
    with open('../{}.json'.format(experiment_name)) as f:
      confs = json.load(f)
      for conf in confs[2:]:
        for iteration in iterations:
          experiment_dir = "{}_{}_{}".format(experiment_name,conf['id'],iteration)
          #print(experiment_dir)
          train, val = read_results(os.path.join(experiment_dir,"err_{}.txt".format(conf['rep'])))
          #print(train.min)
          tmp = pd.DataFrame()
          epochs = list(range(0,len(train)*5,5))
          tmp['err'] = np.concatenate((train,val),axis=0)
          tmp['name'] = np.concatenate((['train']*len(train),['val']*len(val)),axis=0)
          tmp['epochs'] = np.concatenate((epochs,epochs),axis=0)
          df = df.append(tmp)

    print(df.describe())
    #sns.pointplot(x="epochs", y="train", data=df)
    sns.lineplot(x="epochs", y="err", hue="name", data=df)
    plt.yscale('log')
    plt.show()

    # find relevant folders

    # load parameters




    #train_val_err(train, val, backend='sns', path='err_n.png')
