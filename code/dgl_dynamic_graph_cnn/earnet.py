import numpy as np
from torch.utils.data import Dataset
import json
import pandas as pd
from pyntcloud import PyntCloud
import random
from in_out import load_x_y, sample_x


class EarNet(object):
    def __init__(self,path='/home/datasets/'):
        self.path = path
    def train(self,split):
        return EarNetDataset(self,dataset='train',path=self.path,split=split)

    def val(self):
        return EarNetDataset(self,dataset='val',path=self.path)

    def test(self):
        return EarNetDataset(self,dataset='test',path=self.path)

class EarNetDataset(Dataset):
    def __init__(self,modelnet,dataset='train',path='data/',use_npy=False,split=None):
        super(EarNetDataset, self).__init__()
        if use_npy:
            self.data = np.load('{}/{}x.npy'.format(path,dataset),allow_pickle=True)
            self.label = np.load('{}/{}trainy.npy'.format(path,dataset),allow_pickle=True)[:,:].astype('float32')
        else:
            with open('{}/{}_docker.txt'.format(path,dataset)) as f:
                pc_files = f.read().splitlines()
                if split:
                    self.data, self.label = load_x_y(pc_files[:split])
                else:
                    self.data, self.label = load_x_y(pc_files)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        x = sample_x(self.data[i])
        y = self.label[i]
        #if self.mode == 'train':
            #x = self.translate(x)
            #np.random.shuffle(x)
        return x, y
