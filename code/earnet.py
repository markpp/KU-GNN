import numpy as np
from torch.utils.data import Dataset
import json
import pandas as pd
from pyntcloud import PyntCloud
import random
from in_out import load_x_y, sample_x


class EarNetDataset(Dataset):
    def __init__(self,dataset='train',path='input',split=None):
        super(EarNetDataset, self).__init__()
        self.data = np.load('{}/{}_x.npy'.format(path,dataset),allow_pickle=True)
        self.label = np.load('{}/{}_y.npy'.format(path,dataset),allow_pickle=True)[:,:].astype('float32')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        x = self.data[i]
        y = self.label[i]
        #if self.mode == 'train':
            #x = self.translate(x)
            #np.random.shuffle(x)
        return x, y
