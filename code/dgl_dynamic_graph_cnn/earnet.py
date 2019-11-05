import numpy as np
from torch.utils.data import Dataset

class EarNet(object):
    def __init__(self, path, num_points=1024):
        self.num_points = num_points

    def train(self):
        return EarNetDataset(self, 'train')

    def valid(self):
        return EarNetDataset(self, 'valid')

    def test(self):
        return EarNetDataset(self, 'test')

class EarNetDataset(Dataset):
    def __init__(self, modelnet, mode):
        super(EarNetDataset, self).__init__()
        self.num_points = modelnet.num_points
        self.mode = mode

        if mode == 'train':
            self.data = np.load('data/trainx.npy',allow_pickle=True)
            self.label = np.load('data/trainy.npy',allow_pickle=True)[:,:].astype('float32')
            print(self.data.shape)#(7872, 2048, 3)
            print(self.label.shape)#(7872, 1)
        elif mode == 'valid':
            self.data = np.load('data/valx.npy',allow_pickle=True)
            self.label = np.load('data/valy.npy',allow_pickle=True)[:,:].astype('float32')
        elif mode == 'test':
            self.data = np.load('data/testx.npy',allow_pickle=True)
            self.label = np.load('data/testy.npy',allow_pickle=True)[:,:].astype('float32')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        x = self.data[i]
        y = self.label[i]
        #if self.mode == 'train':
            #x = self.translate(x)
            #np.random.shuffle(x)
        return x, y
