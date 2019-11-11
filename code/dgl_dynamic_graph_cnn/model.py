import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import KNNGraph, EdgeConv


class Model(nn.Module):
    def __init__(self, k, feature_dims, emb_dims, output_dims, input_dims=3,
                 dropout_prob=0.5, rep='pn'):
        super(Model, self).__init__()

        self.nng = KNNGraph(k)
        self.conv = nn.ModuleList()

        self.num_layers = len(feature_dims)
        for i in range(self.num_layers):
            self.conv.append(EdgeConv(
                feature_dims[i - 1] if i > 0 else input_dims,
                feature_dims[i],
                batch_norm=True))

        self.proj = nn.Linear(sum(feature_dims), emb_dims[0])

        self.embs = nn.ModuleList()
        self.bn_embs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.num_embs = len(emb_dims) - 1
        for i in range(1, self.num_embs + 1):
            self.embs.append(nn.Linear(
                # * 2 because of concatenation of max- and mean-pooling
                emb_dims[i - 1] if i > 1 else (emb_dims[i - 1] * 2),
                emb_dims[i]))
            self.bn_embs.append(nn.BatchNorm1d(emb_dims[i]))
            self.dropouts.append(nn.Dropout(dropout_prob))

        self.proj_output = nn.Linear(emb_dims[-1], output_dims)

        self.rep = rep # selects output representation

        '''
        enc_dims = [16, 8, 3]
        self.encs = nn.ModuleList()
        self.bn_encs = nn.ModuleList()
        #self.dropouts = nn.ModuleList()
        self.num_encs = len(enc_dims)
        for i in range(self.num_encs):
            self.encs.append(nn.Linear(enc_dims[i - 1] if i > 0 else output_dims,
                                       enc_dims[i]))
            self.bn_encs.append(nn.BatchNorm1d(emb_dims[i]))
        self.enc_output = nn.Linear(emb_dims[-1], 3)
        '''
        self.l1 = nn.Linear(output_dims, 16)
        self.b1 = nn.BatchNorm1d(16)
        self.l2 = nn.Linear(16, 8)
        self.b2 = nn.BatchNorm1d(8)
        self.l3 = nn.Linear(8, 3)

    def forward(self, x):
        hs = []
        batch_size, n_points, x_dims = x.shape
        h = x

        for i in range(self.num_layers):
            g = self.nng(h)
            h = h.view(batch_size * n_points, -1)
            h = self.conv[i](g, h)
            h = F.leaky_relu(h, 0.2)
            h = h.view(batch_size, n_points, -1)
            hs.append(h)

        h = torch.cat(hs, 2)
        h = self.proj(h)
        h_max, _ = torch.max(h, 1)
        h_avg = torch.mean(h, 1)
        h = torch.cat([h_max, h_avg], 1)

        for i in range(self.num_embs):
            h = self.embs[i](h)
            h = self.bn_embs[i](h)
            h = F.leaky_relu(h, 0.2)
            h = self.dropouts[i](h)

        h = self.proj_output(h)
        #return h



        p = F.leaky_relu(self.b1(self.l1(h)))
        p = F.leaky_relu(self.b2(self.l2(p)))
        p = self.l3(p)
        #'''
        n = F.leaky_relu(self.b1(self.l1(h)))
        n = F.leaky_relu(self.b2(self.l2(n)))
        n = self.l3(n)
        #'''
        if self.rep == 'p':
            '''
            for i in range(self.num_encs):
                print(h.shape)
                h = self.encs[i](h)
                print(h.shape)
                h = self.bn_encs[i](h)
                print(h.shape)
                h = F.leaky_relu(h, 0.2)
                #h = self.dropouts[i](h)
            return self.enc_output(h)
            '''
            return p
        elif self.rep == 'n':
            return n
        return p, n
