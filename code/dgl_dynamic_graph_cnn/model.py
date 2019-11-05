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

        self.rep = rep
        #pose_dims = [16, 8, 3]

        #self.pose = nn.ModuleList()
        #self.bn_pose = nn.ModuleList()
        #self.pose_dropouts = nn.ModuleList()

        #self.num_pose = len(pose_dims) - 1
        #for i in range(1, self.num_pose + 1):
            #self.pose.append(nn.Linear(pose_dims[i - 1])
            #self.bn_pose.append(nn.BatchNorm1d(pose_dims[i]))
            #self.pose_dropouts.append(nn.Dropout(dropout_prob))

        self.l1 = nn.Linear(output_dims, 16)
        self.l2 = nn.Linear(16, 8)
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

        #pose_dims = [16, 8, 3]
        #for i in pose_dims:

        p = F.relu(self.l1(h))
        p = F.relu(self.l2(p))
        p = F.relu(self.l3(p))

        n = F.relu(self.l1(h))
        n = F.relu(self.l2(n))
        n = F.relu(self.l3(n))
        if self.rep == 'p':
            return p
        elif self.rep == 'n':
            return n
        return p, n
