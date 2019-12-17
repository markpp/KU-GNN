import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from chamfer_loss import ChamferLoss

NUM_PTS = 1024
LATENT_SIZE = 32

class cvae(nn.Module):
    def __init__(self):
        super(cvae, self).__init__()

        # Encoder (PointNet)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Encoder
        self.el1 = nn.Linear(1024, 1024)
        self.el2 = nn.Linear(1024, 512)
        self.mu = nn.Linear(512, LATENT_SIZE)
        self.logvar = nn.Linear(512, LATENT_SIZE)

        # Decoder
        self.l4 = nn.Linear(LATENT_SIZE, 512)
        self.l5 = nn.Linear(512, 1024)
        self.l6 = nn.Linear(1024, NUM_PTS * 3)

    # Encoder based on PointNet
    def encoder(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.el1(x))
        x = F.relu(self.el2(x))
        mean = self.mu(x)
        logvar = self.logvar(x)
        return x, mean, logvar

    # Re-paramization trick
    def sample_z(self, x, mean, logvar):
        epsilon = torch.randn((x.shape[0],LATENT_SIZE)).cuda()
        z = epsilon.mul((logvar/2.0).exp()).add(mean)
        return z

    # Decoder
    def decoder(self, z):
        y = F.relu(self.l4(z))
        y = F.relu(self.l5(y))
        y = self.l6(y)
        y = y.reshape(y.shape[0], 3, NUM_PTS) # Reshape into point cloud
        return y

    def forward(self, x):
        x, mean, logvar = self.encoder(x)
        z = self.sample_z(x, mean, logvar)
        y = self.decoder(z)

        return y, mean, logvar, z


class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.chamfer = ChamferLoss()
        self.beta = 0.1 # Weight of KL term

    def forward(self, recon_x, x, mu, logvar):
        # Normalized reconstruction error using Chamfer distance
        RE = self.chamfer(recon_x, x)
        RE /= NUM_PTS

        # Normalized KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2)-logvar.exp())
        KLD /= LATENT_SIZE
        return RE + KLD*self.beta
