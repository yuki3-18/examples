from topologylayer.nn.features import *
import torch
import torch.nn as nn
from torch.nn import functional as F

class SquaredBarcodeLengths(nn.Module):
    """
    Layer that sums up lengths of barcode in persistence diagram
    ignores infinite bars, and padding
    Options:
        dim - bardocde dimension to sum over (default 0)

    forward input:
        (dgms, issub) tuple, passed from diagram layer
    """
    def __init__(self, dim=0):
        super(SquaredBarcodeLengths, self).__init__()
        self.dim = dim

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        lengths = get_barcode_lengths(dgms[self.dim], issublevel)

        # return Squared norm of the barcode lengths
        return torch.sum(lengths**2, dim=0)


class PartialSquaredBarcodeLengths(nn.Module):
    """
    Layer that computes a partial Squared lengths of barcode lengths

    inputs:
        dim - homology dimension
        skip - skip this number of the longest bars

    ignores infinite bars and padding
    """
    def __init__(self, dim, skip):
        super(PartialSquaredBarcodeLengths, self).__init__()
        self.skip = skip
        self.dim = dim

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        lengths = get_barcode_lengths(dgms[self.dim], issublevel)

        # sort lengths
        sortl, indl = torch.sort(lengths, descending=True)

        return torch.sum(sortl[self.skip:]**2)

class VAE(nn.Module):
    def __init__(self, latent_dim, do=0):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(9**3, 200)
        self.fc21 = nn.Linear(200, latent_dim)
        self.fc22 = nn.Linear(200, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 200)
        self.fc4 = nn.Linear(200, 9**3)
        self.dropout = torch.nn.Dropout(p=do)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout(h1)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = self.dropout(h3)
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 9**3))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar