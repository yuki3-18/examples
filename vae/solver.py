from topologylayer.nn.features import *
from topologylayer.nn.levelset import LevelSetLayer
from topologylayer.util.construction import unique_simplices
from scipy.spatial import Delaunay
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


class TopLoss(nn.Module):
    def __init__(self, size, betti):
        super(TopLoss, self).__init__()
        self.size = size
        self.b0 = betti[0]
        self.b1 = betti[1]
        self.b2 = betti[2]
        self.cpx = self.init_tri_complex_3d(self.size)
        self.pdfn = LevelSetLayer(self.cpx, maxdim=2, sublevel=False)
        self.topfn = PartialSumBarcodeLengths(dim=1, skip=1) # penalize more than 1 hole
        self.topfn2 = SumBarcodeLengths(dim=0) # penalize more than 1 max

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo) + self.topfn2(dgminfo)

    def init_tri_complex_3d(self, size):
        """
        initialize 3d complex in dumbest possible way
        """
        # initialize complex to use for persistence calculations
        width, height, depth = size
        axis_x = np.arange(0, width)
        axis_y = np.arange(0, height)
        axis_z = np.arange(0, depth)
        grid_axes = np.array(np.meshgrid(axis_x, axis_y, axis_z))
        grid_axes = np.transpose(grid_axes, (1, 2, 3, 0))
        # creation of a complex for calculations
        tri = Delaunay(grid_axes.reshape([-1, 3]))
        return unique_simplices(tri.simplices, 3)


