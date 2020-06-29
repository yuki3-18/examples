from topologylayer.nn.features import *
import torch
import torch.nn as nn

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