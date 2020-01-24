from __future__ import print_function
import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

import cloudpickle
from utils import visualize_slices, init_tri_complex_3d
import numpy as np
import matplotlib.pyplot as plt
import dataIO as io
from tqdm import trange
from topologylayer.nn import *
# from topologylayer.functional.utils_dionysus import *

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--input', type=str, default="E:/git/TFRecord_example/input/CT/patch/size9/filename.txt",
                    help='File path of input images')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--beta', type=float, default=0.1, metavar='B',
                    help='beta')
parser.add_argument('--ramda', type=float, default=0, metavar='R',
                    help='ramda')
parser.add_argument('--topo', '-t', type=bool, default=False, help='topo')
parser.add_argument('--constrain', '-c', type=bool, default=False, help='topo con')
parser.add_argument('--mode', type=int, default=1,
                    help='[mode: process] = [0: artificial], [1: real], [2: debug]')
parser.add_argument('--model', type=str, default="",
                    help='File path of loaded model')
parser.add_argument('--latent_dim', type=int, default=24,
                    help='dimension of latent space')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

viz = Visdom()

# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# val_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

image_size = 9

if args.mode==0:
    num_of_data = 10000
    num_of_test = 2000
    num_of_val = 2000
    outdir = "./results/artificial/z_{}/B_{}/R_{}/".format(args.latent_dim, args.beta, args.ramda)
elif args.constrain==True:
    num_of_data = 1978
    num_of_test = 467
    num_of_val = 425
    outdir = "./results/CT/con/z_{}/B_{}/R_{}/".format(args.latent_dim, args.beta, args.ramda)
elif args.mode==1:
    num_of_data = 3039
    num_of_test = 607
    num_of_val = 607
    outdir = "./results/CT/z_{}/B_{}/R_{}/".format(args.latent_dim, args.beta, args.ramda)
else:
    num_of_data = 10000
    num_of_test = 2000
    num_of_val = 2000
    outdir = "./results/debug/".format(args.latent_dim, args.beta, args.ramda)


writer = SummaryWriter(log_dir=outdir+"logs")

if not (os.path.exists(outdir)):
    os.makedirs(outdir)

print('load data')
list = io.load_list(args.input)
data_set = np.zeros((len(list), image_size, image_size, image_size))

for i in trange(len(list)):
    data_set[i, :] = np.reshape(io.read_mhd_and_raw(list[i]), [image_size, image_size, image_size])

data = data_set.reshape(num_of_data, image_size * image_size * image_size)

def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)

data = min_max(data, axis=1)

test_data = torch.from_numpy(data[:num_of_test]).float()
val_data = torch.from_numpy(data[num_of_test:num_of_test+num_of_val]).float().to(device)
train_data = torch.from_numpy(data[num_of_test+num_of_val:]).float().to(device)

train_loader = torch.utils.data.DataLoader(train_data,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=False,
                          drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=False,
                          drop_last=True)

# initialize list for plot graph after training
train_loss_list, val_loss_list = [], []


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(729, 200)
        self.fc21 = nn.Linear(200, latent_dim)
        self.fc22 = nn.Linear(200, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 200)
        self.fc4 = nn.Linear(200, 729)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 729))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

if args.model:
    with open(args.model, 'rb') as f:
        model = cloudpickle.load(f).to(device)
    summary(model, (1,9*9*9))
else:
    model = VAE(args.latent_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    batch_size = x.size(0)
    feature_size = x.size(1)
    assert batch_size != 0

    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 729), reduction='sum')
    MSE = F.mse_loss(recon_x, x, size_average=False).div(batch_size)
    SE = MSE*feature_size
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD *= args.beta

    if args.topo==True:
        topo, b01, b0, b1, b2 = topological_loss(recon_x, x)
        topo *= args.ramda
        total_loss = SE + KLD + topo
        return total_loss, SE, KLD, topo, b01, b0, b1, b2
    else:
        total_loss = SE + KLD
        return total_loss, SE, KLD


def topological_loss(recon_x, x):
    batch_size = x.size(0)
    b01 = 0
    b0 = 0
    b1 = 0
    b2 = 0
    cpx = init_tri_complex_3d(9, 9, 9)
    layer = LevelSetLayer(cpx, maxdim=2, sublevel=False)
    f01 = TopKBarcodeLengths(dim=0, k=1)
    f0 = PartialSumBarcodeLengths(dim=0, skip=1)
    f1 = SumBarcodeLengths(dim=1)
    f2 = SumBarcodeLengths(dim=2)
    for i in range(batch_size):
        dgminfo = layer(recon_x.view(batch_size, 9, 9, 9)[i])
        b01 += ((1 - f01(dgminfo)) ** 2).sum()
        b0 += (f0(dgminfo) ** 2).sum()
        b1 += (f1(dgminfo) ** 2).sum()
        b2 += (f2(dgminfo) ** 2).sum()
    b01 = b01.div(batch_size)
    b0 = b0.div(batch_size)
    b1 = b1.div(batch_size)
    b2 = b2.div(batch_size)
    topo = b01 + b0 + b1 + b2
    return topo, b01, b0, b1, b2


# class TopLoss(nn.Module):
#     def __init__(self, cpx = init_tri_complex_3d(9,9,9)):
#         super(TopLoss, self).__init__()
#         self.pdfn = LevelSetLayer(cpx, maxdim=2,  sublevel=False)
#         self.topfn = SumBarcodeLengths(dim=0)
#         self.topfn2 = SumBarcodeLengths(dim=1)
#         self.topfn3 = SumBarcodeLengths(dim=2)
#
#     def forward(self):
#         dgminfo = self.pdfn()
#         return self.topfn(dgminfo) + self.topfn2(dgminfo) + self.topfn3(dgminfo)


def train(epoch):
    model.train()
    train_loss = 0
    b01 = 0
    b0 = 0
    b1 = 0
    b2 = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        # loss = loss_function(recon_batch, data, mu, logvar)
        # loss, l_SE, l_KLD, l_topo, l01, l0, l1, l2 = loss_function(recon_batch, data, mu, logvar)
        loss,l01,l0,l1,l2 = topological_loss(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        b01 += l01.item()
        b0 += l0.item()
        b1 += l1.item()
        b2 += l2.item()
        # SE += l_SE.item()
        # KLD += l_KLD.item()
        # topo += l_topo.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)
    b01 /= len(train_loader.dataset)
    b0 /= len(train_loader.dataset)
    b1 /= len(train_loader.dataset)
    b2 /= len(train_loader.dataset)
    # if
    # SE /= len(train_loader.dataset)
    # KLD /= len(train_loader.dataset)
    # topo /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss))

    writer.add_scalars("loss/topological_loss", {'topo': train_loss,
                                                 'b01': b01,
                                                 'b0': b0,
                                                 'b1': b1,
                                                 'b2': b2}, epoch)
    #     writer.add_scalars("loss/each_loss", {'Train': train_loss,
#                                              'Rec': SE,
#                                              'KL': KLD,
#                                              'Topo': topo, epoch)


    viz.line(X=np.array([epoch]), Y=np.array([train_loss]), win='topo_loss', name='topo', update='append',
             opts=dict(showlegend=True))
    viz.line(X=np.array([epoch]), Y=np.array([b01]), win='topo_loss', name='b01', update='append')
    viz.line(X=np.array([epoch]), Y=np.array([b0]), win='topo_loss', name='b0', update='append')
    viz.line(X=np.array([epoch]), Y=np.array([b1]), win='topo_loss', name='b1', update='append')
    viz.line(X=np.array([epoch]), Y=np.array([b2]), win='topo_loss', name='b2', update='append')
    # viz.line(X=np.array([epoch]), Y=np.array([train_loss]), win='each_loss', name='train', update='append',
    #          opts=dict(showlegend=True))
    # viz.line(X=np.array([epoch]), Y=np.array([SE]), win='each_loss', name='Rec', update='append')
    # viz.line(X=np.array([epoch]), Y=np.array([KLD]), win='each_loss', name='KL', update='append')
    # viz.line(X=np.array([epoch]), Y=np.array([topo]), win='each_loss', name='Topo', update='append')

    return train_loss


def val(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, _, _, _, _ = topological_loss(recon_batch, data)
            val_loss += loss.item()
            # val_loss += loss_function(recon_batch, data, mu, logvar).item()
            # if i == 0 and epoch>200:
            #     n = min(data.size(0), 11)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 729)[:n]])
            #     ori = np.reshape(data[:n].cpu().numpy(), [n, 9, 9, 9])
            #     rec = np.reshape(recon_batch[:n].cpu().numpy(), [n, 9, 9, 9])
            #     visualize_slices(ori[:,4,:], rec[:,4,:], outdir + 'reconstruction_' + str(epoch))
                # save_image(comparison.cpu(),
                #          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    val_loss /= len(val_loader.dataset)
    val_loss_list.append(val_loss)
    print('====> val set loss: {:.4f}'.format(val_loss))

    return val_loss

if __name__ == "__main__":
    val_loss_min = 100
    for epoch in trange(1, args.epochs + 1):
        train_loss = train(epoch)
        val_loss = val(epoch)
        writer.add_scalars("loss/total_loss", {'train':train_loss,
                                    'val':val_loss}, epoch)
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            # modelの保存
            if epoch > 1:
                path = os.path.join(outdir, 'weight/')
                if not (os.path.exists(path)):
                    os.makedirs(path)
                torch.save(model.state_dict(), path + '{}epoch-{}.pth'.format(epoch, round(val_loss, 2)))
            with open(outdir + 'model.pkl', 'wb') as f:
                cloudpickle.dump(model, f)
        # with torch.no_grad():
        #     sample = torch.randn(32, 24).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(32, 1, 9, 9, 9)[:,:,4,:],
        #                'results/sample_' + str(epoch) + '.png')

        viz.line(X=np.array([epoch]), Y=np.array([train_loss]), win='loss', name='train_loss', update='append', opts=dict(showlegend=True))
        viz.line(X=np.array([epoch]), Y=np.array([val_loss]), win='loss', name='val_loss', update='append')
        with open(outdir + 'train_loss', 'wb') as f:
            cloudpickle.dump(train_loss_list, f)
        with open(outdir + 'val_loss', 'wb') as f:
            cloudpickle.dump(val_loss_list, f)