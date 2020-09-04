from __future__ import print_function
import json
import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchsummary import summary
# from torch.utils.tensorboard import SummaryWriter
import tensorboardX as tbx
from visdom import Visdom
from solver import SquaredBarcodeLengths, PartialSquaredBarcodeLengths

import cloudpickle
import numpy as np
import matplotlib.pyplot as plt
import dataIO as io
from tqdm import trange
from utils import init_tri_complex_3d, plt_loss
from topologylayer.nn import *
# from topologylayer.functional.utils_dionysus import *

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--input', type=str, default="hole/",
                    help='File path of input images')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--beta', type=float, default=0.1, metavar='B',
                    help='beta')
parser.add_argument('--L1', '-l1', action='store_true', default=False, help='use L1 norm as rec loss')
parser.add_argument('--lam', type=float, default=1, metavar='L',
                    help='lambda')
parser.add_argument('--alpha', '-alp', type=float, default=1, metavar='A',
                    help='gamma')
parser.add_argument('--gamma', '-gam', type=float, default=1, metavar='C',
                    help='gamma')
parser.add_argument('--topo', '-t', action='store_true', default=False, help='topo')
parser.add_argument('--constrain', '-c', action='store_true', default=False, help='topo con')
parser.add_argument('--mode', type=int, default=0,
                    help='[mode: process] = [0: artificial], [1: real], [2: only topological loss]')
parser.add_argument('--model', type=str, default="",
                    help='File path of loaded model')
parser.add_argument('--latent_dim', type=int, default=6,
                    help='dimension of latent space')
parser.add_argument('--do', type=int, default=0,
                    help='drop out ratio')
parser.add_argument('--patient', type=int, default=100,
                    help='epochs for early stopping')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

viz = Visdom()

patch_side = 9

data_path = os.path.join("./input/", args.input, "filename.txt")
width_number = len(str(args.lam))

if args.mode==0:
    num_of_data = 10000
    num_of_test = 2000
    num_of_val = 2000
    if args.L1==True:
        outdir = os.path.join("./results/artificial/", args.input, "l1",
                              "z_{}/B_{:g}/batch{}/L_{:g}/C_{:g}/".format(args.latent_dim, args.beta, args.batch_size, args.lam, args.gamma))
    else:
        outdir = os.path.join("./results/artificial/", args.input, "z_{}/B_{:g}/batch{}/L_{:g}/C_{:g}/A_{:g}/".format(args.latent_dim, args.beta, args.batch_size, args.lam, args.gamma, args.alpha))
elif args.constrain==True:
    num_of_data = 1978
    num_of_test = 467
    num_of_val = 425
    outdir = "./results/CT/con/z_{}/B_{:g}/L_{:g}/".format(args.latent_dim, args.beta, args.lam)
elif args.mode==1:
    num_of_data = 3039
    num_of_test = 607
    num_of_val = 607
    outdir = "./results/CT/z_{}/B_{:g}/L_{:g}/".format(args.latent_dim, args.beta, args.lam)
else:
    num_of_data = 10000
    num_of_test = 2000
    num_of_val = 2000
    outdir = os.path.join("./results/artificial/", args.input, "z_{}/topo/L_{:g}/".format(args.latent_dim, args.beta, args.lam))

num_of_train = num_of_data - num_of_val - num_of_test

if not (os.path.exists(outdir)):
    os.makedirs(outdir)

writer = tbx.SummaryWriter(log_dir=outdir+"logs")

# save parameters
with open(os.path.join(outdir, "params.json"), mode="w") as f:
    json.dump(args.__dict__, f, indent=4)

print('-'*20, 'Data loading', '-'*20)
list = io.load_list(data_path)
data_set = np.zeros((len(list), patch_side, patch_side, patch_side))

for i in trange(len(list)):
    data_set[i, :] = np.reshape(io.read_mhd_and_raw(list[i]), [patch_side, patch_side, patch_side])

data = data_set.reshape(num_of_data, patch_side **3)

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
    def __init__(self, latent_dim, do=0):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(patch_side**3, 200)
        self.fc21 = nn.Linear(200, latent_dim)
        self.fc22 = nn.Linear(200, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 200)
        self.fc4 = nn.Linear(200, patch_side**3)
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
        mu, logvar = self.encode(x.view(-1, patch_side**3))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

if args.model:
    root, ext = os.path.splitext(args.model)
    if ext=='.pkl':
        with open(args.model, 'rb') as f:
            model = cloudpickle.load(f).to(device)
    else:
        model = VAE(args.latent_dim, args.do).to(device)
        model.load_state_dict(torch.load(args.model))
    summary(model, (1, patch_side**3))
else:
    model = VAE(args.latent_dim, args.do).to(device)

writer.add_graph(model, train_data[0].view(1, patch_side**3))

optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    batch_size = x.size(0)
    feature_size = x.size(1)
    assert batch_size != 0

    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, patch_side**3), reduction='sum')
    if args.L1==True:
        MAE = F.l1_loss(recon_x, x, size_average=False).div(batch_size)
        REC = MAE * feature_size
    else:
        MSE = F.mse_loss(recon_x, x, size_average=False).div(batch_size)
        REC = MSE * feature_size
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD *= args.beta

    if args.topo==True:
        topo, t01, t0, t1, t2 = topological_loss(recon_x, x)
        topo, t01, t0, t1, t2 = topo*args.lam, t01*args.lam, t0*args.lam, t1*args.lam, t2*args.lam
        total_loss = REC + KLD + topo
        return total_loss, REC, KLD, topo, t01, t0, t1, t2
    else:
        total_loss = REC + KLD
        return total_loss, REC, KLD


def topological_loss(recon_x, x):
    batch_size = x.size(0)
    # feature_size = x.size(1)
    t01, t0, t1, t2 = 0., 0., 0., 0.
    # bt01, bt0, bt11, bt1, bt2 = 0., 0., 0., 0., 0.
    cpx = init_tri_complex_3d(patch_side, patch_side, patch_side)
    layer = LevelSetLayer(cpx, maxdim=2, sublevel=False)
    f01 = TopKBarcodeLengths(dim=0, k=1)
    f0 = PartialSquaredBarcodeLengths(dim=0, skip=1)
    f1 = SquaredBarcodeLengths(dim=1)
    f2 = SquaredBarcodeLengths(dim=2)

    for i in range(batch_size):
        dgminfo = layer(recon_x.view(batch_size, patch_side, patch_side, patch_side)[i])
        t01 += ((1 - f01(dgminfo) ** 2)).sum()
        t0 += f0(dgminfo).sum()
        t1 += f1(dgminfo).sum()
        t2 += f2(dgminfo).sum()
        # if args.bkg==True:
        #     bkginfo = layer_b((1-recon_x).view(batch_size, patch_side, patch_side, patch_side)[i])
        #     bt01 += ((1 - ft0(bkginfo)[0] ** 2)).sum()
        #     bt0 += (ft0(bkginfo)[1:] ** 2).sum()
        #     bt11 += ((1 - ft1(bkginfo)[0] ** 2)).sum()
        #     bt1 += (ft1(bkginfo)[1:] ** 2).sum()
        #     bt2 += (ft2(bkginfo) ** 2).sum()

    t01 = t01.div(batch_size)
    t0 = t0.div(batch_size)
    t1 = t1.div(batch_size) * args.alpha
    t2 = t2.div(batch_size) * args.gamma
    topo = t01 + t0 + t1 + t2

    # if args.bkg==True:
    #     bt01 = bt01.div(batch_size)
    #     bt0 = bt0.div(batch_size)
    #     bt11 = bt11.div(batch_size)
    #     bt1 = bt1.div(batch_size)
    #     bt2 = bt2.div(batch_size)
    #
    #     bb = bt01 + bt0 + bt11 + bt1 + bt2
    #     topo += bb
    #
    # if args.bkg == True:
    #     return topo, t01, t0, t1, t2, bt01, bt0, bt11, bt1, bt2
    # else:
    return topo, t01, t0, t1, t2

def train(epoch):
    model.train()
    train_loss = 0
    SE, KLD = 0., 0.
    topo = 0.
    t01, t0, t1, t2 = 0., 0., 0., 0.
    bt01, bt0, bt11, bt1, bt2 = 0., 0., 0., 0., 0.
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        # loss = loss_function(recon_batch, data, mu, logvar)
        if args.mode==2:
            # if args.bkg == True:
            #     loss, l01, l0, l1, l2, lt01, lt0, lt11, lt1, lt2 = topological_loss(recon_batch, data)
            # else:
            loss, l01, l0, l1, l2 = topological_loss(recon_batch, data)
            train_loss += loss.item()
            t01 += l01.item()
            t0 += l0.item()
            t1 += l1.item()
            t2 += l2.item()
            # if args.bkg == True:
            #     bt01 += lt01.item()
            #     bt0 += lt0.item()
            #     bt11 += lt11.item()
            #     bt1 += lt1.item()
            #     bt2 += lt2.item()

        elif args.topo==True:
            # if args.bkg == True:
            #    loss, l_SE, l_KLD, l_topo, l01, l0, l1, l2, lt01, lt0, lt11, lt1, lt2 = loss_function(recon_batch, data, mu, logvar)
            # else:
            loss, l_SE, l_KLD, l_topo, l01, l0, l1, l2 = loss_function(recon_batch, data, mu, logvar)
            train_loss += loss.item()
            SE += l_SE.item()
            KLD += l_KLD.item()
            topo += l_topo.item()
            t01 += l01.item()
            t0 += l0.item()
            t1 += l1.item()
            t2 += l2.item()
            # if args.bkg == True:
            #     bt01 += lt01.item()
            #     bt0 += lt0.item()
            #     bt1 += lt1.item()
            #     bt1 += lt1.item()
            #     bt2 += lt2.item()

        else:
            loss, l_SE, l_KLD = loss_function(recon_batch, data, mu, logvar)
            train_loss += loss.item()
            SE += l_SE.item()
            KLD += l_KLD.item()

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    writer.add_images('images/train',
                      recon_batch.view(args.batch_size, 1, patch_side, patch_side, patch_side)[:, :, 4, :], epoch)

    train_loss /= len(train_loader.dataset)
    SE /= len(train_loader.dataset)
    KLD /= len(train_loader.dataset)

    train_loss_list.append(train_loss)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss))

    writer.add_scalars("loss/vae_loss", {'Train': train_loss,
                                             'Rec': SE,
                                             'KL': KLD}, epoch)

    viz.line(X=np.array([epoch]), Y=np.array([train_loss]), win='each_loss', name='train', update='append',
             opts=dict(showlegend=True))
    viz.line(X=np.array([epoch]), Y=np.array([SE]), win='each_loss', name='Rec', update='append')
    viz.line(X=np.array([epoch]), Y=np.array([KLD]), win='each_loss', name='KL', update='append')

    if args.topo==True:
        t01 /= len(train_loader.dataset)
        t0 /= len(train_loader.dataset)
        t1 /= len(train_loader.dataset)
        t2 /= len(train_loader.dataset)
        topo /= len(train_loader.dataset)
        # if args.bkg == True:
        #     bt01 /= len(train_loader.dataset)
        #     bt0 /= len(train_loader.dataset)
        #     bt11 /= len(train_loader.dataset)
        #     bt1 /= len(train_loader.dataset)
        #     bt2 /= len(train_loader.dataset)
        #     writer.add_scalars("loss/topological_loss", {'Topo': topo,
        #                                                  't01': t01,
        #                                                  't0': t0,
        #                                                  't1': t1,
        #                                                  't2': t2,
        #                                                  'bt01': bt01,
        #                                                  'bt0': bt0,
        #                                                  'bt11': bt11,
        #                                                  'bt1': bt1,
        #                                                  'bt2': bt2}, epoch)
        # else:
        writer.add_scalars("loss/topological_loss", {'Topo': topo,
                                                     't01': t01,
                                                     't0': t0,
                                                     't1': t1,
                                                     't2': t2}, epoch)

        writer.add_scalars("loss/each_loss", {'Train': train_loss,
                                              'Rec': SE,
                                              'KL': KLD,
                                              'Topo': topo}, epoch)

        viz.line(X=np.array([epoch]), Y=np.array([topo]), win='each_loss', name='Topo', update='append')
        viz.line(X=np.array([epoch]), Y=np.array([topo]), win='topo_loss', name='topo', update='append',
                 opts=dict(showlegend=True))
        viz.line(X=np.array([epoch]), Y=np.array([t01]), win='topo_loss', name='t01', update='append')
        viz.line(X=np.array([epoch]), Y=np.array([t0]), win='topo_loss', name='t0', update='append')
        viz.line(X=np.array([epoch]), Y=np.array([t1]), win='topo_loss', name='t1', update='append')
        viz.line(X=np.array([epoch]), Y=np.array([t2]), win='topo_loss', name='t2', update='append')
        # if args.bkg == True:
        #     viz.line(X=np.array([epoch]), Y=np.array([bt01]), win='topo_loss', name='bt01', update='append')
        #     viz.line(X=np.array([epoch]), Y=np.array([bt0]), win='topo_loss', name='bt0', update='append')
        #     viz.line(X=np.array([epoch]), Y=np.array([bt11]), win='topo_loss', name='bt11', update='append')
        #     viz.line(X=np.array([epoch]), Y=np.array([bt1]), win='topo_loss', name='bt1', update='append')
        #     viz.line(X=np.array([epoch]), Y=np.array([bt2]), win='topo_loss', name='bt2', update='append')

    return train_loss


def val(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            val_data = val_data.to(device)
            recon_batch, mu, logvar = model(val_data)
            if args.mode == 2:
                loss, _, _, _, _, _, _ = topological_loss(recon_batch, val_data)
            elif args.topo == True:
                # if args.bkg == True:
                #     loss, l_SE, l_KLD, l_topo, l01, l0, l1, l2, lt01, lt0, lt11, lt1, lt2 = loss_function(recon_batch, val_data, mu, logvar)
                # else:
                loss, l_SE, l_KLD, l_topo, l01, l0, l1, l2 = loss_function(recon_batch, val_data, mu,
                                                                                         logvar)
            else:
                loss, l_SE, l_KLD = loss_function(recon_batch, val_data, mu, logvar)
            val_loss += loss.item()

        writer.add_images('images/val', recon_batch.view(args.batch_size, 1, patch_side, patch_side, patch_side)[:,:,4,:], epoch)
            # val_loss += loss_function(recon_batch, data, mu, logvar).item()

    val_loss /= len(val_loader.dataset)
    val_loss_list.append(val_loss)
    print('====> val set loss: {:.4f}'.format(val_loss))

    return val_loss

if __name__ == "__main__":
    val_loss_min = 10000
    epochs_no_improve = 0
    min_delta = 0.001
    n_epochs_stop = args.patient
    for epoch in trange(1, args.epochs + 1):
        train_loss = train(epoch)
        val_loss = val(epoch)
        writer.add_scalars("loss/total_loss", {'train':train_loss,
                                    'val':val_loss}, epoch)
        viz.line(X=np.array([epoch]), Y=np.array([train_loss]), win='loss', name='train_loss', update='append', opts=dict(showlegend=True))
        viz.line(X=np.array([epoch]), Y=np.array([val_loss]), win='loss', name='val_loss', update='append')
        with open(outdir + 'train_loss', 'wb') as f:
            cloudpickle.dump(train_loss_list, f)
        with open(outdir + 'val_loss', 'wb') as f:
            cloudpickle.dump(val_loss_list, f)

        if val_loss < val_loss_min - min_delta:
            epochs_no_improve = 0
            val_loss_min = val_loss
            # save model
            if epoch > args.log_interval:
                path = os.path.join(outdir, 'weight/')
                if not (os.path.exists(path)):
                    os.makedirs(path)
                torch.save(model.state_dict(), path + '{}epoch-{}.pth'.format(epoch, round(val_loss, 2)))
            with open(outdir + 'model.pkl', 'wb') as f:
                cloudpickle.dump(model, f)
        else:
            epochs_no_improve += 1

        # Check early stopping condition
        if epochs_no_improve >= n_epochs_stop:
            print('-'*20, 'Early stopping!', '-'*20)
            plt_loss(epoch, train_loss_list, val_loss_list, outdir)
            break
        plt_loss(epoch, train_loss_list, val_loss_list, outdir)
    writer.close()
