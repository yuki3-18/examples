'''
# Reconstruct image and evalute the performance by Generalization
# Author: Yuki Saeki
'''

import os
import argparse
import numpy as np
import SimpleITK as sitk
import csv
import dataIO as io
import matplotlib.pyplot as plt
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import cloudpickle
from utils import *
from tqdm import trange
import pickle
from importlib import machinery
from main import VAE

parser = argparse.ArgumentParser(description='VAE test')
parser.add_argument('--input', type=str, default="E:/git/pytorch/vae/input/tip/filename.txt",
                    help='File path of input images')
parser.add_argument('--model', type=str, default="E:/git/pytorch/vae/results/artificial/tip/z_3/B_0/L_0/model.pkl",
                    help='File path of model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--constrain', '-c', type=bool, default=False, help='topo con')
parser.add_argument('--outdir', type=str, default="E:/git/pytorch/vae/results/artificial/tip/z_3/B_0/L_0/",
                    help='File path of output images')
parser.add_argument('--mode', type=int, default=0,
                    help='[mode: process] = [0: artificial], [1: real]')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

patch_side = 9
latent_dim = 6

if args.mode==0:
    num_of_data = 10000
    num_of_test = 2000
    num_of_val = 2000
elif args.constrain == True:
    num_of_data = 1978
    num_of_test = 467
    num_of_val = 425
else:
    num_of_data = 3039
    num_of_test = 607
    num_of_val = 607

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

# check folder
if not (os.path.exists(args.outdir + 'spe/')):
    os.makedirs(args.outdir + 'spe/')

# get data
data_set = get_dataset(args.input, patch_side, num_of_data)
data = data_set.reshape(num_of_data, patch_side * patch_side * patch_side)
data = min_max(data, axis=1)
# divide data
test_data = torch.from_numpy(data[:num_of_test]).float()
train_data = torch.from_numpy(data[num_of_test+num_of_val:]).float().to(device)
train_loader = torch.utils.data.DataLoader(train_data,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=False,
                          drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=False,
                          drop_last=False)
# load model
with open(args.model, 'rb') as f:
    model = cloudpickle.load(f)

# testing
specificity = []
mean = []
std = []
mean_single = []
std_single = []
num_of_gen = 2000
patch_side = 9


def spe(model):
    #  calculate mu and sigma
    for i in enumerate(train_loader):
        with torch.no_grad():
            train_data_cuda = train_data.to(device)
            recon_batch, mean_batch, logvar_batch = model(train_data_cuda)
            mean_single = mean_batch[0, :]
            std_single = torch.exp(0.5 * logvar_batch)[0, :]
            mean.append(mean_single.cpu().numpy())
            std.append(std_single.cpu().numpy())

    mu = np.mean(mean)
    sigma = np.mean(std)

    model.eval()
    with torch.no_grad():
        gen = np.zeros(num_of_gen, dtype="float")
        ori = np.reshape(test_data.cpu().numpy(), [num_of_test, patch_side, patch_side, patch_side])
        for j in trange(num_of_gen):
            # sample_z = np.random.normal(mu, sigma, (1, latent_dim))
            sample_z = torch.normal(mu, sigma, (1, latent_dim)).to(device)
            gen_batch = model.decode(sample_z)
            gen_single = gen_batch.cpu().numpy()
            gen = np.reshape(gen_single, [patch_side, patch_side, patch_side])
            # EUDT
            eudt_image = sitk.GetImageFromArray(gen)
            eudt_image.SetSpacing([0.885, 0.885, 1])
            eudt_image.SetOrigin([0, 0, 0])

            # calculate spe
            case_min_specificity = 1.0
            for image_index in range(num_of_test):
                specificity_tmp = L1norm(ori[image_index] ,gen)
                if specificity_tmp < case_min_specificity:
                    case_min_specificity = specificity_tmp
            specificity.append([case_min_specificity])

            # output image
            io.write_mhd_and_raw(eudt_image, '{}.mhd'.format(os.path.join(args.outdir, 'spe', '{}'.format(str(j).zfill(4)))))

    print('specificity = %f' % np.mean(specificity))
    np.savetxt(os.path.join(args.outdir, 'specificity.csv'), specificity, delimiter=",")

if __name__ == '__main__':
    model = VAE(latent_dim=latent_dim).to(device)
    model.eval()
    model.load_state_dict(torch.load(args.model))
    spe(model)