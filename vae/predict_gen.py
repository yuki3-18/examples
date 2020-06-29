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
parser.add_argument('--input', type=str, default="E:/git/pytorch/vae/input/hole/filename.txt",
                    help='File path of input images')
parser.add_argument('--model', type=str, default="E:/git/pytorch/vae/results/artificial/hole/z_6/B_0.1/L_30000/weight/246epoch-42.21.pth",
                    help='File path of model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--constrain', '-c', type=bool, default=False, help='topo con')
parser.add_argument('--outdir', type=str, default="E:/git/pytorch/vae/results/artificial/hole/z_6/B_0.1/L_30000/gen/",
                    help='File path of output images')
parser.add_argument('--mode', type=int, default=0,
                    help='[mode: process] = [0: artificial], [1: real]')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

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

patch_side = 9

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

# check folder
if not (os.path.exists(args.outdir + 'rec/')):
    os.makedirs(args.outdir + 'ori/')
    os.makedirs(args.outdir + 'rec/')

print('load data')
list = io.load_list(args.input)
data_set = np.zeros((num_of_test, patch_side, patch_side, patch_side))

for i in trange(num_of_test):
    data_set[i, :] = np.reshape(io.read_mhd_and_raw(list[i]), [patch_side, patch_side, patch_side])

data = data_set.reshape(num_of_test, patch_side * patch_side * patch_side)
data = min_max(data, axis=1)

test_data = torch.from_numpy(data).float()
test_loader = torch.utils.data.DataLoader(test_data,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=False,
                          drop_last=False)

# with open(args.model, 'rb') as f:
#     model = cloudpickle.load(f)

def gen(model):
    with torch.no_grad():
        ori=[]
        rec=[]
        for i, data in enumerate(test_loader):
            # print(i)
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            ori_single = data[0, :]
            rec_single = recon_batch[0, :]
            ori.append(ori_single.cpu().numpy())
            rec.append(rec_single.cpu().numpy())

    ori = np.reshape(ori, [num_of_test, patch_side, patch_side, patch_side])
    rec = np.reshape(rec, [num_of_test, patch_side, patch_side, patch_side])

    generalization_single = []
    file_ori = open(args.outdir + 'ori/list.txt', 'w')
    file_rec = open(args.outdir + 'rec/list.txt', 'w')

    for j in trange(len(rec)):

        # EUDT
        ori_image = sitk.GetImageFromArray(ori[j])
        ori_image.SetOrigin([0, 0, 0])
        ori_image.SetSpacing([0.885,0.885,1])

        rec_image = sitk.GetImageFromArray(rec[j])
        rec_image.SetOrigin([0, 0, 0])
        rec_image.SetSpacing([0.885,0.885,1])

        # output image
        io.write_mhd_and_raw(ori_image, '{}.mhd'.format(os.path.join(args.outdir, 'ori','{}'.format(str(j).zfill(4)))))
        io.write_mhd_and_raw(rec_image, '{}.mhd'.format(os.path.join(args.outdir, 'rec', '{}'.format(str(j).zfill(4)))))
        file_ori.write('{}.mhd'.format(os.path.join(args.outdir, 'ori', '{}'.format(str(j).zfill(4)))) + "\n")
        file_rec.write('{}.mhd'.format(os.path.join(args.outdir, 'rec', '{}'.format(str(j).zfill(4)))) + "\n")

        generalization_single.append(L1norm(ori[j], rec[j]))

    file_ori.close()
    file_rec.close()

    generalization = np.average(generalization_single)
    print('generalization = %f' % generalization)

    np.savetxt(os.path.join(args.outdir, 'generalization.csv'), generalization_single, delimiter=",")

    # plot reconstruction
    a_X = ori[:, 4, :]
    a_Xe = rec[:, 4, :]
    c_X = ori[:, :, 4, :]
    c_Xe = rec[:, :, 4, :]
    s_X = ori[:, :, :, 4]
    s_Xe = rec[:, :, :, 4]
    visualize_slices(a_X, a_Xe, args.outdir + "axial_")
    visualize_slices(c_X, c_Xe, args.outdir + "coronal_")
    visualize_slices(s_X, s_Xe, args.outdir + "sagital_")

if __name__ == '__main__':
    model = VAE(latent_dim=6).to(device)
    model.eval()
    model.load_state_dict(torch.load(args.model))
    gen(model)