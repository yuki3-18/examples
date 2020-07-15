'''
# Reconstruct image and evaluate the performance by Generalization
# Author: Yuki Saeki
'''

import os
import argparse
import torch
from torchsummary import summary
import torch.utils.data
import cloudpickle
from tqdm import trange, tqdm
from solver import VAE
import numpy as np
import gudhi as gd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from topologylayer.nn import (PartialSumBarcodeLengths,
                              SumBarcodeLengths, TopKBarcodeLengths)
from utils import *
from topologylayer.nn.levelset import *
from tqdm import trange

parser = argparse.ArgumentParser(description='VAE test')
parser.add_argument('--input', type=str, default='hole',
                    help='File path of input images')
parser.add_argument('--model', type=str, default='',
                    help='Model path')
parser.add_argument('--outdir', type=str, default='E:/git/pytorch/vae/results/artificial/hole/z_3/B_0.1/L_60000/',
                    help='File path of output images')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--constrain', '-c', action='store_true', default=False, help='topo con')
parser.add_argument('--PH', '-ph', action='store_true', default=False, help='PH calculation')
parser.add_argument('--mode', type=int, default=0,
                    help='[mode: process] = [0: artificial], [1: real]')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.mode == 0:
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

# settings
patch_side = 9
patch_center = patch_side//2
n_sample = 2000

torch.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

# set path
in_path = os.path.join('./input', args.input, 'filename.txt')
gen_path = os.path.join(args.outdir, 'gen')
gen_ori_path = os.path.join(gen_path, 'ori')
gen_rec_path = os.path.join(gen_path, 'rec')
gen_topo_path = os.path.join(gen_path, 'topo')
spe_path = os.path.join(args.outdir, 'spe')
spe_ori_path = os.path.join(spe_path, 'ori')
spe_topo_path = os.path.join(spe_path, 'topo')
os.makedirs(gen_ori_path, exist_ok=True)
os.makedirs(gen_rec_path, exist_ok=True)
os.makedirs(spe_ori_path, exist_ok=True)

model_path = os.path.join(args.outdir, 'model.pkl')

# get data
data_set = get_dataset(in_path, patch_side, num_of_data)
data = data_set.reshape(num_of_data, patch_side * patch_side * patch_side)
data = min_max(data, axis=1)
test = data[:num_of_test]

gs_data_set = get_dataset('./input/{}0/filename.txt'.format(args.input), patch_side, num_of_data)
gs_data_set = gs_data_set.reshape(num_of_data, patch_side * patch_side * patch_side)
gs_data = min_max(gs_data_set, axis=1)
gs = gs_data[:num_of_test]

# divide data
test_data = torch.from_numpy(test).float()
gs_data = torch.from_numpy(gs).float()
train_data = torch.from_numpy(data[num_of_test+num_of_val:]).float()
# train_loader = torch.utils.data.DataLoader(train_data,
#                           batch_size=train_data.size(0),
#                           shuffle=False,
#                           num_workers=0,
#                           pin_memory=False,
#                           drop_last=False)
# test_loader = torch.utils.data.DataLoader(test_data,
#                           batch_size=test_data.size(0),
#                           shuffle=False,
#                           num_workers=0,
#                           pin_memory=False,
#                           drop_last=False)

# load model
if args.model:
    model = VAE(latent_dim=6).to(device)
    model.load_state_dict(torch.load(args.model))
else:
    with open(model_path, 'rb') as f:
        model = cloudpickle.load(f)
summary(model, (1, patch_side**3))

def gen(model):
    file_ori = open(os.path.join(gen_ori_path, 'list.txt'), 'w')
    file_rec = open(os.path.join(gen_rec_path, 'list.txt'), 'w')
    bar01, bar0, bar1, bar2 = [], [], [], []
    bar01_o, bar0_o, bar1_o, bar2_o = [], [], [], []
    score = []

    model.eval()
    with torch.no_grad():
        test_data_cuda = test_data.to(device)
        rec_batch, mu, logvar = model(test_data_cuda)
        rec_batch = rec_batch.cpu().numpy()
        test = test_data.numpy()
        generalization = np.double(np.mean(abs(rec_batch - test), axis=1))
        gen_gs = np.double(np.mean(abs(rec_batch - gs), axis=1))

        rec_batch = np.reshape(rec_batch,[num_of_test, patch_side, patch_side, patch_side])
        test = np.reshape(test, [num_of_test, patch_side, patch_side, patch_side])

        for i in trange(num_of_test):
            ori = test[i]
            # ori = np.reshape(test[i], [patch_side, patch_side, patch_side])
            ori_img = sitk.GetImageFromArray(ori)
            ori_img.SetOrigin([0, 0, 0])
            ori_img.SetSpacing([0.885, 0.885, 1])

            rec = rec_batch[i]
            # rec = np.reshape(rec_batch[i], [patch_side, patch_side, patch_side])
            rec_img = sitk.GetImageFromArray(rec)
            rec_img.SetOrigin([0, 0, 0])
            rec_img.SetSpacing([0.885, 0.885, 1])

            # output image
            io.write_mhd_and_raw(ori_img, '{}.mhd'.format(os.path.join(gen_ori_path, '{}'.format(str(i).zfill(4)))))
            io.write_mhd_and_raw(rec_img, '{}.mhd'.format(os.path.join(gen_rec_path, '{}'.format(str(i).zfill(4)))))
            file_ori.write('{}.mhd'.format(os.path.join(gen_ori_path, '{}'.format(str(i).zfill(4)))) + '\n')
            file_rec.write('{}.mhd'.format(os.path.join(gen_rec_path, '{}'.format(str(i).zfill(4)))) + '\n')

            if args.PH == True:
                # calculate PH of original image
                # b01_o, b0_o, b1_o, b2_o = PH(ori)
                # bar01_o.append(b01_o.item())
                # bar0_o.append(b0_o.item())
                # bar1_o.append(b1_o.item())
                # bar2_o.append(b2_o.item())
                # calculate PH of reconstruction
                b01, b0, b1, b2 = PH(rec)
                bar01.append(b01.item())
                bar0.append(b0.item())
                bar1.append(b1.item())
                bar2.append(b2.item())
                # calculate score
                # s = (1.0 - b01.item()**2) + b0.item()**2 + b1.item()**2 + b2.item()**2
                # score.append([s])

    file_ori.close()
    file_rec.close()

    dif = generalization - gen_gs

    i_min = dif.argmin()
    i_max = dif.argmax()

    # save images
    save_img_planes(test[i_min], os.path.join(gen_path, 'min', 'ori_{:.4f}'.format(generalization[i_min])))
    save_img_planes(gs[i_min], os.path.join(gen_path, 'min', 'gs_{:.4f}'.format(gen_gs[i_min])))
    save_img_planes(rec_batch[i_min], os.path.join(gen_path, 'min', 'rec_{}'.format(str(i_min).zfill(4))))

    save_img_planes(test[i_max], os.path.join(gen_path, 'max', 'ori_{:.4f}'.format(generalization[i_max])))
    save_img_planes(gs[i_max], os.path.join(gen_path, 'max', 'gs_{:.4f}'.format(gen_gs[i_max])))
    save_img_planes(rec_batch[i_max], os.path.join(gen_path, 'max', 'rec_{}'.format(str(i_max).zfill(4))))


    # transpose bar
    bar = [bar01, bar0, bar1, bar2]
    bar_o = [bar01_o, bar0_o, bar1_o, bar2_o]
    if args.PH == True:
        bar = np.transpose(bar)
        bar_o = np.transpose(bar_o)
        # score = np.array(score).flatten()
        # save_PH_diag(test[i_min], os.path.join(gen_topo_path, 'min', 'ori{}'.format(str(i_min).zfill(4))))
        # save_PH_diag(rec_batch[i_min], os.path.join(gen_topo_path, 'min', 'rec{}'.format(str(i_min).zfill(4))))
        # save_PH_diag(test[i_max], os.path.join(gen_topo_path, 'max', 'ori{}'.format(str(i_max).zfill(4))))
        # save_PH_diag(rec_batch[i_max], os.path.join(gen_topo_path, 'max', 'rec{}'.format(str(i_max).zfill(4))))

    # plot reconstruction
    a_X = test[:, patch_center, :]
    a_Xe = rec_batch[:, patch_center, :]
    c_X = test[:, :, patch_center, :]
    c_Xe = rec_batch[:, :, patch_center, :]
    s_X = test[:, :, :, patch_center]
    s_Xe = rec_batch[:, :, :, patch_center]
    visualize_slices(a_X, a_Xe, args.outdir + 'gen/axial_')
    visualize_slices(c_X, c_Xe, args.outdir + 'gen/coronal_')
    visualize_slices(s_X, s_Xe, args.outdir + 'gen/sagital_')
    np.savetxt(os.path.join(args.outdir, 'gen_gs_{:.4f}.csv'.format(np.mean(gen_gs))), gen_gs, delimiter=',')

    return generalization, bar, bar_o


def spe(model):
    specificity = []
    bar01, bar0, bar1, bar2 = [], [], [], []
    score = []
    sample = []
    id = []
    spe_gs = []

    model.eval()
    #  calculate mu and sigma
    with torch.no_grad():
        train_data_cuda = train_data.to(device)
        recon_batch, mean, logvar = model(train_data_cuda)
        # mu = mean.mean().item()
        # sigma = (torch.exp(0.5 * logvar)).mean().item()
        mu = torch.mean(mean, 0)
        # std = torch.mean(torch.exp(0.5 * logvar), 0)
        sigma = torch.std(mean, 0)

        print(mu)
        print(sigma)

        file_spe = open(os.path.join(spe_path, 'list.txt'), 'w')
        file_spe_ori = open(os.path.join(spe_ori_path, 'list.txt'), 'w')

        ori = np.reshape(test, [num_of_test, patch_side, patch_side, patch_side])

        for j in trange(n_sample):
            eps = torch.randn_like(sigma)
            sample_z = mu + sigma * eps
            # sample_z = torch.normal(mu, sigma, (1, latent_dim)).to(device)
            sam_batch = model.decode(sample_z)
            sam_single = sam_batch.cpu().numpy()
            # sample.append(sam_single)
            sam = np.reshape(sam_single, [patch_side, patch_side, patch_side])

            # calculate spe
            # case_min_specificity = 1.0

            specificity_tmp = np.double(np.mean(abs(test - sam_single), axis=1))
            spe_gs_tmp = np.double(np.mean(abs(gs - sam_single), axis=1))
            index = np.argmin(specificity_tmp)
            # index_gs = np.argmax(spe_gs_tmp)
            case_min_specificity = specificity_tmp[index]
            cm_spe_gs = spe_gs_tmp[index]

            # for image_index in range(num_of_test):
            #     specificity_tmp = L1norm(ori[image_index] ,sam)
            #     # spe_gs_tmp = L1norm(gs[image_index], sam)
            #     if specificity_tmp < case_min_specificity:
            #         case_min_specificity = specificity_tmp
            #         index = image_index

            id.append([index])
            specificity.append([case_min_specificity])
            spe_gs.append([cm_spe_gs])

            # EUDT
            # ori_img = sitk.GetImageFromArray(ori[index])
            # ori_img.SetSpacing([0.885, 0.885, 1])
            # ori_img.SetOrigin([0, 0, 0])

            sam_img = sitk.GetImageFromArray(sam)
            sam_img.SetSpacing([0.885, 0.885, 1])
            sam_img.SetOrigin([0, 0, 0])

            # output image
            # io.write_mhd_and_raw(ori_img, '{}.mhd'.format(os.path.join(spe_path, 'ori', '{}'.format(str(index).zfill(4)))))
            io.write_mhd_and_raw(sam_img, '{}.mhd'.format(os.path.join(spe_path, '{}'.format(str(j).zfill(4)))))
            file_spe.write('{}.mhd'.format(os.path.join(spe_path, '{}'.format(str(j).zfill(4)))) + '\n')
            file_spe_ori.write('{}.mhd'.format(os.path.join(gen_ori_path, '{}'.format(str(index).zfill(4)))) + '\n')

            if args.PH == True:
                # calculate PH
                b01, b0, b1, b2 = PH(sam)
                bar01.append(b01.item())
                bar0.append(b0.item())
                bar1.append(b1.item())
                bar2.append(b2.item())
                # s = (1.0 - b01.item()**2) + b0.item()**2 + b1.item()**2 + b2.item()**2
                # score.append([s])
    file_spe.close()

    bar = [bar01, bar0, bar1, bar2]
    if args.PH == True:
        bar = np.transpose(bar)
    #     score = np.array(score).flatten()
    #     id = np.array(id).flatten()
    #     i_min = score.argmin()
    #     i_max = score.argmax()
    #
    #     save_PH_diag(ori[id[i_min]], os.path.join(spe_topo_path, 'min', 'ori{}'.format(str(id[i_min][0]).zfill(4))))
    #     save_PH_diag(sample[i_min], os.path.join(spe_topo_path, 'min', 'sam{}'.format(str(i_min).zfill(4))))
    #     save_PH_diag(ori[id[i_max]], os.path.join(spe_topo_path, 'max', 'ori{}'.format(str(id[i_max][0]).zfill(4))))
    #     save_PH_diag(sample[i_max], os.path.join(spe_topo_path, 'max', 'sam{}'.format(str(i_max).zfill(4))))

    np.savetxt(os.path.join(args.outdir, 'spe_gs_{:.4f}.csv'.format(np.mean(spe_gs))), spe_gs, delimiter=',')

    return specificity, bar

def PH(data):
    z, y, x = data.shape
    # data = np.ndarray(data, [z, y, x])
    cpx = init_tri_complex_3d(z, y, x)
    layer = LevelSetLayer(cpx, maxdim=2, sublevel=False)
    dgminfo = layer(torch.from_numpy(data).float())
    f01 = TopKBarcodeLengths(dim=0, k=1)
    f0 = PartialSumBarcodeLengths(dim=0, skip=1)
    f1 = SumBarcodeLengths(dim=1)
    f2 = SumBarcodeLengths(dim=2)
    b01 = f01(dgminfo).sum()
    b0 = f0(dgminfo)
    b1 = f1(dgminfo)
    b2 = f2(dgminfo)
    return b01, b0, b1, b2

if __name__ == '__main__':

    # generalization
    print('-' * 20, 'Computing Generalization', '-' * 20)
    generalization, bar, bar_o = gen(model)
    gen_mean = np.mean(generalization)
    print('generalization = %f' % gen_mean)
    np.savetxt(os.path.join(args.outdir, 'generalization_{:.4f}.csv'.format(gen_mean)), generalization, delimiter=',')
    if args.PH == True:
        np.savetxt(os.path.join(args.outdir, 'gen_topo.csv'), bar, delimiter=',')
        # np.savetxt(os.path.join(args.outdir, 'ori_topo.csv'), bar_o, delimiter=',')

    # specificity
    print('-' * 20, 'Computing Specificity', '-' * 20)
    specificity, bar = spe(model)
    spe_mean = np.mean(specificity)
    print('specificity = %f' % np.mean(spe_mean))
    np.savetxt(os.path.join(args.outdir, 'specificity_{:.4f}.csv'.format(spe_mean)), specificity, delimiter=',')
    if args.PH == True:
        np.savetxt(os.path.join(args.outdir, 'spe_topo.csv'), bar, delimiter=',')

    print('-' * 20, 'Finish!', '-' * 20)

