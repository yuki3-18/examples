import os
import dataIO as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
import torch
import torch.nn as nn
import gudhi as gd
from topologylayer.nn.features import get_start_end
from topologylayer.nn import (PartialSumBarcodeLengths,
                              SumBarcodeLengths, TopKBarcodeLengths)
from topologylayer.nn.levelset import LevelSetLayer
from topologylayer.functional.persistence import SimplicialComplex
from topologylayer.util.construction import unique_simplices
from scipy.spatial import Delaunay
from tqdm import trange


# calculate jaccard
def jaccard(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.bitwise_and(im1, im2).sum()) / np.double(np.bitwise_or(im1, im2).sum())

# calculate L1
def L1norm(im1, im2):
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.mean(abs(im1 - im2)))

# calculate L2
def L2norm(im1, im2):
    im1 = np.asarray(im1)
    im2 = np.asarray(im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.double(np.mean((im1 - im2)**2))

def matplotlib_plt(X, filename):
    fig = plt.figure()
    plt.title('latent distribution')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('dim_1')
    ax.set_ylabel('dim_2')
    ax.set_zlabel('dim_3')
    ax.scatter(X[:,0], X[:,1], X[:,2] , marker="x"
               # , c=y/len(set(y))
    )
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.savefig(os.path.join(filename, "{:03d}.png".format(angle)))
    # plt.savefig(filename)
    # plt.show()

def visualize_slices(X, Xe, outdir):
    # plot reconstruction
    fig, axes = plt.subplots(ncols=10, nrows=2, figsize=(18, 4))
    for i in range(10):
        # minX = np.min(X[i, :])
        # maxX = np.max(X[i, :])
        axes[0, i].imshow(X[i, :].reshape(9, 9), cmap=cm.Greys_r, vmin=0, vmax=1,
                          interpolation='none')
        axes[0, i].set_title('ori %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        # minXe = np.min(Xe[i, :])
        # maxXe = np.max(Xe[i, :])
        axes[1, i].imshow(Xe[i, :].reshape(9, 9), cmap=cm.Greys_r, vmin=0, vmax=1,
                          interpolation='none')
        axes[1, i].set_title('rec %d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)
    plt.savefig(outdir + "reconstruction.png")
    # plt.show()

def display_slices(case):
    # case: image data (num_data, size, size, size)
    min = 0
    max = 1
    num_data, size, y, x = case.shape
    if num_data==1:
        case = case.reshape(size, y, x)
        # sagital
        fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 3, num_data), dpi=150)
        for i in range(size):
            axes[i].imshow(case[:, :, i].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[i].set_title('x = %d' % i)
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        plt.show()
        # coronal
        fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 3, num_data), dpi=150)
        for i in range(size):
            axes[i].imshow(case[:, i, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[i].set_title('y = %d' % i)
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        plt.show()
        # axial
        fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 3, num_data), dpi=150)
        for i in range(size):
            axes[i].imshow(case[i, :, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
            axes[i].set_title('z = %d' % i)
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        plt.show()
    else:
        # sagital
        fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 2, num_data), dpi=150)
        for i in range(size):
            for j in range(num_data):
                axes[j, i].imshow(case[j, :, :, i].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
                axes[j, i].set_title('x = %d' % i)
                axes[j, i].get_xaxis().set_visible(False)
                axes[j, i].get_yaxis().set_visible(False)
        plt.show()
        # coronal
        fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 2, num_data), dpi=150)
        for i in range(size):
            for j in range(num_data):
                axes[j, i].imshow(case[j, :, i, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
                axes[j, i].set_title('y = %d' % i)
                axes[j, i].get_xaxis().set_visible(False)
                axes[j, i].get_yaxis().set_visible(False)
        plt.show()
        # axial
        fig, axes = plt.subplots(ncols=size, nrows=num_data, figsize=(size - 2, num_data), dpi=150)
        for i in range(size):
            for j in range(num_data):
                axes[j, i].imshow(case[j, i, :, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
                axes[j, i].set_title('z = %d' % i)
                axes[j, i].get_xaxis().set_visible(False)
                axes[j, i].get_yaxis().set_visible(False)
        plt.show()

def save_img_planes(img, path):
    # img: image data (size, size, size)
    os.makedirs(path, exist_ok=True)
    min = 0
    max = 1
    img = np.reshape(img, [9, 9, 9])
    z, y, x = img.shape
    # sagital
    fig, axes = plt.subplots(ncols=x, nrows=1, figsize=(x - 3, 1), dpi=150)
    for i in range(x):
        axes[i].imshow(img[:, :, i].reshape(y, z), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
        axes[i].set_title('x = %d' % i)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    plt.savefig(path + '/sagital.png')
    # coronal
    fig, axes = plt.subplots(ncols=y, nrows=1, figsize=(y - 3, 1), dpi=150)
    for i in range(y):
        axes[i].imshow(img[:, i, :].reshape(z, x), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
        axes[i].set_title('y = %d' % i)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    plt.savefig(path + '/coronal.png')
    # axial
    fig, axes = plt.subplots(ncols=z, nrows=1, figsize=(z - 3, 1), dpi=150)
    for i in range(z):
        axes[i].imshow(img[i, :, :].reshape(x, y), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
        axes[i].set_title('z = %d' % i)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    plt.savefig(path + '/axial.png')

def display_center_slices(case, size, num_data, outdir):
    # case: image data, num_data: number of data, size: length of a side
    min = np.min(case)
    max = np.max(case)
    # axial
    fig, axes = plt.subplots(ncols=num_data, nrows=1, figsize=(num_data, 2))
    for i in range(num_data):
        axes[i].imshow(case[i, 3, :].reshape(size, size), cmap=cm.Greys_r, vmin=min, vmax=max, interpolation='none')
        axes[i].set_title('img%d' % i)
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    plt.savefig(os.path.join(outdir, "interpolation.png"))
    # plt.show()

def init_freudenthal_3d(width, height, depth):
    """
    Freudenthal triangulation of 2d grid
    """
    s = SimplicialComplex()
    # row-major format
    # 0-cells
    for i in range(depth):
        for j in range(height):
            for k in range(width):
               ind = i*width*height + j*width + depth
               s.append([ind])
    # 1-cells
    for i in range(depth):
        for j in range(height):
            for k in range(width-1):
                ind = i * width * height + j * width + depth
                s.append([ind, ind + 1])
    for i in range(depth-1):
        for j in range(width):
            for k in range(width):
                ind = i*width + j
                s.append([ind, ind + width])
    # 2-cells + diagonal 1-cells
    for i in range(depth-1):
        for j in range(width-1):
            for k in range(width):
                ind = i*width + j
                # diagonal
                s.append([ind, ind + width + 1])
                # 2-cells
                s.append([ind, ind + 1, ind + width + 1])
                s.append([ind, ind + width, ind + width + 1])
    return s

def init_tri_complex_3d(width, height, depth):
    """
    initialize 3d complex in dumbest possible way
    """
    # initialize complex to use for persistence calculations
    axis_x = np.arange(0, width)
    axis_y = np.arange(0, height)
    axis_z = np.arange(0, depth)
    grid_axes = np.array(np.meshgrid(axis_x, axis_y, axis_z))
    grid_axes = np.transpose(grid_axes, (1, 2, 3, 0))

    # creation of a complex for calculations
    tri = Delaunay(grid_axes.reshape([-1, 3]))
    return unique_simplices(tri.simplices, 3)

def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)

def diag_tidy(diag, eps=1e-1):
    new_diag = []
    for _, x in diag:
        if np.abs(x[0] - x[1]) > eps:
            new_diag.append((_, x))
    return new_diag

def calc_PH(data):
    z, y, x = data.shape
    size = x * y * z
    # data = np.ndarray(data, [z, y, x])
    cpx = init_tri_complex_3d(z, y, x)
    layer = LevelSetLayer(cpx, maxdim=2, sublevel=False)
    dgminfo = layer(torch.from_numpy(data).float())
    # f01 = TopKBarcodeLengths(dim=0, k=1)
    f0 = TopKBarcodeLengths(dim=0, k=size)
    f1 = TopKBarcodeLengths(dim=1, k=size)
    f2 = TopKBarcodeLengths(dim=2, k=size)
    b01 = f0(dgminfo)[0]
    b0 = f0(dgminfo)[1:].sum()
    b1 = f1(dgminfo).sum()
    b2 = f2(dgminfo).sum()
    l01 = 1. - f0(dgminfo)[0]**2
    l0 = (f0(dgminfo)[1:]**2).sum()
    l1 = (f1(dgminfo)**2).sum()
    l2 = (f2(dgminfo)**2).sum()

    return b01, b0, b1, b2, l01, l0, l1, l2


def getPB(dgminfo, dim):
    dgms, issublevel = dgminfo
    start, end = get_start_end(dgms[dim], issublevel)
    lengths = end - start
    death = start[lengths != 0]
    birth = end[lengths != 0]
    bar = torch.stack([1 - birth, 1 - death], dim=1).tolist()
    if len(bar) != 0:
        diag = []
        for i in range(len(bar)):
            diag.append([dim, bar[i]])
        return diag
    else:
        return None


def drawPB(data):
    z, y, x = data.shape
    cpx = init_tri_complex_3d(z, y, x)
    layer = LevelSetLayer(cpx, maxdim=2, sublevel=False)
    dgminfo = layer(torch.from_numpy(data).float())
    diag = []
    diag2 = getPB(dgminfo, 2)
    diag1 = getPB(dgminfo, 1)
    diag0 = getPB(dgminfo, 0)
    if diag2 != None: diag += diag2
    if diag1 != None: diag += diag1
    if diag0 != None: diag += diag0

    diag = diag_tidy(diag, 1e-3)
    print(diag)

    plt.figure(figsize=(3, 3))
    gd.plot_persistence_barcode(diag, max_intervals=0,inf_delta=100)
    plt.xlim(0, 1)
    plt.ylim(-1, len(diag))
    plt.xticks(ticks=np.linspace(0, 1, 6), labels=np.round(np.linspace(1, 0, 6), 2))
    plt.yticks([])
    plt.show()


def compute_Betti_bumbers(img, th=0.5):
    print(img.shape)
    z, y, x = img.shape
    img_v = img.flatten()
    img_v = (img_v > th) * 1.0
    cc = gd.CubicalComplex(dimensions=(x, y, z),
                           top_dimensional_cells=1 - img_v)
    print(cc.betti_numbers())

def PH_diag(img, patch_side):
    cc = gd.CubicalComplex(dimensions=(patch_side, patch_side, patch_side),
                           top_dimensional_cells=1 - img.flatten())
    diag = cc.persistence()
    plt.figure(figsize=(3, 3))
    diag_clean = diag_tidy(diag, 1e-3)
    gd.plot_persistence_barcode(diag_clean, max_intervals=0,inf_delta=100)
    print(diag_clean)
    plt.xlim(0, 1)
    plt.ylim(-1, len(diag_clean))
    plt.xticks(ticks=np.linspace(0, 1, 6), labels=np.round(np.linspace(1, 0, 6), 2))
    plt.yticks([])
    plt.show()
    # gd.plot_persistence_diagram(diag_clean, legend=True)
    # plt.show()
    # gd.plot_persistence_density(diag_clean, legend=True)
    # plt.show()

def save_PH_diag(img, outdir):
    z, y, x = img.shape
    os.makedirs(outdir, exist_ok=True)
    cc = gd.CubicalComplex(dimensions=(z, y, x),
                           top_dimensional_cells=1 - img.flatten())
    diag = cc.persistence()
    # diag_clean = diag_tidy(diag, 1e-3)
    # print(diag_clean)
    # np.savetxt(os.path.join(outdir, 'PH.csv'), diag_clean, delimiter=",")
    # with open(os.path.join(outdir, 'generalization.txt'), 'wt') as f:
    #     for ele in diag_clean:
    #         f.write(ele + '\n')
    fig1 = plt.figure(figsize=(3, 3))
    gd.plot_persistence_barcode(diag, max_intervals=0,inf_delta=100)
    plt.xlim(0, 1)
    plt.ylim(-1, len(diag))
    plt.xticks(ticks=np.linspace(0, 1, 6), labels=np.round(np.linspace(1, 0, 6), 2))
    plt.yticks([])
    plt.savefig(os.path.join(outdir, "Persistence_barcode.png"))

    fig2 = plt.figure()
    gd.plot_persistence_diagram(diag, legend=True)
    plt.savefig(os.path.join(outdir, "Persistence_diagram.png"))

    fig3 = plt.figure()
    gd.plot_persistence_density(diag, legend=True)
    plt.savefig(os.path.join(outdir, "Persistence_density.png"))

def get_dataset(input, patch_side, num_of_test):
    print('load data')
    list = io.load_list(input)
    data_set = np.zeros((num_of_test, patch_side, patch_side, patch_side))
    for i in trange(num_of_test):
        data_set[i, :] = np.reshape(io.read_mhd_and_raw(list[i]), [patch_side, patch_side, patch_side])
    return data_set

def plt_loss(epochs, train_loss_list, val_loss_list, outdir):
    # plot graph
    plt.figure()
    plt.plot(range(epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.grid()
    plt.savefig(outdir + "loss.png")

def plt_latent(latent_space, axis_n, std, out_path='latent_space.png', label=np.zeros(0)):
    # plt latent space
    a1, a2 = axis_n
    plt.figure(figsize=(10, 10))
    plt.axis([-3.5 * std[a1], 3.5 * std[a1], -3.5 * std[a2], 3.5 * std[a2]])  # グラフ範囲の手動設定 [xmin, xmax, ymin, ymax]
    if label.size==0:
        plt.scatter(latent_space[:, a1], latent_space[:, a2])
    else:
        plt.scatter(latent_space[:, a1], latent_space[:, a2], c=label)
        plt.colorbar()
    plt.xlabel('z {}'.format(a1))
    plt.ylabel('z {}'.format(a2))
    plt.title('Latent space')
    plt.savefig(out_path)

def plt_grid(fig, grid_x, grid_y, out_path="./grid.png", a1=0, a2=1, n_data=7, patch_side=9):

    # set graph
    start_range = patch_side // 2
    end_range = n_data * patch_side + start_range + 1
    pixel_range = np.arange(start_range, end_range, patch_side)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)

    plt.figure(figsize=(10, 10))
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    # plt.xticks(color="None")
    # plt.yticks(color="None")
    # plt.tick_params(length=0)
    plt.xlabel('z {}'.format(a1))
    plt.ylabel('z {}'.format(a2))
    plt.imshow(fig, cmap='Greys_r', vmin = 0, vmax = 1, interpolation='none')
    plt.savefig(out_path)
    plt.show()

def morphing(model, l, out_path, patch_side=9):
    patch_center = int(patch_side / 2)
    morphing = model.decode(l).data.cpu().numpy()
    morphing_img = morphing[0].reshape(patch_side, patch_side, patch_side)
    morphing_axial = morphing_img[patch_center, :, :]
    fig = plt.imshow(morphing_axial, cmap='Greys_r', vmin = 0, vmax = 1, interpolation='none')
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    plt.savefig(out_path + 'linear_morphing/' + str(l)  + 'fig.png')

