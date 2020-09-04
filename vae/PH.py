from utils import *
import dataIO as io
import argparse
import SimpleITK as sitk
from topologylayer.nn.features import get_start_end


parser = argparse.ArgumentParser(description='VAE test')
parser.add_argument('--input', type=str, default="E:/git/pytorch/vae/results/artificial/hole/z_6/B_0.1/batch128/L_60000/C_10/gen/",
                    help='File path of input images')
parser.add_argument('--patch_side', type=int, default=9,
                    help='how long patch side for input')
parser.add_argument('--num_of_data', type=int, default=2,
                    help='number of dataset')
args = parser.parse_args()

# set path
data_path = os.path.join(args.input, 'rec/list.txt')
ori_path = os.path.join(args.input, 'ori/list.txt')
gs_path ="E:/git/pytorch/vae/input/hole0/std/"
out_path = os.path.join(args.input, '{}'.format(args.num_of_data))
os.makedirs(out_path, exist_ok=True)

# get data
list = io.load_list(data_path)
ori_list = io.load_list(ori_path)
gs_file = os.path.basename(ori_list[args.num_of_data-1])
gs_list = os.path.join(gs_path, gs_file)

img = np.reshape(io.read_mhd_and_raw(list[args.num_of_data-1]), [args.patch_side, args.patch_side, args.patch_side])
ori_img = np.reshape(io.read_mhd_and_raw(ori_list[args.num_of_data-1]), [args.patch_side, args.patch_side, args.patch_side])
gs_img = np.reshape(io.read_mhd_and_raw(gs_list), [args.patch_side, args.patch_side, args.patch_side])


# data_set = get_dataset(data_path, args.patch_side, args.num_of_data)
# ori_data = get_dataset(ori_path, args.patch_side, args.num_of_data)
# gs_data = get_dataset(gs_path, args.patch_side, args.num_of_data)
# # data_set = min_max(data_set)
# img = data_set[args.num_of_data-1:args.num_of_data,:]
# ori_img = ori_data[args.num_of_data-1:args.num_of_data,:]
# gs_img = gs_data[args.num_of_data-1:args.num_of_data,:]

# dif = np.abs(img - ori_img)
# dif_gs = np.abs(img - gs_img)

# threshold
# for th in np.linspace()
# img = 1 - ori_img
# img = gs_img
# th = 0.2
# img = (img > th) * 1
# print(data_set)

# display image
display_slices(img.reshape([1, 9, 9, 9]))
# save_img_planes(img, out_path)
# save_img_planes(ori_img, out_path+'/ori')
# save_img_planes(gs_img, out_path+'/gs')
# save_img_planes(dif, out_path+'/dif')
# save_img_planes(dif_gs, out_path+'/dif_gs')

# plot PH diagram
# compute_Betti_bumbers(data_set[args.num_of_data-1])
PH_diag(img, args.patch_side)
print(drawPB(img))
# save_PH_diag(img, out_path)
# save_PH_diag(ori_img, out_path+'/ori')
# save_PH_diag(gs_img, out_path+'/gs')

# # persistent
# bar01, bar0, bar1, bar2 = [], [], [], []
# for i in trange(args.num_of_data):
#     b01, b0, b1, b2 = calc_PH(data_set[i])
#     bar01.append(b01.item())
#     bar0.append(b0.item())
#     bar1.append(b1.item())
#     bar2.append(b2.item())
# bar = [bar01, bar0, bar1, bar2]
# bar = np.transpose(bar)
# np.savetxt(os.path.join(args.output, 'topo.csv'), bar, delimiter=',')
