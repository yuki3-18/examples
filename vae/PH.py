from utils import *
import dataIO as io
import argparse

parser = argparse.ArgumentParser(description='VAE test')
parser.add_argument('--input', type=str, default="E:/git/pytorch/vae/results/artificial/tip/z_3/B_0.1/L_0/gen/rec/list.txt",
                    help='File path of input images')
parser.add_argument('--patch_side', type=int, default=9,
                    help='how long patch side for input')
parser.add_argument('--num_of_data', type=int, default=1380,
                    help='number of dataset')
parser.add_argument('--output', type=str, default="E:/git/pytorch/vae/results/artificial/tip/z_3/B_0.1/L_0/gen/rec/",
                    help='File path of output images')
args = parser.parse_args()

# get data
data_set = get_dataset(args.input, args.patch_side, args.num_of_data)
# data_set = min_max(data_set)
# threshold
# for th in np.linspace()
data_set = data_set > 0.33

# display image
display_slices(data_set[args.num_of_data-1:args.num_of_data,:])
# print(data_set[args.num_of_data-1:args.num_of_data,:])

# compute_Betti_bumbers(data_set[args.num_of_data-1])

# plot PH diagram
# PH_diag(data_set[args.num_of_data-1], args.patch_side)
# save_PH_diag(data_set[args.num_of_data-1], args.output)

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
