from utils import *
import dataIO as io
import argparse

parser = argparse.ArgumentParser(description='VAE test')
parser.add_argument('--input', type=str, default="E:/git/pytorch/vae/results/CT/z_24/B_0.1/R_10000.0/gen/rec/list.txt",
                    help='File path of input images')
parser.add_argument('--patch_side', type=int, default=9,
                    help='how long patch side for input')
parser.add_argument('--num_of_data', type=int, default=2,
                    help='number of dataset')
parser.add_argument('--output', type=str, default="E:/git/pytorch/vae/results/debug/gen/",
                    help='File path of output images')
args = parser.parse_args()

# get data
data_set = get_dataset(args.input, args.patch_side, args.num_of_data)

# threshold
# for th in np.linspace()

# data_set = data_set > 0.90
# display image
display_slices(data_set[args.num_of_data-1:args.num_of_data,:])
# print(data_set[args.num_of_data-1:args.num_of_data,:])

# plot PH diagram
PH_diag(data_set[args.num_of_data-1], args.patch_side)
# save_PH_diag(data_set[args.num_of_data-1], args.patch_side, args.output)