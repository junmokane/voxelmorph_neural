# imports
import os, sys
import skimage.io as io
import skvideo.io as vio
import scipy.io as scio
import time

# third party imports
import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

# local imports
import voxelmorph as vxm
import neurite as ne
import matplotlib.pyplot as plt

from calc_general_mse import calculate_mse

# read original static data (registered gt)
Y = torch.from_numpy(io.imread(f'./zebrafish/Y.tif').astype(float)).float()  # 60,512,512
Y /= Y.max()
t, w, h = Y.size()
Y_reshape = Y.view(t, 1, w, h)[1:]  # 59x1x512x512

# read flow_syn and flow_voxelmorph
tr, ro = '0.0', '0.0'
result_path = f'./result/[2022-06-16 12:26:05]_tr_{tr}_ro_{ro}'

# calculate mse
flow_syn = io.imread(f'{result_path}/flow_syn_0.tif').astype(float)  # t-1,h,w,2
flow_syn = torch.from_numpy(flow_syn).type(torch.float32)  
print(flow_syn.size(), flow_syn.max(), flow_syn.min())

flow_voxelmorph = io.imread(f'{result_path}/flow_voxelmorph_0.tif').astype(float)  # t-1,h,w,2
flow_voxelmorph = torch.from_numpy(flow_voxelmorph).type(torch.float32)  
print(flow_voxelmorph.size(), flow_voxelmorph.max(), flow_voxelmorph.min())

mse = float(calculate_mse(Y_reshape, flow_syn, flow_voxelmorph))
print(f'mse is {mse}')

# save result
size = [512, 512]
vectors = [torch.arange(0, s) for s in size]
grids = torch.meshgrid(vectors)
default_grid = torch.stack(grids)
default_grid = torch.unsqueeze(default_grid, 0)
default_grid = default_grid.type(torch.FloatTensor)

for i in range(len(size)):
    default_grid[:, i, ...] = 2 * (default_grid[:, i, ...] / (size[i] - 1) - 0.5)
default_grid = default_grid.permute(0, 3, 2, 1)  # 1,h,w,2 (w,h are flipped!)
default_grid = default_grid.repeat_interleave(Y_reshape.size(0), dim=0) # t,h,w,2

flow_compose = flow_syn + flow_voxelmorph  # t,h,w,2
grid_compose = default_grid + flow_compose
Y_tau = F.grid_sample(Y_reshape, grid_compose, align_corners=True)  # 59x1x512x512
Y_tau = Y_tau.detach().numpy()
Y_tau_sv = (Y_tau[:, 0, ...] * 255).astype(np.uint8)
vio.vwrite(f'./reg.mp4', Y_tau_sv)






