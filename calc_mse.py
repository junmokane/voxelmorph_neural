# imports
import os, sys
import skimage.io as io
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

# data specification
tr, ro = '24.0', '16.0'
registered_images = io.imread(f'./zebrafish/deformable_reals/L_tr_{tr}_ro_{ro}_0.tif').astype(float)  # 60,512,512
print(registered_images.shape, registered_images.max(), registered_images.min())

# data generation for evaluation
data_path = f'./zebrafish/Y_tr_{tr}_ro_{ro}'
Y = torch.from_numpy(io.imread(f'./zebrafish/Y.tif').astype(float)).float()  # 60,512,512
Y /= Y.max()
t, w, h = Y.size()
Y_reshape = Y.view(t, 1, w, h)  # 60x1x512x512
tau = scio.loadmat(f'{data_path}/tau_tr_{tr}_ro_{ro}_0.mat')['tau'][0:1, 0:2].repeat(60, axis=0)
grid =  F.affine_grid(torch.from_numpy(tau).float(), Y_reshape.size(), align_corners=True)
Y_tau = F.grid_sample(Y_reshape, grid, align_corners=True)  # 60x1x512x512
gt = Y_tau.numpy()[:, 0]  # 60x512x512

mse = np.mean((gt[1:] - registered_images[1:]) ** 2)
print(f'mse is {mse}')
