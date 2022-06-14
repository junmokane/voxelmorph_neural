import skimage.io as io
import numpy as np
import torch
import torch.nn.functional as F

'''
Note that flow should satisfy that grid = flow + default_grid
flow: t,h,w,2 (default_grid is identity transformation grid)

Synthetic deformation flow -> flow_syn
REALS deformation flow -> flow_reals
Composed flow -> flow_syn + flow_reals
'''

# read data
Y = torch.from_numpy(io.imread(f'./zebrafish/Y.tif').astype(float)).float()  # 60,512,512
Y /= Y.max()
t, w, h = Y.size()
Y_reshape = Y.view(t, 1, w, h)  # 60x1x512x512

# Generate default grid (grid that does identity transformation)
size = [512, 512]
vectors = [torch.arange(0, s) for s in size]
grids = torch.meshgrid(vectors)
default_grid = torch.stack(grids)
default_grid = torch.unsqueeze(default_grid, 0)
default_grid = default_grid.type(torch.FloatTensor)
for i in range(len(size)):
    default_grid[:, i, ...] = 2 * (default_grid[:, i, ...] / (size[i] - 1) - 0.5)
default_grid = default_grid.permute(0, 3, 2, 1)  # 1,h,w,2 (w,h are flipped!)
default_grid = default_grid.repeat_interleave(t, dim=0) # t,h,w,2
print(f'default_grid spec: {default_grid.size()}, {default_grid.max()}, {default_grid.min()}')

# Read synthetic deformation flow
theta_syn = torch.tensor([[1, 0, 10], [0, 1, 0]], dtype=torch.float).unsqueeze(0).repeat_interleave(t, dim=0)
grid_syn = F.affine_grid(theta_syn, Y_reshape.size(), align_corners=True)
flow_syn = grid_syn - default_grid  # t,h,w,2
print(f'flow_syn spec: {flow_syn.size()}, {flow_syn.max()}, {flow_syn.min()}')

# Read reals deformation flow 
theta_reals = torch.tensor([[1, 0, 0], [0, 1, 10]], dtype=torch.float).unsqueeze(0).repeat_interleave(t, dim=0)
grid_reals = F.affine_grid(theta_reals, Y_reshape.size(), align_corners=True)
flow_reals = grid_reals - default_grid  # t,h,w,2
print(f'flow_reals spec: {flow_reals.size()}, {flow_reals.max()}, {flow_reals.min()}')

# Composition of two deformation flows
flow_compose = flow_syn + flow_reals  # t,h,w,2
grid_compose = default_grid + flow_compose
grid_mu = grid_compose.mean(dim=0, keepdim=True).repeat_interleave(t, dim=0)

# Calculate mse loss
Y_tau = F.grid_sample(Y_reshape, grid_compose, align_corners=True)  # 60x1x512x512
Y_mu = F.grid_sample(Y_reshape, grid_mu, align_corners=True)  # 60x1x512x512
mse = torch.mean((Y_tau - Y_mu) ** 2)
print(f'mse is {mse}')  # this is definitely zero!