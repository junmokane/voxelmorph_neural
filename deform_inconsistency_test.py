import sys
sys.path.append(".")
from scripts.stn import SpatialTransformation
import torch
import torch.nn.functional as F
import skimage.io as skio
import scipy.io as scio
import matplotlib.pyplot as plt

import skimage.io as skio
import numpy as np
torch.manual_seed(0)
np.random.seed(0)


dir = './data/2d_zebrafish_brain_data'
path = f'{dir}/Y.tif'


Y = torch.from_numpy(skio.imread(path).astype(float)).float()[:512]  # (t,w,h)
t, w, h = Y.size()







size = [512, 512]
vectors = [torch.arange(0, s) for s in size]
grids = torch.meshgrid(vectors)
grid = torch.stack(grids)
grid = torch.unsqueeze(grid, 0)
grid = grid.type(torch.FloatTensor)
print(grid.max(), grid.size())

for i in range(len(size)):
    grid[:, i, ...] = 2 * (grid[:, i, ...] / (size[i] - 1) - 0.5)
print(grid.max(), grid.size())




theta1 = torch.tensor([[1, 0, 0],
                    [0, 1, 0]]
                    , dtype=torch.float)
theta1 = theta1.unsqueeze(0).repeat(1, 1, 1).detach().requires_grad_(True)
print(theta1.size())
# exit()

print(Y.size())
gtgrid = F.affine_grid(theta1, Y[0:1, ...].unsqueeze(1).size(), align_corners=True)[0].permute(2, 0, 1)
print(gtgrid.size(), gtgrid.max())




theta1 = torch.tensor([[1, 0, 10],
                    [0, 1, 0]]
                    , dtype=torch.float)
theta1 = theta1.unsqueeze(0).repeat(1, 1, 1).detach().requires_grad_(True)
grid1 = F.affine_grid(theta1, Y[0:1, ...].unsqueeze(1).size(), align_corners=True)[0].permute(2, 0, 1)
flow1 = grid1 - gtgrid

theta2 = torch.tensor([[1, 0, 0],
                    [0, 1, 10]]
                    , dtype=torch.float)
theta2 = theta2.unsqueeze(0).repeat(1, 1, 1).detach().requires_grad_(True)
grid2 = F.affine_grid(theta2, Y[0:1, ...].unsqueeze(1).size(), align_corners=True)[0].permute(2, 0, 1)
flow2 = grid2 - gtgrid


theta3 = torch.tensor([[1, 0, 10],
                    [0, 1, 10]]
                    , dtype=torch.float)
theta3 = theta3.unsqueeze(0).repeat(1, 1, 1).detach().requires_grad_(True)
grid3 = F.affine_grid(theta3, Y[0:1, ...].unsqueeze(1).size(), align_corners=True)[0].permute(2, 0, 1)
flow3 = grid3 - gtgrid


flow12 = flow1 + flow2

print(torch.norm(flow3 - flow12, p=1))
skio.imsave("flow1.tif", flow1.detach().cpu().numpy())
skio.imsave("flow2.tif", flow2.detach().cpu().numpy())
skio.imsave("flow3.tif", flow3.detach().cpu().numpy())


raise RuntimeError

theta1 = torch.tensor([[1, 0, 2],
                    [0, 1, 0]]
                    , dtype=torch.float)
theta1 = theta1.unsqueeze(0).repeat(1, 1, 1).detach().requires_grad_(True)
print(theta1.size())
# exit()

print(Y.size())
grid = F.affine_grid(theta1, Y[0:1, ...].unsqueeze(1).size())[0].permute(2, 0, 1)
print(grid.size())

skio.imsave("theta1.tif", grid.detach().cpu().numpy())
print(grid[0, :, :].min(), grid[0, :, :].max())
print(grid[1, :, :].min(), grid[1, :, :].max())
skio.imsave("theta1_vector.tif", (grid - gtgrid).detach().cpu().numpy())





theta2 = torch.tensor([[0.5, -0.5, 1],
                    [0.5, 0.5, 1]]
                    , dtype=torch.float)
theta2 = theta2.unsqueeze(0).repeat(1, 1, 1).detach().requires_grad_(True)
print(theta2.size())
# exit()

print(Y.size())
grid = F.affine_grid(theta2, Y[0:1, ...].unsqueeze(1).size())[0].permute(2, 0, 1)
print(grid.size())

skio.imsave("theta2.tif", grid.detach().cpu().numpy())


skio.imsave("theta2_vector.tif", (grid - gtgrid).detach().cpu().numpy())
