import torch
import torch.nn.functional as F

'''
This code is tutorial that how flow works in spatial transformation.
Note that grid is used when applying grid_sample function.
'''

Y = torch.zeros((60, 512, 512))
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

# Check whether this is same as identity transformation
thetai = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float).unsqueeze(0).repeat_interleave(t, dim=0)  # t,2,3
gridi = F.affine_grid(thetai, Y_reshape.size(), align_corners=True)
print(f'difference: {torch.max(torch.abs(gridi - default_grid))}')

# Now we examine how the flow can be computed and composed
theta1 = torch.tensor([[1, 0, 10], [0, 1, 0]], dtype=torch.float).unsqueeze(0).repeat_interleave(t, dim=0)
grid1 = F.affine_grid(theta1, Y_reshape.size(), align_corners=True)
flow1 = grid1 - default_grid  # t,h,w,2
print(f'flow1 spec: {flow1.size()}, {flow1.max()}, {flow1.min()}')

theta2 = torch.tensor([[1, 0, 0], [0, 1, 10]], dtype=torch.float).unsqueeze(0).repeat_interleave(t, dim=0)
grid2 = F.affine_grid(theta2, Y_reshape.size(), align_corners=True)
flow2 = grid2 - default_grid  # t,h,w,2
print(f'flow2 spec: {flow2.size()}, {flow2.max()}, {flow2.min()}')

# flow1 + flow2 should be same as flow3 which is composition of two transformations
theta3 = torch.tensor([[1, 0, 10], [0, 1, 10]], dtype=torch.float).unsqueeze(0).repeat_interleave(t, dim=0)
grid3 = F.affine_grid(theta3, Y_reshape.size(), align_corners=True)
flow3 = grid3 - default_grid  # t,h,w,2
print(f'flow3 - flow1 - flow2: {torch.max(torch.abs(flow3 - flow1 - flow2))}')
