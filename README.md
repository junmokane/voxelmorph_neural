# VoxelMorph on Neural Data

This is same code from VoxelMorph tutorial but applied to neural data. We test VoxelMorph on Zebrafish brain dataset (not provided).

## How does it work?

We have a data with size (t, h, w) = (60, 512, 512), where t, h, w is time, height, and width respectively. We set fixed image as the first image of the data, and moving images as the remaining images of the data. So, we simply register 59 moving images to single fixed image using VoxelMorph. The result shows that registration partially works, while the neuronal signal is destroyed due to objective used in VoxelMorph (objective is simply reconstruction error).

## Result sample