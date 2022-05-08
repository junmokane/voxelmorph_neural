# VoxelMorph on Neural Data

This is same code from VoxelMorph tutorial but applied to neural data. We test VoxelMorph on Zebrafish brain dataset (not provided).

## How does it work?

We have a data with size (t, h, w) = (60, 512, 512), where t, h, w is time, height, and width respectively. We set fixed image as the first image of the data, and moving images as the remaining images of the data. So, we simply register 59 moving images to single fixed image using VoxelMorph. The result shows that registration partially works, while the neuronal signal is destroyed due to objective used in VoxelMorph (objective is simply reconstruction error). We calculate MSE loss between images with registration and ground truth image data. 

## MSE loss of baselines

|Method|(tr, ro) = (12, 8)|(tr, ro) = (18, 12)|(tr, ro) = (24, 16)|
|:---|:---:|:---:|:---:|
|REALS|1.0644e-5 |1.01501e-5|1.0439e-5|
|VoxelMorph|1.2246e-3|1.9343e-3|2.2554e-3|
|Without Training|4.5476e-3|6.5298e-3|7.1357e-3|

## Result sample