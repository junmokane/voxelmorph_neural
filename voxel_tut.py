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
tr, ro = '12.0', '8.0'
data_path = f'./zebrafish/Y_tr_{tr}_ro_{ro}'
x_train = io.imread(f'{data_path}/Y_tr_{tr}_ro_{ro}_0.tif').astype(float)  # 60,512,512

# experiment setting
exp = f'./result/[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]_tr_{tr}_ro_{ro}'
os.makedirs(exp, exist_ok=True)

# data generation for evaluation
Y = torch.from_numpy(io.imread(f'./zebrafish/Y.tif').astype(float)).float()  # 60,512,512
Y /= Y.max()
t, w, h = Y.size()
Y_reshape = Y.view(t, 1, w, h)  # 60x1x512x512
tau = scio.loadmat(f'{data_path}/tau_tr_{tr}_ro_{ro}_0.mat')['tau'][0:1, 0:2].repeat(60, axis=0)
grid =  F.affine_grid(torch.from_numpy(tau).float(), Y_reshape.size(), align_corners=True)
Y_tau = F.grid_sample(Y_reshape, grid, align_corners=True)  # 60x1x512x512
gt = Y_tau.numpy()[:, 0]  # 60x512x512

# configure unet input shape (concatenation of moving and fixed images)
ndim = 2
unet_input_features = 2
inshape = (*x_train.shape[1:], unet_input_features)

# configure unet features 
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]

# build model
unet = vxm.networks.Unet(inshape=inshape, nb_features=nb_features)
print('input shape: ', unet.input.shape)
print('output shape:', unet.output.shape)

# transform the results into a flow field.
disp_tensor = tf.keras.layers.Conv2D(ndim, kernel_size=3, padding='same', name='disp')(unet.output)

# check tensor shape
print('displacement tensor:', disp_tensor.shape)

# using keras, we can easily form new models via tensor pointers
def_model = tf.keras.models.Model(unet.inputs, disp_tensor)

# build transformer layer
spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

# extract the first frame (i.e. the "moving" image) from unet input tensor
moving_image = tf.expand_dims(unet.input[..., 0], axis=-1)

# warp the moving image with the transformer
moved_image_tensor = spatial_transformer([moving_image, disp_tensor])

outputs = [moved_image_tensor, disp_tensor]
vxm_model = tf.keras.models.Model(inputs=unet.inputs, outputs=outputs)

# build model using VxmDense
inshape = x_train.shape[1:]
vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

# voxelmorph has a variety of custom loss classes
# usually, we have to balance the two losses by a hyper-parameter
# losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
# lambda_param = 0.05
# loss_weights = [1, lambda_param]
# vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

# losses and loss weights
losses = ['mse', vxm.losses.Grad('l2').loss]
loss_weights = [1, 0.01]
vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)

def vxm_data_generator(x_data, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0] - 1, size=batch_size)
        moving_images = x_data[1:][idx1, ..., np.newaxis]
        # idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[0:1, ..., np.newaxis].repeat(batch_size, axis=0)
        # print(fixed_images.shape)
        inputs = [moving_images, fixed_images]
        
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]
        
        yield (inputs, outputs)

# let's test it
train_generator = vxm_data_generator(x_train, batch_size=8)
in_sample, out_sample = next(train_generator)

# visualize
images = [img[0, :, :, 0] for img in in_sample + out_sample] 
titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);
plt.savefig(f'{exp}/sample.png')
plt.close()

# train
nb_epochs = 50
steps_per_epoch = 100
hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)

def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(f'{exp}/loss_curve.png')
    plt.close()

plot_history(hist)

# evaluate result
moving_images = x_train[1:][:, ..., np.newaxis]  # 59x512x512x1     
fixed_images = x_train[0:1, ..., np.newaxis].repeat(59, axis=0)  # 59x512x512x1
val_input = [moving_images, fixed_images]
registered_images, flow = vxm_model.predict(val_input)  # 59x512x512x1, 59x512x512x2

mse = np.mean((gt[1:] - registered_images[..., 0]) ** 2)
print(f'mse is {mse}')

# save result
plt.imsave(f'{exp}/fix_image.png', fixed_images[0, ..., 0], cmap='gray')
for i in range(59):
    plt.imsave(f'{exp}/{i}_mov_image.png', moving_images[i, ..., 0], cmap='gray')
    plt.imsave(f'{exp}/{i}_reg_image.png', registered_images[i, ..., 0], cmap='gray')
f = open(f'{exp}/mse.txt', 'w')
f.write(str(mse))
f.close()    