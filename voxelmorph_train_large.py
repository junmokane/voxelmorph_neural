# imports
import os, sys
import shutil
import glob
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

from calc_general_mse import calculate_mse, get_default_grid


folder_list = glob.glob(f'./zebrafish/deformable_REALS_data_final_0.75_*div512_0616/tr_*_ro_*') 

for folder in folder_list:
    root1 = folder.split('/')[-2]  # deformable_REALS_data_final_0.75_*div512_0616
    root2 = folder.split('/')[-1]  # tr_*_ro_*
    token = root2.split('_')
    tr = token[1] 
    ro = token[3]
    
    exp = f'./result/{root1}/{root2}'
    if os.path.exists(exp) and os.path.isdir(exp):
        shutil.rmtree(exp)
    os.makedirs(exp, exist_ok=True)
    print(f'current folder: {exp}')

    for num in range(1):
        # data specification
        x_train = io.imread(f'{folder}/Y_tr_{tr}_ro_{ro}_{num}.tif').astype(float)  # t,h,w
        x_train /= x_train.max()
        print(x_train.shape, x_train.max(), x_train.min())

        # experiment setting
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        s_time = time.time()

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

        # train
        nb_epochs = 60
        steps_per_epoch = 100
        hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)

        def plot_history(hist, loss_name='loss'):
            # Simple function to plot training history.
            plt.figure()
            plt.plot(hist.epoch, hist.history[loss_name], '.-')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig(f'{exp}/loss_curve_{num}.png')
            plt.close()

        plot_history(hist)

        # evaluate result
        moving_images = x_train[1:][:, ..., np.newaxis]  # 59x512x512x1     
        fixed_images = x_train[0:1, ..., np.newaxis].repeat(59, axis=0)  # 59x512x512x1
        val_input = [moving_images, fixed_images]
        registered_images, flow_voxelmorph = vxm_model.predict(val_input)  # 59x512x512x1, 59x512x512x2
        flow_voxelmorph = torch.from_numpy(flow_voxelmorph).type(torch.float32)  # t-1,h,w,2
    
        # This is done because of official voxelmoprh implementation.
        # normalization should be reconsidered and channel change.
        size = [512, 512]
        for i in range(len(size)):
            flow_voxelmorph[..., i] = 2 * (flow_voxelmorph[..., i] / (size[i] - 1))
        flow_voxelmorph = flow_voxelmorph[..., [1, 0]]

        # read original static data
        Y = torch.from_numpy(io.imread(f'./zebrafish/Y.tif').astype(float)).float()  # 60,512,512
        Y /= Y.max()
        t, w, h = Y.size()
        Y_reshape = Y.view(t, 1, w, h)[1:]  # t-1,1,h,w

        '''
        Calculate mse.
        This is done by reading affine, deformation tr and apply those to
        original static gt data, then apply voxelmorph
        '''
        # deformation flow
        flow_deform = io.imread(f'{folder}/flow_only_tr_{tr}_ro_{ro}_{num}.tif').astype(float)  # t,2,h,w
        flow_deform = torch.from_numpy(flow_deform).permute(0, 2, 3, 1)[1:].type(torch.float32)  # t-1,h,w,2
        
        # affine flow
        tau_affine = scio.loadmat(f'{folder}/tau_tr_{tr}_ro_{ro}_{num}.mat')['tau'][1:, :2, :3]
        grid_affine =  F.affine_grid(torch.from_numpy(tau_affine).float(), Y_reshape.size(), align_corners=True)
        flow_affine = grid_affine - get_default_grid(Y_reshape)
        
        flow_syn = flow_deform + flow_affine
        mse = float(calculate_mse(Y_reshape, flow_syn, flow_voxelmorph))
        print(flow_voxelmorph.size(), flow_voxelmorph.max(), flow_voxelmorph.min())
        print(flow_syn.size(), flow_syn.max(), flow_syn.min())
        print(f'mse is {mse}')

        # this is to check whether moving_images is same as flow applied one
        Y_syn = F.grid_sample(Y_reshape, get_default_grid(Y_reshape) + flow_syn, align_corners=True)  # 59x1x512x512
        Y_syn = Y_syn.detach().numpy()

        # save result
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        e_time = time.time()
        moving_images_sv = (moving_images[..., 0] * 255).astype(np.uint8)
        Y_syn_sv = (Y_syn[:, 0] * 255).astype(np.uint8)
        cat_sv = np.concatenate((moving_images_sv, Y_syn_sv), axis=2)
        registered_images_sv = (registered_images[..., 0] * 255).astype(np.uint8)
        vio.vwrite(f'{exp}/moving_images_{num}.mp4', moving_images_sv)
        vio.vwrite(f'{exp}/moving_images_from_flow_{num}.mp4', Y_syn_sv)
        vio.vwrite(f'{exp}/moving_images_and_from_flow_{num}.mp4', cat_sv)
        vio.vwrite(f'{exp}/registered_images_{num}.mp4', registered_images_sv)
        io.imsave(f'{exp}/flow_syn_{num}.tif', flow_syn.detach().numpy())
        io.imsave(f'{exp}/flow_voxelmorph_{num}.tif', flow_voxelmorph.detach().numpy())
        f = open(f'{exp}/mse_{num}.txt', 'w')
        f.write(str(mse)+'\n')
        f.write(str(e_time - s_time))
        f.close()    
