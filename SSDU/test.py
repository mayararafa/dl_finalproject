import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py as h5
import time
import utils
import parser_ops
from masks.subsample import create_mask_for_mask_type
from sens_map_gen.bart_espirit import espirit
from data import slice_coil
from tqdm import tqdm

parser = parser_ops.get_parser()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# .......................Load the Data...........................................
args.data_opt = "Coronal_PD"
print('\n Loading ' + args.data_opt + ' test dataset...')
kspace_dir, coil_dir, mask_dir, saved_model_dir = utils.get_test_directory(args)

# Read in data
# fastMRI data -> (num_slices, num_coils, h, w)
kspace_test = h5.File(kspace_dir, "r")['kspace']
if args.challenge == "singlecoil":
    kspace_test = np.resize(kspace_test, (kspace_test.shape[0],) + (1,) + kspace_test.shape[1:])

# Crop kspace (h, w) to (args.nrow_GLOB, args.ncol_GLOB)
crop_h = (kspace_test.shape[2] - args.nrow_GLOB) // 2
crop_w = (kspace_test.shape[3] - args.ncol_GLOB) // 2
if crop_h != 0 and crop_w != 0:
    kspace_test = kspace_test[:, :, crop_h:-crop_h, crop_w:-crop_w]
elif crop_w == 0:
    kspace_test = kspace_test[:, :, crop_h:-crop_h, :]
elif crop_h == 0:
    kspace_test = kspace_test[:, :, :, crop_w:-crop_w]

# Reshaped to (num_slices, h, w, num_coils)
kspace_test = np.transpose(kspace_test, (0, 2, 3, 1))

# Slice coil data
kspace_test = slice_coil(kspace_test, args)

print("\nGenerating sensitivity maps - num_slices={} ...".format(kspace_test.shape[0]))
sens_maps_testAll = espirit(args, kspace_test, 6, 24, 0.02, 0.95)

original_mask_func = create_mask_for_mask_type(args.subsample_mask_type, args.center_fractions, [args.acc_rate])
original_mask = original_mask_func(kspace_test.shape[-3:-1], seed=42)  # (h, w)

print('\n Normalize kspace to 0-1 region')
for ii in range(np.shape(kspace_test)[0]):
    kspace_test[ii, :, :, :] = kspace_test[ii, :, :, :] / np.max(np.abs(kspace_test[ii, :, :, :][:]))

# %% Train and loss masks are kept same as original mask during inference
nSlices, *_ = kspace_test.shape
test_mask = np.complex64(np.tile(original_mask[np.newaxis, :, :], (nSlices, 1, 1)))

print('\n size of kspace: ', kspace_test.shape, ', maps: ', sens_maps_testAll.shape, ', mask: ', test_mask.shape)

# %%  zero-padded outer edges of k-space with no signal- check github readme file for explanation for further
# explanations for coronal PD dataset, first 17 and last 16 columns of k-space has no signal in the training mask we
# set corresponding columns as 1 to ensure data consistency
if args.data_opt == 'Coronal_PD':
    test_mask[:, :, 0:17] = np.ones((nSlices, args.nrow_GLOB, 17))
    test_mask[:, :, args.ncol_GLOB-16:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 16))

test_refAll = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
test_inputAll = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

print('\n generating the refs and sense1 input images')
for ii in range(nSlices):
    sub_kspace = kspace_test[ii] * np.tile(test_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    test_refAll[ii] = utils.sense1(kspace_test[ii, ...], sens_maps_testAll[ii, ...])
    test_inputAll[ii] = utils.sense1(sub_kspace, sens_maps_testAll[ii, ...])

sens_maps_testAll = np.transpose(sens_maps_testAll, (0, 3, 1, 2))
all_ref_slices, all_input_slices, all_recon_slices = [], [], []
all_psnr, all_ssim = [], []

print('\n  loading the saved model ...')
tf.compat.v1.reset_default_graph()
loadChkPoint = tf.train.latest_checkpoint(saved_model_dir)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
    new_saver = tf.compat.v1.train.import_meta_graph(saved_model_dir + '/model_test.meta')
    new_saver.restore(sess, loadChkPoint)

    # ..................................................................................................................
    graph = tf.compat.v1.get_default_graph()
    nw_output = graph.get_tensor_by_name('nw_output:0')
    nw_kspace_output = graph.get_tensor_by_name('nw_kspace_output:0')
    mu_param = graph.get_tensor_by_name('mu:0')
    x0_output = graph.get_tensor_by_name('x0:0')
    all_intermediate_outputs = graph.get_tensor_by_name('all_intermediate_outputs:0')

    # ...................................................................................................................
    trn_maskP = graph.get_tensor_by_name('trn_mask:0')
    loss_maskP = graph.get_tensor_by_name('loss_mask:0')
    nw_inputP = graph.get_tensor_by_name('nw_input:0')
    sens_mapsP = graph.get_tensor_by_name('sens_maps:0')
    weights = sess.run(tf.compat.v1.global_variables())

    for ii in tqdm(range(nSlices), desc="Iteration"):

        ref_image_test = np.copy(test_refAll[ii, :, :])[np.newaxis]
        nw_input_test = np.copy(test_inputAll[ii, :, :])[np.newaxis]
        sens_maps_test = np.copy(sens_maps_testAll[ii, :, :, :])[np.newaxis]
        testMask = np.copy(test_mask[ii, :, :])[np.newaxis]
        ref_image_test, nw_input_test = utils.complex2real(ref_image_test), utils.complex2real(nw_input_test)

        tic = time.time()
        dataDict = {nw_inputP: nw_input_test, trn_maskP: testMask, loss_maskP: testMask, sens_mapsP: sens_maps_test}
        nw_output_ssdu, *_ = sess.run([nw_output, nw_kspace_output, x0_output, all_intermediate_outputs, mu_param], feed_dict=dataDict)
        toc = time.time() - tic
        ref_image_test = utils.real2complex(ref_image_test.squeeze())
        nw_input_test = utils.real2complex(nw_input_test.squeeze())
        nw_output_ssdu = utils.real2complex(nw_output_ssdu.squeeze())

        if args.data_opt == 'Coronal_PD':
            """window levelling in presence of fully-sampled data"""
            factor = np.max(np.abs(ref_image_test[:]))
        else:
            factor = 1

        ref_image_test = np.abs(ref_image_test) / factor
        nw_input_test = np.abs(nw_input_test) / factor
        nw_output_ssdu = np.abs(nw_output_ssdu) / factor

        # ...............................................................................................................
        all_recon_slices.append(nw_output_ssdu)
        all_ref_slices.append(ref_image_test)
        all_input_slices.append(nw_input_test)

        all_psnr.append(utils.getPSNR(ref_image_test, nw_output_ssdu))
        all_ssim.append(utils.getSSIM(ref_image_test, nw_output_ssdu))

        # print('\n Iteration: ', ii, 'elapsed time %f seconds' % toc)

plt.figure()
slice_num = 20
plt.subplot(1, 3, 1), plt.imshow(np.abs(all_ref_slices[slice_num]), cmap='gray'), plt.title('ref')
plt.subplot(1, 3, 2), plt.imshow(np.abs(all_input_slices[slice_num]), cmap='gray'), plt.title('input')
plt.subplot(1, 3, 3), plt.imshow(np.abs(all_recon_slices[slice_num]), cmap='gray'), plt.title('recon')
save_plt_fname = "{}_{}.png".format(args.challenge, kspace_dir.split('/')[-1].split('.')[0])
plt.savefig(os.path.join(saved_model_dir, save_plt_fname))
plt.show()
print("PSNR = {}, SSIM = {}".format(all_psnr[slice_num], all_ssim[slice_num]))
