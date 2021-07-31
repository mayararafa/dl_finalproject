import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import os
import h5py as h5
import utils
import tf_utils
import parser_ops
import masks.ssdu_masks as ssdu_masks
from masks.subsample import create_mask_for_mask_type
from sens_map_gen.bart_espirit import espirit
import UnrollNet
from data import slice_coil
from tqdm import tqdm

parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

save_dir = 'saved_models'
directory = os.path.join(save_dir,
                         'MY_SSDU_' + args.data_opt + '_' + str(args.epochs) + 'Epochs_Rate' + str(args.acc_rate) +
                         '_' + str(args.nb_unroll_blocks) + 'Unrolls_' + args.mask_type + 'Selection_' +
                         args.reg_model + "Model_" + str(args.nrow_GLOB) + "x" + str(args.ncol_GLOB) + "InputSize")

if not os.path.exists(directory):
    os.makedirs(directory)

print('\n create a test model for the testing')
test_graph_generator = tf_utils.test_graph(directory)

# ...........................................................................d....
start_time = time.time()
print('.................SSDU Training.....................')
tf.compat.v1.reset_default_graph()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# .......................Load the Data..........................................
print('\n Loading ', args.data_opt, ' data, acc rate : ', args.acc_rate, ', mask type :', args.mask_type)
kspace_dir, coil_dir, mask_dir = utils.get_train_directory(args)

npz_dir = "/home/mayararafa/dl_finalproject/SSDU/data"
if not os.path.exists(npz_dir):
    os.mkdir(npz_dir)
npz_fname = "{}_{}.npz".format(args.challenge, kspace_dir.split('/')[-1].split('.')[0])

# Read in data
if os.path.exists(os.path.join(npz_dir, npz_fname)):
    print("\n Loading data")
    data = np.load(os.path.join(npz_dir, npz_fname))
    sens_maps = data["sens_maps"]
    ref_kspace = data["ref_kspace"]
    nw_input = data["nw_input"]
    trn_mask = data["trn_mask"]
    loss_mask = data["loss_mask"]
else:
    # fastMRI data -> (num_slices, num_coils, h, w)
    kspace_train = h5.File(kspace_dir, "r")['kspace'][:1, :]
    if args.challenge == "singlecoil":
        kspace_train = np.resize(kspace_train, (kspace_train.shape[0],) + (1,) + kspace_train.shape[1:])
        args.ncoil_GLOB = 1

    # Crop kspace (h, w) to (args.nrow_GLOB, args.ncol_GLOB)
    crop_h = (kspace_train.shape[2] - args.nrow_GLOB) // 2
    crop_w = (kspace_train.shape[3] - args.ncol_GLOB) // 2
    if crop_h != 0 and crop_w != 0:
        kspace_train = kspace_train[:, :, crop_h:-crop_h, crop_w:-crop_w]
    elif crop_w == 0:
        kspace_train = kspace_train[:, :, crop_h:-crop_h, :]
    elif crop_h == 0:
        kspace_train = kspace_train[:, :, :, crop_w:-crop_w]

    # Reshaped to (num_slices, h, w, num_coils)
    kspace_train = np.transpose(kspace_train, (0, 2, 3, 1))

    # Slice coil data
    kspace_train = slice_coil(kspace_train, args)

    print("\nGenerating sensitivity maps - num_slices={} ...".format(kspace_train.shape[0]))
    sens_maps = espirit(args, kspace_train, 6, 24, 0.02, 0.95)

    original_mask_func = create_mask_for_mask_type(args.subsample_mask_type, args.center_fractions, [args.acc_rate])
    original_mask = original_mask_func(kspace_train.shape[-3:-1], seed=42)  # (h, w)

    print('\n Normalize the kspace to 0-1 region')
    for ii in range(np.shape(kspace_train)[0]):
        kspace_train[ii, :, :, :] = kspace_train[ii, :, :, :] / np.max(np.abs(kspace_train[ii, :, :, :][:]))

    print('\n size of kspace: ', kspace_train.shape, ', maps: ', sens_maps.shape, ', mask: ', original_mask.shape)

    nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB = kspace_train.shape
    trn_mask, loss_mask = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                          np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

    nw_input = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
    ref_kspace = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)

    print('\n create training and loss masks and generate network inputs... ')
    ssdu_masker = ssdu_masks.ssdu_masks()
    for ii in range(nSlices):
        print('\n Iteration: ', ii)

        if args.mask_type == 'Gaussian':
            trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.Gaussian_selection(kspace_train[ii], original_mask,
                                                                                   num_iter=ii)

        elif args.mask_type == 'Uniform':
            trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.uniform_selection(kspace_train[ii], original_mask,
                                                                                  num_iter=ii)

        else:
            raise ValueError('Invalid mask selection')

        sub_kspace = kspace_train[ii] * np.tile(trn_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
        ref_kspace[ii, ...] = kspace_train[ii] * np.tile(loss_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
        nw_input[ii, ...] = utils.sense1(sub_kspace, sens_maps[ii, ...])

    # %%  zero-padded outer edges of k-space with no signal- check github readme file for explanation for further
    # explanations for coronal PD dataset, first 17 and last 16 columns of k-space has no signal in the training mask we
    # set corresponding columns as 1 to ensure data consistency
    if args.data_opt == 'Coronal_PD':
        trn_mask[:, :, 0:17] = np.ones((nSlices, args.nrow_GLOB, 17))
        trn_mask[:, :, args.ncol_GLOB - 16:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 16))

    # %% Prepare the data for the training
    sens_maps = np.transpose(sens_maps, (0, 3, 1, 2))
    ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))
    nw_input = utils.complex2real(nw_input)

    print("\n Dumping data to", os.path.join(npz_dir, npz_fname))
    np.savez(os.path.join(npz_dir, npz_fname), sens_maps=sens_maps, ref_kspace=ref_kspace, nw_input=nw_input,
             trn_mask=trn_mask, loss_mask=loss_mask)

print('\n size of ref kspace: ', ref_kspace.shape, ', nw_input: ', nw_input.shape, ', maps: ', sens_maps.shape,
      ', mask: ', trn_mask.shape)

# %% set the batch size
total_batch = int(np.floor(np.float32(nw_input.shape[0]) / args.batchSize))
kspaceP = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None, None, 2), name='refkspace')
sens_mapsP = tf.compat.v1.placeholder(tf.complex64, shape=(None, None, None, None), name='sens_maps')
trn_maskP = tf.compat.v1.placeholder(tf.complex64, shape=(None, None, None), name='trn_mask')
loss_maskP = tf.compat.v1.placeholder(tf.complex64, shape=(None, None, None), name='loss_mask')
nw_inputP = tf.compat.v1.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, 2), name='nw_input')

# %% creating the dataset
dataset = tf.compat.v1.data.Dataset.from_tensor_slices((kspaceP, nw_inputP, sens_mapsP, trn_maskP, loss_maskP))
dataset = dataset.shuffle(buffer_size=10 * args.batchSize)
dataset = dataset.batch(args.batchSize)
dataset = dataset.prefetch(args.batchSize)
iterator = dataset.make_initializable_iterator()
ref_kspace_tensor, nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor = iterator.get_next('getNext')

# %% make training model
nw_output_img, nw_output_kspace, *_ = UnrollNet.UnrolledNet(nw_input_tensor, sens_maps_tensor, trn_mask_tensor,
                                                            loss_mask_tensor).model
scalar = tf.constant(0.5, dtype=tf.float32)
loss = tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace) / tf.norm(ref_kspace_tensor)) + \
       tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(ref_kspace_tensor, ord=1))

all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.compat.v1.trainable_variables()])
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

saver = tf.compat.v1.train.Saver(max_to_keep=100)
sess_trn_filename = os.path.join(directory, 'model')
totalLoss = []
avg_cost = 0
with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print('SSDU Parameters: Epochs: ', args.epochs, ', Batch Size:', args.batchSize,
          ', Number of trainable parameters: ', sess.run(all_trainable_vars))
    feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: trn_mask, loss_maskP: loss_mask,
                sens_mapsP: sens_maps}

    print('Training...')
    for ep in range(1, args.epochs + 1):
        sess.run(iterator.initializer, feed_dict=feedDict)
        avg_cost = 0
        tic = time.time()
        try:

            for jj in tqdm(range(total_batch), desc="Batch"):
                tmp, _, _ = sess.run([loss, update_ops, optimizer])
                avg_cost += tmp / total_batch
            toc = time.time() - tic
            totalLoss.append(avg_cost)
            print("Epoch:", ep, "elapsed_time = ""{:f}".format(toc), "cost =", "{:.3f}".format(avg_cost))

        except tf.errors.OutOfRangeError:
            pass

        if np.mod(ep, 10) == 0:
            saver.save(sess, sess_trn_filename, global_step=ep)
            sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})

end_time = time.time()
sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})
print('Training completed in  ', ((end_time - start_time) / 60), ' minutes')
plt.scatter(np.linspace(1, args.epochs, args.epochs), totalLoss)
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.savefig(os.path.join(directory, "train_loss_plot.png"))
plt.show()
