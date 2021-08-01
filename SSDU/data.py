import numpy as np
import os
import h5py as h5
import utils
import parser_ops
import masks.ssdu_masks as ssdu_masks
from masks.subsample import create_mask_for_mask_type
from sens_map_gen.bart_espirit import espirit


def slice_coil(kspace, args):
    # kspace shape: (num_slices, h, w, num_coils)
    num_slices, h, w, num_coils = kspace.shape

    # num_coils is <= args.ncoil_GLOB, return kspace_train unchanged
    if num_coils <= args.ncoil_GLOB:
        return kspace

    coil_slices = num_coils // args.ncoil_GLOB
    new_kspace = np.empty((num_slices * coil_slices, h, w, args.ncoil_GLOB), dtype=complex)

    for slice_idx in range(num_slices):
        kspace_slice = kspace[slice_idx]
        for coil_slice_idx in range(coil_slices):
            slice_num = slice_idx * coil_slices + coil_slice_idx
            coil_slice_num = coil_slice_idx * args.ncoil_GLOB
            new_kspace[slice_num, :, :, :] = kspace_slice[:, :, coil_slice_num:coil_slice_num + args.ncoil_GLOB]

    return new_kspace


if __name__ == "__main__":
    parser = parser_ops.get_parser()
    args = parser.parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    print('\n Loading ', args.data_opt, ' data, acc rate : ', args.acc_rate, ', mask type :', args.mask_type)
    kspace_dir, *_ = utils.get_train_directory(args)

    data_dir = "/".join(kspace_dir.split("/")[:-1])
    fnames = os.listdir(data_dir)

    npz_dir = "data/{}".format(data_dir.split("/")[-1])
    if not os.path.exists(npz_dir):
        os.mkdir(npz_dir)

    fnames = ["file1001067.h5"]
    for fname in fnames:
        kspace_dir = os.path.join(data_dir, fname)

        npz_fname = "{}_{}_acc{}.npz".format(args.challenge, kspace_dir.split('/')[-1].split('.')[0], args.acc_rate)

        # fastMRI data -> (num_slices, num_coils, h, w)
        kspace_train = h5.File(kspace_dir, "r")['kspace'][:20]
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
