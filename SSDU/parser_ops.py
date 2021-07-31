import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='SSDU: Self-Supervision via Data Undersampling ')

    # %% hyperparameters for the unrolled network
    parser.add_argument('--acc_rate', type=int, default=8,
                        help='acceleration rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='batch size')
    parser.add_argument('--nb_unroll_blocks', type=int, default=10,
                        help='number of unrolled blocks')
    parser.add_argument('--nb_res_blocks', type=int, default=5,
                        help="number of residual blocks in ResNet")
    parser.add_argument('--CG_Iter', type=int, default=10,
                        help='number of Conjugate Gradient iterations for DC')
    parser.add_argument('--num_pool_layers', type=int, default=4,
                        help='number of U-Net pooling layers')

    # %% hyperparameters for the dataset
    parser.add_argument('--data_opt', type=str, default='Coronal_PD',
                        help=' directories for the kspace, sensitivity maps and mask')
    parser.add_argument('--nrow_GLOB', type=int, default=320,
                        help='number of rows of the slices in the dataset')
    parser.add_argument('--ncol_GLOB', type=int, default=320,
                        help='number of columns of the slices in the dataset')
    parser.add_argument('--ncoil_GLOB', type=int, default=15,
                        help='number of coils of the slices in the dataset')
    parser.add_argument('--subsample_mask_type', type=str, default='equispaced',
                        help='type of k-space subsampling mask', choices=['random', 'equispaced'])
    parser.add_argument('--center_fractions', type=float, default=[0.08],
                        help='number of center lines to use in subsampling mask')
    parser.add_argument('--challenge', type=str, default="singlecoil", choices=("singlecoil", "multicoil"),
                        help='which challenge to preprocess for, single or multi coil')

    # %% hyperparameters for the SSDU
    parser.add_argument('--mask_type', type=str, default='Gaussian',
                        help='mask selection for training and loss masks', choices=['Gaussian', 'Uniform'])
    parser.add_argument('--rho', type=float, default=0.4,
                        help='cardinality of the loss mask, \ rho = |\ Lambda| / |\ Omega|')

    return parser
