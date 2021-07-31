from bart import bart
import numpy as np


def espirit(args, kspace, k, r, t, c):
    """
    Run ESPIRIT coil sensitivity estimation and Total Variation Minimization
    based reconstruction algorithm using the BART toolkit.

    Args:
        args (argparse.Namespace): Arguments including ESPIRiT parameters.

    Returns:
        np.array: Sensitivity maps
    """
    sens_maps = np.ndarray(kspace.shape, dtype=complex)

    for idx in range(kspace.shape[0]):
        kspace_slice = kspace[idx]

        if np.iscomplexobj(kspace_slice):
            kspace_slice = np.stack((kspace_slice.real, kspace_slice.imag), axis=-1)

        kspace_slice = np.expand_dims(kspace_slice, 0)
        kspace_slice = kspace_slice[..., 0] + 1j * kspace_slice[..., 1]

        # estimate sensitivity maps
        sens_map = bart(1, f"ecalib -m1 -d0 -k{k} -r{r} -t{t} -c{c}", kspace_slice)

        sens_map = np.squeeze(sens_map)
        if args.challenge == "singlecoil" or args.ncoil_GLOB == 1:
            sens_map = np.expand_dims(sens_map, -1)

        sens_maps[idx] = sens_map

    return sens_maps
