import os
import re
from os.path import join

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from nilearn.input_data import NiftiMasker
from sklearn.utils import gen_batches

from cogspaces.datasets.contrasts import fetch_all
from cogspaces.datasets.dictionaries import fetch_atlas_modl
from cogspaces.datasets.utils import get_data_dir, fetch_mask

idx = pd.IndexSlice


def init_fetch_mask() -> str:
    """For mask bootstrapping"""
    return join(get_data_dir(), 'mask', 'hcp_mask.nii.gz')


def single_mask(masker, imgs):
    return masker.transform(imgs)


def single_reduce(components, data, lstsq=False):
    if not lstsq:
        return data.dot(components.T)
    else:
        X, _, _, _ = np.linalg.lstsq(components.T, data.T)
        return X.T


def mask_all(output_dir: str or None, n_jobs: int=1, mask: str='hcp'):
    batch_size = 10

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = fetch_all()
    mask = fetch_mask()[mask]
    masker = NiftiMasker(smoothing_fwhm=4, mask_img=mask,
                         verbose=0, memory_level=1, memory=None).fit()

    for study, this_data in data.groupby('study'):
        imgs = this_data['z_map'].values
        targets = this_data.reset_index()[['study', 'subject', 'contrast']]

        n_samples = this_data.shape[0]
        batches = list(gen_batches(n_samples, batch_size))
        this_data = Parallel(n_jobs=n_jobs, verbose=10,
                             backend='multiprocessing', mmap_mode='r')(
            delayed(single_mask)(masker, imgs[batch]) for batch in batches)
        this_data = np.concatenate(this_data, axis=0)

        dump((this_data, targets), join(output_dir, 'data_%s.pt' % study))


def reduce_all(masked_dir, output_dir, n_jobs=1, lstsq=False,
               mask: str = 'icbm_gm'):
    batch_size = 200

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    modl_atlas = fetch_atlas_modl()
    mask = fetch_mask()[mask]
    dictionary = modl_atlas['components512']
    masker = NiftiMasker(mask_img=mask).fit()
    components = masker.transform(dictionary)

    expr = re.compile("data_(.*).pt")

    for file in os.listdir(masked_dir):
        match = re.match(expr, file)
        if match:
            study = match.group(1)
            this_data, targets = load(join(masked_dir, file))
            n_samples = this_data.shape[0]
            batches = list(gen_batches(n_samples, batch_size))
            this_data = Parallel(n_jobs=n_jobs, verbose=10,
                                 backend='multiprocessing', mmap_mode='r')(
                delayed(single_reduce)(components,
                                       this_data[batch], lstsq=lstsq)
                for batch in batches)
            this_data = np.concatenate(this_data, axis=0)

            dump((this_data, targets), join(output_dir,
                                            'data_%s.pt' % study))


def main(mask: str='hcp'):
    masked_dir = join(get_data_dir(), 'masked_%s' % mask)
    reduced_dir = join(get_data_dir(), 'reduced_512_%s' % mask)
    mask_all(output_dir=masked_dir, n_jobs=30, mask=mask)
    reduce_all(output_dir=reduced_dir, masked_dir=masked_dir,
               n_jobs=30, mask=mask)
    # Data can now be loaded using `cogspaces.utils.data.load_masked_data`


if __name__ == '__main__':
    main()
