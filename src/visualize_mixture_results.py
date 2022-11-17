import argparse
import math

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import torch

from matplotlib import pyplot as plt
from ava.models.vae import VAE

from plotting import subplots

RANDOM_SEED = 2022
rng = np.random.default_rng(RANDOM_SEED)

DEFAULT_VAE_CHECKPOINT = Path('aux') / 'vae_checkpoint.tar'
ORIGINAL_SPECTROGRAM_SHAPE = (128, 128)

def load_vae(checkpoint=DEFAULT_VAE_CHECKPOINT) -> VAE:
    vae = VAE(save_dir='vae')
    vae.load_state(checkpoint)
    return vae

def get_args():
    p = argparse.ArgumentParser(description='Visualize results from mixture model.')
    p.add_argument(
        'result_dir',
        help=('Path to directory containing two files, results.nc and'
            'results.json, storing the results from fitting a diagonal mixture'
            'model.')
        )
    p.add_argument(
        '--out_dir',
        help='Optional path where results should be stored—defaults to `result_dir`',
        )
    p.add_argument(
        '--N_samples',
        help='Optional, the number of samples to draw and plot from each cluster. Default: 12',
        default=12,
        )
    p.add_argument(
        '--no_reproject',
        action='store_true',
        help='Optional flag, indicates that the mixture components should NOT be reprojected before applying VAE decoder.'
        )
    return p.parse_args()

def main(result_dir, out_dir=None, N_samples=12, no_reproject=False):
    vae = load_vae()
    PI = np.load('aux/projection_matrix.npy')


    fit_az = az.from_netcdf(Path(result_dir) / 'results.nc')

    get_posterior_expectation = lambda param: np.array(fit_az['posterior'][param].mean(axis=(0, 1)))
    posterior_beta = get_posterior_expectation('beta')
    posterior_sigma = get_posterior_expectation('sigma')
    posterior_theta = get_posterior_expectation('theta')

    out_dir = Path(out_dir) if out_dir else Path(result_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # reproject beta (if supposed to)
    beta = posterior_beta
    if not no_reproject:
        beta = posterior_beta @ PI
    # run thru VAE
    with torch.no_grad():
        cluster_mean_spectrograms = vae.decode(torch.tensor(beta, dtype=torch.float))
    # and plot
    fig, axs = subplots(len(cluster_mean_spectrograms))
    for s, ax in zip(cluster_mean_spectrograms, axs):
        ax.imshow(s.reshape(*ORIGINAL_SPECTROGRAM_SHAPE))
    fig.suptitle(f'Reconstructed means from each cluster, K = {len(cluster_mean_spectrograms)}')

    plt.savefig(out_dir / 'cluster_means.png')

    # draw N_samples from each cluster and decode back to spectrograms
    fig, axs = plt.subplots(len(cluster_mean_spectrograms), N_samples, figsize=(24, 24))
    for i, (ax_row, beta_value, sigma_value) in enumerate(zip(axs, posterior_beta, posterior_sigma)):
        # draw N samples from each cluster distribution
        samples = rng.multivariate_normal(beta_value, np.diag(sigma_value), size=N_samples)
        # reproject the samples (if supposed to)
        if not no_reproject:
            samples = samples @ PI
        # reconstruct the spectrograms
        with torch.no_grad():
            sample_spectrograms = vae.decode(torch.tensor(samples, dtype=torch.float))
        # and plot them
        for j, (ax, s) in enumerate(zip(ax_row, sample_spectrograms)):
            ax.imshow(s.reshape(*ORIGINAL_SPECTROGRAM_SHAPE))
            if j == 0:
                ax.set_ylabel(f'Cluster {i} (proportion: {posterior_theta[i]:0.4f})', rotation=0, labelpad=70)

    fig.tight_layout()

    plt.savefig(out_dir / 'cluster_samples.png')
if __name__ == '__main__':
    args = get_args()

    main(args.result_dir, args.out_dir, args.N_samples, args.no_reproject)
