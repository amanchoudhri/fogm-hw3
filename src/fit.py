"""
Fit a mixed-membership model on the gerbil vocal call data, using
the provided hyperparameters.
"""

import argparse
import logging
import sys
import time
import json

from pathlib import Path

import numpy as np
import arviz as az

from scipy.special import logsumexp
from cmdstanpy import CmdStanModel, CmdStanVB

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

RANDOM_SEED = 2022
rng = np.random.default_rng(RANDOM_SEED)

DEFAULT_DATA_PATH = Path('data') / 'data_splits_projected.npz'
DEFAULT_NUM_COMPONENTS = 5

def hyperparams(K, D, model_type='diagonal_mixture'):
    """
    Return a set of sensible hyperparameters given the number of
    components K and the embedding dimension D.
    """
    if model_type != 'diagonal_mixture':
        raise NotImplementedError('Models other than diagonal mixture not yet available!')
    return {
        'alpha': np.ones(K), # shape param for exchangeable Dirichlet prior on theta
        'mu': np.zeros((K, D)),     # prior means for beta
        'lambda': 5 * np.ones((K)),  # prior variance for beta
        'gamma': 5,  # scale for half-Normal prior on sigma
    }


def load_data(path):
    d = np.load(path)
    X_train, X_test = d['X_train'], d['X_test']
    return X_train, X_test

def get_args():
    # defaults should be the output of default_hyperparams
    parser = argparse.ArgumentParser()

    parser.add_argument('outdir', type=str)
    parser.add_argument('model_name', type=str)
    parser.add_argument('--data', type=str, required=False, default=DEFAULT_DATA_PATH)
    parser.add_argument('--K', type=int, default=DEFAULT_NUM_COMPONENTS)

    parser.add_argument('--num_train', type=int, required=False)
    parser.add_argument('--num_test', type=int, required=False)

    parser.add_argument('--bare', action='store_true')

    return parser.parse_args()

def fit_meanfield(
    X_train,
    X_test,
    K: int,
    model_file='diagonal_mixture.stan',
    ) -> CmdStanVB:
    """
    Create a mixture model with the following hyperparameters:

        K:  number of vocalization types
        alpha: shape param for exchangeable Dirichlet prior on theta
        mu: locations for Gaussian prior on beta
        lambda: shapes for Gaussian prior on beta
        gamma: scale for half-Normal prior on the mixture variances, sigma

    NOTE: Expects X_train to be of shape (num_cohorts, num_observations, embedding_dimension).
    """
    logger.info(f'Creating model on data of shape {X_train.shape}.')

    N = X_train.shape[0]  # number of observations
    D = X_train.shape[1]  # embedding dimension

    N_out = X_test.shape[0]  # number of test num_observations

    model = CmdStanModel(stan_file=model_file)

    data = {
        'K': K,
        'D': D,
        'N': N,
        'X': X_train,
        'N_out': N_out,
        'X_out': X_test,
        **hyperparams(K, D),
    }

    return model.variational(data=data)


if __name__ == '__main__':
    args = get_args()

    outdir = Path(args.outdir)

    if not args.bare:
        outdir = outdir / f"{time.strftime('%Y_%m_%d_%H')}_K{args.K}"

    outdir.mkdir(parents=True, exist_ok=True)

    logger.addHandler(logging.FileHandler(outdir / 'fit.log'))
    logger.info(f'Arguments recieved: {args}')

    X_train, X_test = load_data(path=args.data)
    logger.info(f'Data from path: {args.data}')

    logger.info(f'Running ADVI...')

    # load model
    filename = f'{args.model_name}.stan' if '.stan' not in args.model_name else args.model_name
    models_dir_attempt = Path('models') / filename
    cwd_attempt = Path(filename)
    # check the models subdirectory
    if models_dir_attempt.is_file():
        model_file = str(models_dir_attempt)
    # if not there, check cwd
    elif cwd_attempt.is_file():
        model_file = str(cwd_attempt)
    else:
        raise ValueError(f'Could not file model {args.model_name}!')

    if args.num_train:
        X_train = X_train[:args.num_train]
    if args.num_test:
        X_test = X_test[:args.num_test]

    fit = fit_meanfield(
        X_train,
        X_test,
        K=args.K,
        model_file=model_file
        )

    fit_az = az.from_cmdstan(fit.runset.csv_files)
    netcdf_path = str(outdir / 'results.nc')
    fit_az.to_netcdf(netcdf_path)

    results = {}

    # record the posterior expectation of each parameter
    for param, expectation in fit.stan_variables().items():
        if type(expectation) == np.ndarray:
            results[param] = expectation.tolist()
        else:
            results[param] = expectation

    # record the held-out predictive log-likelihood
    # during sampling, stan calculated log_p(x_out | h) for every
    # posterior sample h. to combine them and form a monte carlo
    # estimate of log p(x_out | x_in), it suffices to take the log_sum_exp
    # and subtract the log of the number of samples
    log_p_out = fit_az['posterior']['log_p_out']
    results['log_p_out'] = float(logsumexp(log_p_out) - np.log(len(log_p_out)))

    results['netcdf_path'] = netcdf_path

    logger.info(f'Results: {results}')
    with open(outdir / 'results.json', 'w+') as f:
        json.dump(results, f)
