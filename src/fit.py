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
from typing import Union

import numpy as np
import arviz as az

from scipy.special import logsumexp
from cmdstanpy import CmdStanModel, CmdStanVB

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

RANDOM_SEED = 2022
rng = np.random.default_rng(RANDOM_SEED)

DEFAULT_DATA_PATH = Path('data') / 'concatenated.npz'
DEFAULT_NUM_COMPONENTS = 5

def hyperparams(K, D) -> dict[str, Union[np.ndarray, int]]:
    """
    Return a set of sensible hyperparameters given the number of
    components K and the embedding dimension D.

    Specifically, return a dict with the following entries:
        alpha:   shape param for exchangeable Dirichlet prior on theta
        mu:      prior means for beta
        lambda:  prior variance for beta
        gamma:   scale for half-Normal prior on sigma
    """
    return {
        'alpha': np.ones(K), # shape param for exchangeable Dirichlet prior on theta
        'mu': np.zeros((K, D)),     # prior means for beta
        'lambda': 5 * np.ones((K)),  # prior variance for beta
        'gamma': 5,  # scale for half-Normal prior on sigma
    }


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
    data: dict[str, np.ndarray],
    K: int,
    model_file='diagonal_mixture.stan',
    num_train=None,
    num_test=None
    ) -> CmdStanVB:
    """
    Create a mixture or mixed-membership model with the following hyperparameters:

        K:  number of vocalization types
        alpha: shape param for exchangeable Dirichlet prior on theta
        mu: locations for Gaussian prior on beta
        lambda: shapes for Gaussian prior on beta
        gamma: scale for half-Normal prior on the mixture variances, sigma

    NOTE: Expects X_train to be of shape (num_observations, embedding_dimension).
    """
    X_train, X_test = data['X_train'], data['X_test']

    logger.info(
        f'Creating model on input data of shape {X_train.shape} '
        f'with held-out data of shape {X_test.shape}'
        )

    # restrict the number of training and test observations if necessary
    if num_train:
        X_train = X_train[:num_train]
        logger.info(f'Truncating train vocalizations to {num_train}')
    if num_test:
        X_test = X_test[:num_test]
        logger.info(f'Truncating test vocalizations to {num_test}')

    N = X_train.shape[0]  # number of observations
    D = X_train.shape[1]  # embedding dimension

    N_out = X_test.shape[0]  # number of test num_observations

    model = CmdStanModel(stan_file=model_file)

    model_params = {
        'K': K,
        'D': D,
        'N': N,
        'X': X_train,
        'N_out': N_out,
        'X_out': X_test,
        **hyperparams(K, D)
    }

    if 'mixed_membership' in model_file:
        og_train_lens, og_test_lens = data['train_lens'], data['test_lens']
        logger.info('Building mixed-membership model.')
        
        logger.info(
            f'Original number of vox per family in train set: {og_train_lens}. '
            f'Original # in test set: {og_test_lens}'
            )

        # adjust train and test lengths if user truncated data
        def __get_lens(N_vocalizations, og_lens) -> list[int]:
            lens = []
            total = 0
            for l in og_lens:
                if total + l <= N_vocalizations:
                    lens.append(l)
                    total += l
                elif total + l > N_vocalizations:
                    num_able_to_include = max(N_vocalizations - total, 0)
                    lens.append(num_able_to_include)
                    break
            return lens

        train_lens = __get_lens(N, og_train_lens)
        test_lens = __get_lens(N_out, og_test_lens)

        logger.info(
            f'After optional truncations, number of vox per family in train set: {train_lens}. '
            f'In test set: {test_lens}'
            )

        # make sure the same number of families are in both sets
        if len(train_lens) != len(test_lens):
            raise ValueError(
                f'Data truncations resulted in differing numbers of families '
                f'in the train ({len(train_lens)}) and test ({len(test_lens)}) sets.'
                )

        model_params['lens'] = train_lens
        model_params['lens_out'] = test_lens
        model_params['F'] = len(train_lens) # number of families
    else:
        logger.info('Building mixture model.')

    logger.info(f'Running ADVI...')
    return model.variational(data=model_params)


if __name__ == '__main__':
    args = get_args()

    outdir = Path(args.outdir)

    if not args.bare:
        outdir = outdir / f"{time.strftime('%Y_%m_%d_%H')}_K{args.K}"

    outdir.mkdir(parents=True, exist_ok=True)

    logger.addHandler(logging.FileHandler(outdir / 'fit.log'))
    logger.info(f'Arguments recieved: {args}')

    # load the data
    logger.info(f'Data from path: {args.data}')
    data = np.load(args.data)


    # load model
    filename = f'{args.model_name}.stan' if '.stan' not in args.model_name else args.model_name
    logger.info(f'Attempting to locate model architecture with filename: {filename}')
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


    fit = fit_meanfield(
        data,
        K=args.K,
        model_file=model_file,
        num_train=args.num_train,
        num_test=args.num_test,
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
