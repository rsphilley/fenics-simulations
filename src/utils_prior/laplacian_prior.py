import os

import numpy as np
import pandas as pd

from hippylib import *
import dolfin as dl

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_laplacian_prior(filepaths,
                              Vh, mean,
                              gamma, delta):

    #=== Construct Prior ===#
    prior = LaplacianPrior(Vh,
                           gamma, delta)

    #=== Discretized Forms ===#
    mean_vec = mean*np.ones(Vh.dim())

    inv_sqrtM = np.linalg.inv(np.linalg.cholesky(prior.M.array()))
    inv_L = np.matmul(inv_sqrtM, prior.R.array())
    L = np.linalg.inv(inv_L)
    cov = np.matmul(L, L)

    #=== Save Prior ===#
    if not os.path.exists(filepaths.directory_dataset):
        os.makedirs(filepaths.directory_dataset)

    df_prior_mean = pd.DataFrame({'prior_mean': mean_vec.flatten()})
    df_prior_mean.to_csv(filepaths.prior_mean + '.csv', index=False)

    df_prior_covariance = pd.DataFrame({'prior_covariance': cov.flatten()})
    df_prior_covariance.to_csv(filepaths.prior_covariance + '.csv', index=False)

    df_prior_covariance_cholesky = pd.DataFrame({'prior_covariance_cholesky': L.flatten()})
    df_prior_covariance_cholesky.to_csv(
            filepaths.prior_covariance_cholesky + '.csv', index=False)

    df_prior_covariance_cholesky_inverse = pd.DataFrame(
            {'prior_covariance_inverse': inv_L.flatten()})
    df_prior_covariance_cholesky_inverse.to_csv(
            filepaths.prior_covariance_cholesky_inverse + '.csv',
            index=False)

    print('Prior constructed and saved')

    return prior
