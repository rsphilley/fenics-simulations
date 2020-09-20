import numpy as np
import pandas as pd
from scipy import linalg, spatial
import os

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_matern_covariance(filepaths, nodes, kern_type, cov_length):

    #=== Distances ===#
    distances = spatial.distance.pdist(nodes)
    distances = spatial.distance.squareform(distances)

    #=== Matern52 ===#
    if kern_type == 'm52':
        tmp = np.sqrt(5) * distances / cov_length
        prior_covariance = (1 + tmp + tmp * tmp / 3) * np.exp(-tmp)

    #=== Matern32 ===#
    if kern_type == 'm32':
        tmp = np.sqrt(3) * distances / cov_length
        prior_covariance = (1 + tmp) * np.exp(-tmp)

    #=== Construct Inverses and Choleskies ===#
    prior_covariance_inverse = np.linalg.inv(prior_covariance)
    machine_eps = np.finfo(float).eps
    diagonal_perturbation = 10**10*machine_eps*np.identity(prior_covariance.shape[0])
    prior_covariance_cholesky = linalg.cholesky(prior_covariance + diagonal_perturbation)
    prior_covariance_cholesky_inverse = np.linalg.inv(
            linalg.cholesky(prior_covariance + diagonal_perturbation))

    #=== Save Prior ===#
    if not os.path.exists(filepaths.directory_dataset):
        os.makedirs(filepaths.directory_dataset)

    mean_vec = np.zeros(1)
    df_prior_mean = pd.DataFrame({'prior_mean': mean_vec})
    df_prior_mean.to_csv(filepaths.prior_mean + '.csv', index=False)

    df_prior_covariance = pd.DataFrame({'prior_covariance': prior_covariance.flatten()})
    df_prior_covariance.to_csv(filepaths.prior_covariance + '.csv', index=False)

    df_prior_covariance = pd.DataFrame({'prior_covariance': prior_covariance.flatten()})
    df_prior_covariance.to_csv(filepaths.prior_covariance + '.csv', index=False)

    df_prior_covariance_cholesky = pd.DataFrame(
            {'prior_covariance_cholesky': prior_covariance_cholesky.flatten()})
    df_prior_covariance_cholesky.to_csv(
            filepaths.prior_covariance_cholesky + '.csv', index=False)

    df_prior_covariance_cholesky_inverse = pd.DataFrame(
            {'prior_covariance_inverse': prior_covariance_cholesky_inverse.flatten()})
    df_prior_covariance_cholesky_inverse.to_csv(
            filepaths.prior_covariance_cholesky_inverse + '.csv',
            index=False)
