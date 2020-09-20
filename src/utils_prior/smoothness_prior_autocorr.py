import os

import numpy as np
import pandas as pd
from utils_prior.check_symmetry_and_positive_definite import check_symmetry_and_positive_definite

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

"""
SmoothnessPrior_AutoCorr constructs the blocks needed to form the
matrix square root of the covariance matrix.

Inputs:
    nodes - Coordinates of nodes
    mean - Expected Value
    var - Variance
    corr - Correlation Length
    plot_flag - to plot or not to plot, that is the question

Outputs:
    cov - covariance
    trace_cov - Trace of covariance matrix
    L - Matrix square root of the covariance matrix
    inv_L - Inverse of matrix square root of the covariance matrix
    samples - num_nodes by num_samples matrix where each column is a drawn sample

Hwan Goh 01/03/2018, University of Auckland, New Zealand
Hwan Goh 01/07/2020, Transcribed from MATLAB to Python
"""

def smoothness_prior_autocorr(filepaths,
                              nodes,
                              mean, var, corr):
    num_nodes = len(nodes) #No. of nodes
    mean_vec = mean*np.ones(num_nodes)
    cov = np.zeros((num_nodes,num_nodes))

    #=== Covariance Matrix ===#
    for ii in range(0, num_nodes):
        for jj in range(0, num_nodes):
            cov[ii,jj] = var*np.exp(-(np.linalg.norm(nodes[ii,:] - nodes[jj,:],2))**2/(2*corr**2))
            cov[jj,ii] = cov[ii,jj]
    trace_cov = np.trace(cov)

    #=== Normalizing Covariance ===#
    # if Normalize == 1
    #     cov_diag_inv = sparse(diag(diag(1./cov).^(1/2)))
    #     cov = cov_diag_inv*cov*cov_diag_inv
    #     std_diag = sqrt(var)*speye(num_nodes)
    #     cov = std_diag*cov*std_diag

    #=== Cholesky of Covariance ===#
    epsilon = np.finfo(float).eps
    cov += 10**10*epsilon*np.identity(num_nodes)
    L = np.linalg.cholesky(cov)
    inv_L = np.linalg.inv(L)

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
