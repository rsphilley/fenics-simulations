import numpy as np

from hippylib import *
import dolfin as dl

# Import src code
from utils_fenics.convert_array_to_dolfin_function import convert_array_to_dolfin_function
from utils_prior.save_prior import save_prior

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_laplacian_prior(filepaths,
                              Vh, mean,
                              gamma, delta):

    #=== Fenics Mean Vector ===#
    mean_array = mean*np.ones(Vh.dim())
    mean_dl = convert_array_to_dolfin_function(Vh, mean_array)
    mean_dl_vec = mean_dl.vector()

    #=== Construct Prior ===#
    prior = LaplacianPrior(Vh,
                           gamma, delta,
                           mean = mean_dl_vec)

    #=== Discretized Forms ===#
    mean_vec = mean*np.ones(Vh.dim())

    inv_sqrt_C = np.linalg.cholesky(prior.R.array())

    inv_sqrt_M = np.linalg.inv(np.linalg.cholesky(prior.M.array()))
    inv_L = np.matmul(inv_sqrt_M, inv_sqrt_C)
    L = np.linalg.inv(inv_L)
    cov = np.matmul(L, L)

    #=== Save Prior ===#
    save_prior(filepaths, mean_vec, cov, inv_L, L)

    return prior
