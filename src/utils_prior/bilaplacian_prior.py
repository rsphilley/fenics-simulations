import numpy as np

from hippylib import *
import dolfin as dl

from utils_io.prior import save_prior
from utils_fenics.convert_array_to_dolfin_function import convert_array_to_dolfin_function

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_bilaplacian_prior(filepaths,
                                Vh, mean,
                                gamma, delta):

    #=== Fenics Mean Vector ===#
    mean_array = mean*np.ones(Vh.dim())
    mean_dl = convert_array_to_dolfin_function(Vh, mean_array)
    mean_dl_vec = mean_dl.vector()

    #=== Construct Prior ===#
    prior = BiLaplacianPrior(Vh,
                             gamma, delta,
                             mean = mean_dl_vec,
                             robin_bc=True)

    #=== Discretized Forms ===#
    mean_vec = mean*np.ones(Vh.dim())

    inv_sqrt_M = np.linalg.inv(np.linalg.cholesky(prior.M.array()))
    inv_L = np.matmul(inv_sqrt_M, prior.A.array())

    L = np.linalg.inv(inv_L)
    cov = np.matmul(L, L.T)
    inv_cov = np.matmul(inv_L, inv_L.T)

    #=== Save Prior ===#
    save_prior(filepaths, mean_vec, cov, inv_cov, L, inv_L)

    return prior
