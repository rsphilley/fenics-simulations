import sys
import os
sys.path.insert(0, os.path.realpath('../../src'))
sys.path.append('../')

import yaml
from attrdict import AttrDict
import scipy.sparse as sparse

# Import src code
from utils_mesh.construct_mesh_2d import construct_mesh
from utils_prior.smoothness_prior_autocorr import smoothness_prior_autocorr
from utils_prior.gaussian_field import construct_matern_covariance
from utils_io.load_prior import load_prior
from utils_prior.draw_from_distribution import draw_from_distribution
from utils_io.load_parameters import load_parameters
from utils_fenics.plot_fem_function_fenics_2d import plot_fem_function_fenics_2d
from prematrices.construct_prematrix import construct_prematrix
from utils_io.load_fem_matrices import load_prematrices, load_boundary_matrices_and_load_vector

# Import project utilities
from utils_project.filepaths import FilePaths
from utils_project.weak_forms import stiffness
# from utils_project.solve_poisson_2D import solve_PDE_prematrices

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":

    ##################
    #   Setting Up   #
    ##################
    #=== Options ===#
    with open('config_files/options.yaml') as f:
        options = yaml.safe_load(f)
    options = AttrDict(options)
    options.num_nodes = (options.num_nodes_x + 1) * (options.num_nodes_y + 1)

    #=== File Paths ===#
    filepaths = FilePaths(options)

    ############
    #   Mesh   #
    ############
    #=== Construct Mesh ===#
    fe_space, meta_space,\
    u, v, sigma,\
    nodes, dof_fe, dof_meta = construct_mesh(options)

    ############################
    #   Prior and Parameters   #
    ############################
    #=== Construct Prior ===#
    if options.construct_prior == 1:
        if options.prior_type_AC == 1:
            smoothness_prior_autocorr(filepaths,
                    nodes,
                    options.prior_mean_AC,
                    options.prior_variance_AC,
                    options.prior_corr_AC)
        if options.prior_type_matern == 1:
            construct_matern_covariance(filepaths,
                    nodes,
                    options.prior_kern_type,
                    options.prior_cov_length)

    #=== Load Prior ===#
    prior_mean, _, prior_covariance_cholesky, _ = load_prior(filepaths, dof_meta)

    #=== Draw Parameters from Prior ===#
    if options.draw_and_save_parameters == 1:
        draw_from_distribution(filepaths,
                               prior_mean, prior_covariance_cholesky, dof_meta,
                               num_samples = options.num_data)

    #=== Load Parameters ===#
    parameters = load_parameters(filepaths, dof_meta, options.num_data)

    #=== Plot Parameters ===#
    if options.plot_parameters == 1:
        for n in range(0, options.num_data):
            plot_fem_function_fenics_2d(meta_space, parameters[n,:],
                                        'parameter',
                                        filepaths.directory_figures + 'parameter_%d.png' %(n),
                                        (5,5))

    ####################
    ##   FEM Objects   #
    ####################
    ##=== Construct or Load Prematrices ===#
    #if options.construct_and_save_prematrices == 1:
    #    if not os.path.exists(filepaths.directory_dataset):
    #        os.makedirs(filepaths.directory_dataset)
    #    prestiffness = construct_prematrix(options, stiffness, test=False)
    #    sparse.save_npz('prestiffness.npz', prestiffness)
    #premass, prestiffness = load_prematrices(filepaths, dof_meta)

    ##=== Construct or Load Boundary Matrix and Load Vector ===#
    #if options.construct_and_save_boundary_matrices == 1:
    #    construct_boundary_matrices_and_load_vector(filepaths,
    #            nodes, elements, boundary_indices,
    #            options.boundary_matrix_constant, options.load_vector_constant)
    #boundary_matrix, load_vector = load_boundary_matrices_and_load_vector(filepaths, dof_meta)
    #boundary_matrix = options.boundary_matrix_constant*boundary_matrix
    #load_vector = -options.load_vector_constant*load_vector

    ###########################
    ##   Computing Solution   #
    ###########################
    ##=== Solve PDE with Prematrices ===#
    #premass, prestiffness = load_prematrices(filepaths, dof_meta)
    #solution = solve_PDE_prematrices(options, filepaths,
    #                                 parameters,
    #                                 prestiffness, boundary_matrix, load_vector)

    ##=== Form Observation Data ===#
    #boundary_indices_no_bottom = reduce(np.union1d,
    #        (boundary_indices_left, boundary_indices_right, boundary_indices_top))
    #form_observation_data(options, filepaths, boundary_indices_no_bottom)

    ##=== Plot Mesh with Observation Points ===#
    #plot_mesh(filepaths, 'mesh', nodes, elements)

    ##=== Plot Solution ==#
    #if options.plot_solutions == 1:
    #    for n in range(0, options.num_data):
    #        plot_fem_function(filepaths.directory_figures, 'solution',
    #                          nodes, elements,
    #                          solution[n,:], sample_number = n)
