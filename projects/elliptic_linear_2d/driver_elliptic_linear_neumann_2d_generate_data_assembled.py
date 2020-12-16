#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 21:41:12 2020
@author: hwan
"""
import sys
import os
sys.path.insert(0, os.path.realpath('../../src'))
sys.path.append('../')

import yaml
from attrdict import AttrDict
import scipy.sparse as sparse

# Import src code
from utils_mesh.construct_mesh_rectangular import construct_mesh
from utils_mesh.plot_mesh import plot_mesh
from utils_prior.bilaplacian_prior import construct_bilaplacian_prior
from utils_prior.smoothness_prior_autocorr import smoothness_prior_autocorr
from utils_prior.gaussian_field import construct_matern_covariance
from utils_io.load_prior import load_prior
from utils_prior.draw_from_distribution import draw_from_distribution
from utils_io.load_parameters import load_parameters
from utils_fenics.plot_fem_function_fenics_2d import plot_fem_function_fenics_2d
from utils_io.load_fem_matrices import load_boundary_matrices_and_load_vector
from utils_misc.positivity_constraints import positivity_constraint_log_exp
from utils_mesh.observation_points import form_interior_observation_points,\
                                          form_observation_data

# Import project utilities
from utils_project.filepaths import FilePaths
from utils_project.construct_system_matrices_elliptic_linear_neumann_2d import\
        construct_system_matrices, load_system_matrices
from utils_project.solve_elliptic_linear_neumann_2d_assembled import solve_pde_assembled

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":

    ##################
    #   Setting Up   #
    ##################
    #=== Plotting Options ===#
    colourbar_limit_min_parameter = -4
    colourbar_limit_max_parameter = 4
    colourbar_limit_min_state = -10
    colourbar_limit_max_state = 10

    #=== Options ===#
    with open('config_files/options.yaml') as f:
        options = yaml.safe_load(f)
    options = AttrDict(options)
    options.num_nodes = (options.nx + 1) * (options.ny + 1)

    #=== File Paths ===#
    filepaths = FilePaths(options)

    ############
    #   Mesh   #
    ############
    #=== Construct Mesh ===#
    fe_space, meta_space,\
    nodes, dof_fe, dof_meta = construct_mesh(options)

    #=== Plot Mesh ===#
    if options.plot_mesh == True:
        mesh = fe_space.mesh()
        plot_mesh(filepaths,
                  (5,5), '',
                  (-1,1), (-1,1),
                  mesh.coordinates(), mesh.cells())

    ############################
    #   Prior and Parameters   #
    ############################
    #=== Construct Prior ===#
    if options.construct_prior == 1:
        if options.prior_type_blp == 1:
            prior = construct_bilaplacian_prior(filepaths,
                                                meta_space, options.prior_mean_blp,
                                                options.prior_gamma_blp,
                                                options.prior_delta_blp)
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

    #=== Draw Parameters from Prior ===#
    if options.draw_and_save_parameters == 1:
        prior_mean, _, prior_covariance_cholesky, _ = load_prior(filepaths, dof_meta)
        draw_from_distribution(filepaths,
                               prior_mean, prior_covariance_cholesky, dof_meta,
                               positivity_constraint_log_exp, 0.5,
                               num_samples = options.num_data)

    #=== Load Parameters ===#
    parameters = load_parameters(filepaths, dof_meta, options.num_data)

    #=== Plot Parameters ===#
    if options.plot_parameters == 1:
        for n in range(0, options.num_data):
            plot_fem_function_fenics_2d(meta_space, parameters[n,:],
                                        '',
                                        filepaths.directory_figures + 'parameter_%d.png' %(n),
                                        (5,5),
                                        (colourbar_limit_min_parameter,
                                         colourbar_limit_max_parameter))

    ###################
    #   FEM Objects   #
    ###################
    #=== Construct or Load Matrices ===#
    if options.construct_and_save_matrices == 1:
        if not os.path.exists(filepaths.directory_dataset):
            os.makedirs(filepaths.directory_dataset)
        construct_system_matrices(filepaths, meta_space)
    stiffness_matrix, mass_matrix = load_system_matrices(options, filepaths)

    ##########################
    #   Computing Solution   #
    ##########################
    #=== Solve PDE with Prematrices ===#
    state = solve_pde_assembled(options, filepaths,
                                parameters,
                                stiffness_matrix, mass_matrix)

    #=== Plot Solution ===#
    if options.plot_solutions == 1:
        for n in range(0, options.num_data):
            plot_fem_function_fenics_2d(meta_space, state[n,:],
                                        '',
                                        filepaths.directory_figures + 'state_%d.png' %(n),
                                        (5,5),
                                        (colourbar_limit_min_state,colourbar_limit_max_state))

    #=== Form Observation Data ===#
    obs_indices, _ = form_interior_observation_points(options, filepaths, meta_space)
    form_observation_data(filepaths, state, obs_indices)
