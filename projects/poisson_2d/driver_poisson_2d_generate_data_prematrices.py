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
from prematrices.construct_prematrix import construct_prematrix
from utils_fenics.construct_boundary_matrices_and_load_vector import\
        construct_boundary_matrices_and_load_vector
from utils_io.load_fem_matrices import load_boundary_matrices_and_load_vector
from utils_misc.positivity_constraints import positivity_constraint_exp,\
                                             positivity_constraint_log_exp
from utils_mesh.observation_points import form_interior_observation_points,\
                                          form_observation_data

# Import project utilities
from utils_project.filepaths import FilePaths
from utils_project.weak_forms import stiffness
from utils_project.solve_poisson_2d import solve_pde_prematrices

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":

    ##################
    #   Setting Up   #
    ##################
    #=== Plotting Options ===#
    use_hippylib_plotting = False
    use_my_plotting = True
    colourbar_limit_parameter = 7
    colourbar_limit_state = 2

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

    #=== Load Prior ===#
    prior_mean, _, prior_covariance_cholesky, _ = load_prior(filepaths, dof_meta)

    #=== Draw Parameters from Prior ===#
    if options.draw_and_save_parameters == 1:
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
                                        (5,5), (0,colourbar_limit_parameter))

    ###################
    #   FEM Objects   #
    ###################
    #=== Construct or Load Prematrices ===#
    if options.construct_and_save_prematrices == 1:
        if not os.path.exists(filepaths.directory_dataset):
            os.makedirs(filepaths.directory_dataset)
        prestiffness = construct_prematrix(options,
                                           fe_space, meta_space,
                                           dof_fe, dof_meta,
                                           stiffness, test=False)
        sparse.save_npz(filepaths.prestiffness + '.npz', prestiffness)
    prestiffness = sparse.load_npz(filepaths.prestiffness + '.npz')

    #=== Construct or Load Boundary Matrix and Load Vector ===#
    if options.construct_and_save_boundary_matrices == 1:
        construct_boundary_matrices_and_load_vector(filepaths,
                fe_space, options.boundary_matrix_constant, options.load_vector_constant)
    boundary_matrix, load_vector = load_boundary_matrices_and_load_vector(filepaths, dof_fe)

    ##########################
    #   Computing Solution   #
    ##########################
    #=== Solve PDE with Prematrices ===#
    state = solve_pde_prematrices(options, filepaths,
                                  parameters,
                                  prestiffness, boundary_matrix, load_vector)

    #=== Plot Solution ===#
    if options.plot_solutions == 1:
        for n in range(0, options.num_data):
            plot_fem_function_fenics_2d(meta_space, state[n,:],
                                        '',
                                        filepaths.directory_figures + 'state_%d.png' %(n),
                                        (5,5), (0,colourbar_limit_state))

    #=== Form Observation Data ===#
    obs_indices, _ = form_interior_observation_points(options, filepaths, meta_space)
    form_observation_data(filepaths, state, obs_indices)
