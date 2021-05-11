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

import numpy as np
import pandas as pd

import yaml
from attrdict import AttrDict

# Import src code
from utils_mesh.construct_mesh_1d import construct_mesh
from utils_prior.laplacian_prior import construct_laplacian_prior
from utils_io.load_prior import load_prior
from utils_prior.draw_from_distribution import draw_from_distribution
from utils_prior.draw_from_distribution_fenics import draw_from_distribution_fenics
from utils_fenics.apply_mass_matrix import apply_mass_matrix
from utils_io.load_parameters import load_parameters
from utils_fenics.plot_fem_function_fenics_1d import plot_fem_function_fenics_1d
from utils_fenics.construct_system_matrices_screened_poisson_linear import\
        construct_system_matrices, load_system_matrices
from utils_misc.positivity_constraints import positivity_constraint_identity
from utils_mesh.observation_points import form_interior_observation_points,\
                                          form_observation_data

# Import project utilities
from utils_project.filepaths import FilePaths
from utils_project.solve_linear_neumann_assembled import solve_pde_assembled

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":

    ##################
    #   Setting Up   #
    ##################
    #=== Plotting Options ===#
    limit_min_parameter = 0
    limit_max_parameter = 7
    limit_min_state = 0
    limit_max_state = 10

    #=== Options ===#
    with open('config_files/options.yaml') as f:
        options = yaml.safe_load(f)
    options = AttrDict(options)
    options.num_nodes = options.nx + 1

    #=== File Paths ===#
    filepaths = FilePaths(options)

    #=== Creating Directory ===#
    if not os.path.exists(filepaths.directory_dataset):
        os.makedirs(filepaths.directory_dataset)

    ############
    #   Mesh   #
    ############
    #=== Construct Mesh ===#
    Vh, nodes, dof = construct_mesh(options)

    ############################
    #   Prior and Parameters   #
    ############################
    #=== Construct Prior ===#
    if options.construct_prior == 1:
        if options.prior_type_lp == 1:
            prior = construct_laplacian_prior(filepaths,
                                              Vh, options.prior_mean_lp,
                                              options.prior_gamma_lp,
                                              options.prior_delta_lp)

    #=== Draw Parameters from Prior ===#
    if options.draw_and_save_parameters == 1:
        prior_mean, _, prior_covariance_cholesky, _ = load_prior(filepaths, dof)
        draw_from_distribution(filepaths,
                               prior_mean, prior_covariance_cholesky, dof,
                               positivity_constraint_identity, 0.5,
                               num_samples = options.num_data)
        # draw_from_distribution_fenics(filepaths,
        #                               Vh, prior, dof,
        #                               num_samples = options.num_data)

    #=== Load Parameters ===#
    parameters = load_parameters(filepaths, dof, options.num_data)

    #=== Plot Parameters ===#
    if options.plot_parameters == 1:
        for n in range(0, options.num_data):
            plot_fem_function_fenics_1d(Vh, parameters[n,:],
                                        '',
                                        filepaths.directory_figures + 'parameter_%d.png' %(n),
                                        (5,5),
                                        (options.left_boundary, options.right_boundary),
                                        (limit_min_parameter,limit_max_parameter))

    ###################
    #   FEM Objects   #
    ###################
    #=== Construct or Load Matrices ===#
    if options.construct_and_save_matrices == 1:
        if not os.path.exists(filepaths.directory_dataset):
            os.makedirs(filepaths.directory_dataset)
        construct_system_matrices(filepaths, Vh)
    forward_matrix, mass_matrix = load_system_matrices(options, filepaths)

    ##########################
    #   Computing Solution   #
    ##########################
    #=== Solve PDE with Prematrices ===#
    state = solve_pde_assembled(options, filepaths,
                                parameters,
                                forward_matrix, mass_matrix)

    #=== Plot Solution ===#
    if options.plot_solutions == 1:
        for n in range(0, options.num_data):
            plot_fem_function_fenics_1d(Vh, state[n,:],
                                        '',
                                        filepaths.directory_figures + 'state_%d.png' %(n),
                                        (5,5),
                                        (options.left_boundary, options.right_boundary),
                                        (limit_min_state,limit_max_state))

    #=== Form Observation Data ===#
    obs_indices, _ = form_interior_observation_points(options, filepaths, Vh)
    form_observation_data(filepaths, state, obs_indices)
