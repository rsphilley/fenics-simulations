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

# Import hIPPYlib code
import dolfin as dl
import ufl
import math
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.environ.get('HIPPYLIB_BASE_DIR', "../"))
from hippylib import *
sys.path.append(os.environ.get('HIPPYLIB_BASE_DIR', "..") + "/applications/ad_diff/")
# from model_ad_diff import TimeDependentAD, SpaceTimePointwiseStateObservation

# Import src code
from utils_mesh.construct_mesh_rectangular_with_hole import construct_mesh
from utils_prior.bilaplacian_prior import construct_bilaplacian_prior
from utils_io.load_prior import load_prior
from utils_prior.draw_from_distribution import draw_from_distribution
from utils_io.load_parameters import load_parameters
from utils_fenics.plot_fem_function_fenics_2d import plot_fem_function_fenics_2d
# from prematrices.construct_prematrix import construct_prematrix
# from utils_fenics.construct_boundary_matrices_and_load_vector import\
#         construct_boundary_matrices_and_load_vector
# from utils_io.load_fem_matrices import load_boundary_matrices_and_load_vector

# Import project utilities
from utils_project.filepaths import FilePaths
from utils_project.velocity_field import compute_velocity_field
# from utils_project.weak_forms import stiffness
# from utils_project.solve_poisson_2d import solve_pde_prematrices
from utils_project.form_observation_data import form_observation_points, form_observation_data

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

    ############
    #   Mesh   #
    ############
    #=== Construct Mesh and Function Space ===#
    Vh, nodes, dof = construct_mesh(options)

    #=== File Paths ===#
    options.num_nodes = dof
    print(dof)
    filepaths = FilePaths(options)

    ############################
    #   Prior and Parameters   #
    ############################
    #=== Construct Prior ===#
    if options.construct_prior == 1:
            construct_bilaplacian_prior(filepaths,
                                        Vh, options.prior_mean,
                                        options.prior_gamma, options.prior_delta)

    #=== Load Prior ===#
    prior_mean, _, prior_covariance_cholesky, _ = load_prior(filepaths, dof)

    #=== Draw Parameters from Prior ===#
    if options.draw_and_save_parameters == 1:
        draw_from_distribution(filepaths,
                               prior_mean, prior_covariance_cholesky, dof,
                               num_samples = options.num_data)

    #=== Load Parameters ===#
    parameters = load_parameters(filepaths, dof, options.num_data)

    #=== Plot Parameters ===#
    if options.plot_parameters == 1:
        for n in range(0, options.num_data):
            plot_fem_function_fenics_2d(Vh, parameters[n,:],
                                        '',
                                        filepaths.directory_figures + 'parameter_%d.png' %(n),
                                        (5,5), (0,4))

    ######################
    #   Velocity Field   #
    ######################
    wind_velocity = compute_velocity_field(filepaths.directory_figures + 'velocity_field.png',
                                           Vh.mesh())

    ###################
    #   FEM Objects   #
    ###################
    #=== Time Objects ===#
    simulation_times = np.arange(options.time_initial,
                                 options.time_final+.5*options.time_dt,
                                 options.time_dt)
    observation_times = np.arange(options.time_1,
                                  options.time_final+.5*options.time_dt,
                                  options.time_obs)

    #=== Define Observation Points ===#
    obs_coords = form_observation_points(options, filepaths, Vh)

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
                                        (5,5), (0,1))

    #=== Form Observation Data ===#
    form_observation_data(options, filepaths, fe_space, state)
