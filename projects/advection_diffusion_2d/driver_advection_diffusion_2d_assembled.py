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
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl

# Import src code
from utils_mesh.construct_mesh_rectangular_with_hole import construct_mesh
from utils_mesh.plot_mesh import plot_mesh
from utils_prior.bilaplacian_prior import construct_bilaplacian_prior
from utils_io.load_prior import load_prior
from utils_prior.draw_from_distribution import draw_from_distribution
from utils_io.load_parameters import load_parameters
from utils_hippylib.space_time_pointwise_state_observation\
        import SpaceTimePointwiseStateObservation
from utils_io.io_fem_operators import save_fem_operators, load_fem_operators
from utils_io.value_to_string import value_to_string
from utils_fenics.plot_fem_function_fenics_2d import plot_fem_function_fenics_2d

# Import project utilities
from utils_project.filepaths import FilePaths
from utils_project.velocity_field import compute_velocity_field
from utils_project.model_advection_diffusion_2d_initial_condition\
        import TimeDependentAdvectionDiffusionInitialCondition
from utils_project.solve_advection_diffusion_2d import solve_pde
from utils_project.form_observation_data import form_observation_points, form_observation_data

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

if __name__ == "__main__":

###############################################################################
#                                Setup and Mesh                               #
###############################################################################
    #=== Options ===#
    with open('config_files/options.yaml') as f:
        options = yaml.safe_load(f)
    options = AttrDict(options)

    #=== Construct Mesh and Function Space ===#
    Vh, nodes, dof = construct_mesh(options)
    options.num_nodes = dof
    print(dof)

    #=== File Paths ===#
    filepaths = FilePaths(options)

    #=== Plot Mesh ===#
    if options.plot_mesh == True:
        plot_mesh(filepaths,
                  (5,5), '',
                  (0,1), (0,1),
                  Vh.mesh().coordinates(), Vh.mesh().cells())

###############################################################################
#                             Prior and Parameters                            #
###############################################################################
    #=== Construct Prior ===#
    prior = construct_bilaplacian_prior(filepaths,
                                        Vh, options.prior_mean,
                                        options.prior_gamma, options.prior_delta)

    #=== Load Prior ===#
    prior_mean, _, prior_covariance_cholesky, _ = load_prior(filepaths, dof)

    #=== Draw Parameters from Prior ===#
    if options.draw_and_save_parameters == True:
        draw_from_distribution(filepaths,
                               prior_mean, prior_covariance_cholesky, dof,
                               num_samples = options.num_data)

    #=== Load Parameters ===#
    parameters = load_parameters(filepaths, dof, options.num_data)

    #=== Plot Parameters ===#
    if options.plot_parameters == True:
        for n in range(0, options.num_data):
            plot_fem_function_fenics_2d(Vh, parameters[n,:],
                                        '',
                                        filepaths.directory_figures + 'parameter_%d.png' %(n),
                                        (5,5), (0,0.5))

###############################################################################
#                                  Solve PDE                                  #
###############################################################################
    ##################
    #   Setting Up   #
    ##################
    #=== Velocity Field ===#
    if options.flow_navier_stokes == True:
        velocity = compute_velocity_field(filepaths.directory_figures + 'velocity_field.png',
                                          Vh.mesh())

    #=== Time Objects ===#
    simulation_times = np.arange(options.time_initial,
                                 options.time_final+.5*options.time_dt,
                                 options.time_dt)
    observation_times = np.arange(options.time_1,
                                  options.time_final+.5*options.time_dt,
                                  options.time_obs)

    #=== Define Observation Points ===#
    obs_indices, obs_coords = form_observation_points(options, filepaths, Vh)

    #=== Construct or Load FEM Operators ===#
    if options.construct_and_save_matrices == True:
        if not os.path.exists(filepaths.directory_dataset):
            os.makedirs(filepaths.directory_dataset)
        misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, obs_coords)
        pde_opt_problem = TimeDependentAdvectionDiffusionInitialCondition(options,
                                                                          Vh.mesh(), [Vh,Vh,Vh],
                                                                          prior, misfit,
                                                                          simulation_times,
                                                                          velocity, True)
        save_fem_operators(options, filepaths, pde_opt_problem)
    fem_operator_spatial,\
    fem_operator_implicit_ts, fem_operator_implicit_ts_rhs =\
            load_fem_operators(options, filepaths)

    ##########################
    #   Computing Solution   #
    ##########################
    #=== Solve PDE ===#
    sample_number = 0
    state_sample = solve_pde(options, filepaths,
                             parameters,
                             obs_indices, simulation_times.shape[0],
                             fem_operator_spatial,
                             fem_operator_implicit_ts, fem_operator_implicit_ts_rhs,
                             sample_number)

    #=== Plot Solution ===#
    if options.plot_solutions == True:
        for time_step in range(0, simulation_times.shape[0]):
            time_string = value_to_string(simulation_times[time_step])
            plot_fem_function_fenics_2d(
                    Vh, state_sample[time_step,:],
                    'Time = %.2f' %(simulation_times[time_step]),
                    filepaths.directory_figures + 'state_%d_t%s.png' %(sample_number, time_step),
                    (5,5), (0,0.5))

    #################
    #   Test Case   #
    #################
    if options.compute_test_case == True:
        #=== Parameter ===#
        ic_expr = dl.Expression(
            'std::min(0.5,std::exp(-100*(std::pow(x[0]-0.35,2) +  std::pow(x[1]-0.7,2))))',
            element=Vh.ufl_element())
        true_initial_condition = np.expand_dims(
                dl.interpolate(ic_expr, Vh).vector().get_local(), axis=0)
        plot_fem_function_fenics_2d(Vh, true_initial_condition,
                                    '',
                                    filepaths.directory_figures + 'parameter_test.png',
                                    (5,5), 'none')

        #=== State ===#
        state_sample = solve_pde(options, filepaths,
                                true_initial_condition,
                                obs_indices, simulation_times.shape[0],
                                fem_operator_spatial,
                                fem_operator_implicit_ts, fem_operator_implicit_ts_rhs,
                                0)

        for time_step in range(0, simulation_times.shape[0]):
            time_string = value_to_string(simulation_times[time_step])
            plot_fem_function_fenics_2d(
                    Vh, state_sample[time_step,:],
                    'Time = %.2f' %(simulation_times[time_step]),
                    filepaths.directory_figures + 'state_test_t%s.png' %(time_step),
                    (5,5), 'none')
