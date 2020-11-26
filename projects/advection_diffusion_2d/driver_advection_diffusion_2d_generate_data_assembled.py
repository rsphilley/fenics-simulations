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

import numpy as np
import pandas as pd
import dolfin as dl

# Import src code
from utils_mesh.construct_mesh_rectangular_with_hole import construct_mesh
from utils_mesh.observation_points import form_interior_observation_points
from utils_mesh.plot_mesh import plot_mesh
from utils_prior.bilaplacian_prior import construct_bilaplacian_prior
from utils_prior.smoothness_prior_autocorr import smoothness_prior_autocorr
from utils_io.load_prior import load_prior
from utils_misc.positivity_constraints import positivity_constraint_exp,\
                                             positivity_constraint_log_exp
from utils_prior.draw_from_distribution import draw_from_distribution
from utils_io.load_parameters import load_parameters
from utils_hippylib.space_time_pointwise_state_observation\
        import SpaceTimePointwiseStateObservation
from utils_io.io_fem_operators import save_fem_operators, load_fem_operators
from utils_io.value_to_string import value_to_string
from utils_fenics.plot_fem_function_fenics_2d import plot_fem_function_fenics_2d

# Import project utilities
from utils_project.filepaths import FilePaths
from utils_project.velocity_field import compute_velocity_field_navier_stokes
from utils_project.model_advection_diffusion_2d_initial_condition\
        import TimeDependentAdvectionDiffusionInitialCondition
from utils_project.solve_advection_diffusion_2d import solve_pde

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
                                        Vh, options.prior_mean_blp,
                                        options.prior_gamma_blp,
                                        options.prior_delta_blp)
    if options.prior_type_ac == 1:
        smoothness_prior_autocorr(filepaths,
                                  nodes,
                                  options.prior_mean_ac,
                                  options.prior_variance_ac,
                                  options.prior_corr_ac)

    #=== Load Prior ===#
    prior_mean, _, prior_covariance_cholesky, _ = load_prior(filepaths, dof)

    #=== Draw Parameters from Prior ===#
    if options.draw_and_save_parameters == True:
        draw_from_distribution(filepaths,
                               prior_mean, prior_covariance_cholesky, dof,
                               positivity_constraint_log_exp, 0.5,
                               num_samples = options.num_data)

    #=== Load Parameters ===#
    parameters = load_parameters(filepaths, dof, options.num_data)

    #=== Plot Parameters ===#
    if options.plot_parameters == True:
        for n in range(0, options.num_data):
            plot_fem_function_fenics_2d(Vh, parameters[n,:],
                                        '',
                                        filepaths.directory_figures + 'parameter_%d.png' %(n),
                                        (5,5), 'none')

    #=== Specific Parameter for Plotting Time Evolution ===#
    sample_number = 8
    if not os.path.exists(filepaths.directory_dataset):
        os.makedirs(filepaths.directory_dataset)
    input_specific = parameters[sample_number,:]
    df_input_specific = pd.DataFrame({'input_specific': input_specific.flatten()})
    df_input_specific.to_csv(filepaths.input_specific + '.csv', index=False)

###############################################################################
#                                  Solve PDE                                  #
###############################################################################
    ##################
    #   Setting Up   #
    ##################
    #=== Velocity Field ===#
    if options.flow_navier_stokes == True:
        velocity = compute_velocity_field_navier_stokes(
                filepaths.directory_figures + 'velocity_field.png',
                Vh.mesh())

    #=== Time Objects ===#
    if options.time_stepping_implicit == True:
        time_dt = options.time_dt_imp
        time_obs_scalar = options.time_obs_imp_scalar
        simulation_times = np.arange(options.time_initial,
                                     options.time_final+.5*time_dt,
                                     time_dt)
        num_time_steps = simulation_times.shape[0]
    else:
        time_dt = float(options.time_dt_exp)
        time_obs_scalar = float(options.time_obs_exp_scalar)
        simulation_times = [] # Explicit time-stepping is not computed in the
                              # hIPPYlib provided class. Also, you don't want to
                              # store all these steps anyway
        num_time_steps = int(np.ceil((options.time_final+.5*time_dt)/time_dt))

    time_dt_obs = time_obs_scalar*time_dt
    observation_times = np.arange(options.time_initial,
                                  options.time_final+.5*time_dt,
                                  time_dt_obs)

    #=== Define Observation Points ===#
    obs_indices, obs_coords = form_interior_observation_points(options, filepaths, Vh)

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
    state_sample = solve_pde(options, filepaths,
                             parameters,
                             obs_indices,
                             time_dt, num_time_steps, observation_times.shape[0], time_obs_scalar,
                             fem_operator_spatial,
                             fem_operator_implicit_ts, fem_operator_implicit_ts_rhs,
                             sample_number, False)

    #=== Plot Solution ===#
    if options.plot_solutions == True:
        for time_step in range(0, observation_times.shape[0]):
            time_string = value_to_string(observation_times[time_step])
            plot_fem_function_fenics_2d(
                    Vh, state_sample[time_step,:],
                    'Time = %.2f' %(observation_times[time_step]),
                    filepaths.directory_figures + 'state_%d_t%s.png' %(sample_number, time_step),
                    (5,5), 'none')

    #=== Save State Evolution for Specific Parameter ===#
    df_output_specific = pd.DataFrame({'output_specific': state_sample.flatten()})
    df_output_specific.to_csv(filepaths.output_specific + '.csv', index=False)

###############################################################################
#                                Blob Test Case                               #
###############################################################################
    if options.compute_test_case == True:
        #=== Parameter ===#
        ic_expr = dl.Expression(
            'std::min(0.5,std::exp(-100*(std::pow(x[0]-0.35,2) +  std::pow(x[1]-0.7,2))))',
            element=Vh.ufl_element())
        true_initial_condition = np.expand_dims(
                dl.interpolate(ic_expr, Vh).vector().get_local(), axis=0)
        true_initial_condition = 10*true_initial_condition
        plot_fem_function_fenics_2d(Vh, true_initial_condition,
                                    '',
                                    filepaths.directory_figures + 'parameter_test.png',
                                    (5,5), (0,5))

        #=== Save Parameter ===#
        if not os.path.exists(filepaths.directory_dataset):
            os.makedirs(filepaths.directory_dataset)
        df_samples= pd.DataFrame({'parameter_blob': true_initial_condition.flatten()})
        df_samples.to_csv(filepaths.parameter_blob + '.csv', index=False)

        #=== State ===#
        state_sample = solve_pde(options, filepaths,
                                true_initial_condition,
                                obs_indices,
                                time_dt, num_time_steps, observation_times.shape[0], time_obs_scalar,
                                fem_operator_spatial,
                                fem_operator_implicit_ts, fem_operator_implicit_ts_rhs,
                                0, True)

        for time_step in range(0, observation_times.shape[0]):
            time_string = value_to_string(observation_times[time_step])
            plot_fem_function_fenics_2d(
                    Vh, state_sample[time_step,:],
                    'Time = %.2f' %(observation_times[time_step]),
                    filepaths.directory_figures + 'state_test_t%s.png' %(time_step),
                    (5,5), 'none')
