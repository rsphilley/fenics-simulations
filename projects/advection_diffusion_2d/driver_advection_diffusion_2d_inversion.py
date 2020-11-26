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

sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *
import math

# Import src code
from utils_mesh.construct_mesh_rectangular_with_hole import construct_mesh
from utils_mesh.observation_points import form_interior_observation_points
from utils_mesh.plot_mesh import plot_mesh
from utils_prior.bilaplacian_prior import construct_bilaplacian_prior
from utils_io.load_prior import load_prior
from utils_fenics.convert_array_to_dolfin_function import convert_array_to_dolfin_function
from utils_hippylib.space_time_pointwise_state_observation\
        import SpaceTimePointwiseStateObservation
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
#                              Inversion Options                              #
###############################################################################
    #=== True Parameter Options ===#
    true_parameter_blob = False
    true_parameter_specific = True

    #=== Prior Options ===#
    prior_scalar_yaml = True
    prior_scalar_set = False
    prior_scalar_value = 0

    #=== Noise Options ===#
    noise_level = 0.01

    #=== Plotting Options ===#
    use_hippylib_plotting = False
    use_my_plotting = True
    colourbar_limit_parameter = 7
    colourbar_limit_state = 2
    colourbar_limit_prior_variance = 2
    colourbar_limit_posterior_variance = 2
    cross_section_y_limit_min = 1
    cross_section_y_limit_max = 7.5


###############################################################################
#                                  Setting Up                                 #
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

    #=== Define Observation Points ===#
    obs_indices, obs_coords = form_interior_observation_points(options, filepaths, Vh)

    #=== Plot Mesh ===#
    if options.plot_mesh == True:
        plot_mesh(filepaths,
                  (5,5), '',
                  (0,1), (0,1),
                  Vh.mesh().coordinates(), Vh.mesh().cells())

###############################################################################
#                           Prior and True Parameter                          #
###############################################################################
    #=== Prior ===#
    if prior_scalar_yaml == True:
        if options.prior_mean_blp == 0:
            mean_array = 0*np.ones(Vh.dim())
        else:
            mean_array = options.prior_mean_blp*np.ones(Vh.dim())
    if prior_scalar_set == True:
        if prior_scalar_value == 0:
            mean_array = 0*np.ones(Vh.dim())
        else:
            mean_array = np.log(prior_scalar_value)*np.ones(Vh.dim())
    mean_dl = convert_array_to_dolfin_function(Vh, mean_array)
    mean = mean_dl.vector()
    prior = BiLaplacianPrior(Vh,
                             options.prior_gamma_blp,
                             options.prior_delta_blp,
                             mean = mean,
                             robin_bc=True)

    #=== True Parameter ===#
    if true_parameter_blob == True:
        ic_expr = dl.Expression(
            'std::min(0.5,std::exp(-100*(std::pow(x[0]-0.35,2) +  std::pow(x[1]-0.7,2))))',
            element=Vh.ufl_element())
        true_initial_condition = np.expand_dims(
                dl.interpolate(ic_expr, Vh).vector().get_local(), axis=0)
        true_initial_condition = true_initial_condition
    if true_parameter_specific == True:
        df_mtrue = pd.read_csv(filepaths.input_specific + '.csv')
        mtrue_array = df_mtrue.to_numpy()
        mtrue_dl = convert_array_to_dolfin_function(Vh, mtrue_array)
        true_initial_condition = mtrue_dl.vector()

###############################################################################
#                                  PDE Problem                                #
###############################################################################
    #=== Velocity Field ===#
    if options.flow_navier_stokes == True:
        velocity = compute_velocity_field_navier_stokes(
                filepaths.directory_figures + 'velocity_field.png',
                Vh.mesh())

    #=== Temporal Observation Objects ===#
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

    #=== PDE Problem ===#
    misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, obs_coords)
    pde = TimeDependentAdvectionDiffusionInitialCondition(options,
                                                          Vh.mesh(), [Vh,Vh,Vh],
                                                          prior, misfit,
                                                          simulation_times,
                                                          velocity, True)

###############################################################################
#                        Generate Synthetic Observations                      #
###############################################################################
    #=== Generate True State and Observations ===#
    utrue = pde.generate_vector(STATE)
    x = [utrue, true_initial_condition, None]
    pde.solveFwd(x[STATE], x)
    misfit.observe(x, misfit.d)

    #=== Noise Model ===#
    rel_noise = noise_level
    MAX = misfit.d.norm("linf", "linf")
    noise_std_dev = rel_noise * MAX
    parRandom.normal_perturb(noise_std_dev,misfit.d)
    misfit.noise_variance = noise_std_dev*noise_std_dev
