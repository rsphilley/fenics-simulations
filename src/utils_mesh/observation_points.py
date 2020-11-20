import os

import numpy as np
import pandas as pd
from fenics import *

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def form_interior_observation_points(options, filepaths, function_space):

    #=== Extract mesh and triangulate ===#
    mesh = function_space.mesh()
    coords = mesh.coordinates()

    #=== Get Observation Indices ==#
    np.random.seed(options.random_seed)
    obs_indices = np.sort(
                np.random.choice(np.arange(0,coords.shape[0]), options.num_obs_points,
                replace = False))

    #=== Form Observation Coordinates ===#
    obs_coords = np.zeros((options.num_obs_points, 2))
    for ind in range(0, options.num_obs_points):
        obs_coords[ind,:] = coords[obs_indices[ind],:]

    #=== Save observation indices ===#
    if not os.path.exists(filepaths.directory_dataset):
        os.makedirs(filepaths.directory_dataset)
    df_obs_indices = pd.DataFrame({'obs_indices': obs_indices})
    df_obs_indices.to_csv(filepaths.obs_indices + '.csv', index=False)

    return obs_indices, obs_coords

def load_observation_points(filepath, function_space):

    #=== Extract mesh and triangulate ===#
    mesh = function_space.mesh()
    coords = mesh.coordinates()

    #=== Load Observation Indicies ===#
    df_obs_indices = pd.read_csv(filepath + '.csv')
    obs_indices = df_obs_indices.to_numpy()

    #=== Form Observation Coordinates ===#
    obs_coords = np.zeros((len(obs_indices), 2))
    for ind in range(0, len(obs_indices)):
        obs_coords[ind,:] = coords[obs_indices[ind],:]

    return obs_indices, obs_coords

def form_observation_data(filepaths, state, obs_indices):

    state_obs = state[:,obs_indices]
    df_state_obs = pd.DataFrame({'state_obs': state_obs.flatten()})
    df_state_obs.to_csv(filepaths.state_obs + '.csv', index=False)
