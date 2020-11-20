import numpy as np
import pandas as pd
from fenics import *

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def form_observation_data(options, filepaths, fe_space, state):

    # WARNING: DOESN'T ACTUALLY FORM POINTS ON THE BOUNDARY, NEED TO FIX!

    #=== Get boundary indices ==#
    exterior_domain = CompiledSubDomain("!near(x[1], 0.0) && on_boundary")
    exterior_bc = DirichletBC(fe_space, 1, exterior_domain)
    u = Function(fe_space)
    exterior_bc.apply(u.vector())
    boundary_indices = (u.vector() == 1)
    boundary_indices = np.nonzero(boundary_indices)[0]

    #=== Get obs indices ==#
    np.random.seed(options.random_seed)
    obs_indices = np.sort(
            np.random.choice(boundary_indices, options.num_obs_points,
                replace = False))

    #=== Form observation data ===#
    state_obs = state[:,obs_indices]

    #=== Save observation indices and data ===#
    df_obs_indices = pd.DataFrame({'obs_indices': obs_indices})
    df_obs_indices.to_csv(filepaths.obs_indices + '.csv', index=False)
    df_state_obs = pd.DataFrame({'state_obs': state_obs.flatten()})
    df_state_obs.to_csv(filepaths.state_obs + '.csv', index=False)
