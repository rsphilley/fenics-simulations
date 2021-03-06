import numpy as np
import pandas as pd

from utils_time_stepping.time_stepping_implicit import time_stepping_implicit
from utils_time_stepping.time_stepping_explicit import time_stepping_erk4

from scipy import sparse
from scipy.sparse.linalg import spsolve
import time
import warnings
warnings.filterwarnings('ignore')

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                               Using Prematrices                             #
###############################################################################
def solve_pde(options, filepaths,
              parameters,
              obs_indices,
              time_dt, num_time_steps, num_time_steps_obs, time_obs_scalar,
              fem_operator_spatial,
              fem_operator_implicit_ts, fem_operator_implicit_ts_rhs,
              sample_number, generate_blob_flag):

    #######################
    #   Storage Vectors   #
    #######################
    state_obs = np.zeros((parameters.shape[0], options.num_obs_points*num_time_steps_obs))

    #########################################
    #    Computing All States in Dataset    #
    #########################################
    start_time_solver = time.time()
    for n in range(0, parameters.shape[0]):
        start_time_sample = time.time()

        #=== Setting up Initial Structures ===#
        state_current = parameters[n,:]
        state_n = np.expand_dims(state_current, axis=0)
        state_obs_n = state_n[:,obs_indices]

        #=== Time Stepping ===#
        for time_step in range(1, num_time_steps):
            start_time_step = time.time()
            if options.time_stepping_implicit == True:
                state_current = time_stepping_implicit(
                        fem_operator_implicit_ts, fem_operator_implicit_ts_rhs, state_current)
            if options.time_stepping_erk4 == True:
                state_current = time_stepping_erk4(time_dt, -fem_operator_spatial, state_current)
            elapsed_time_step = time.time() - start_time_step
            print('Time Step: %d of %d. Time taken: %4f'\
                    %(time_step, num_time_steps, elapsed_time_step))
            #=== Store Observations ===#
            if np.mod(time_step, time_obs_scalar) == 0:
                state_current_expanded = np.expand_dims(state_current, axis=0)
                state_obs_n = np.concatenate(
                    (state_obs_n, state_current_expanded[:,obs_indices]), axis=1)
                if n == sample_number: # For visualization purposes
                    state_n = np.concatenate((state_n, state_current_expanded), axis=1)

        #=== Finalizing Sample ===#
        state_obs[n,:] = state_obs_n
        if n == sample_number: # For visualization purposes
            state_sample = np.reshape(state_n, (num_time_steps_obs, options.num_nodes))

        elapsed_time_sample = time.time() - start_time_sample
        print('============================================================================')
        print('Solved: %d of %d. Time taken: %4f'%(n, options.num_data, elapsed_time_sample))
        print('============================================================================')

    elapsed_time_solver = time.time() - start_time_solver
    print('All solutions computed. Total time taken: %4f'%(elapsed_time_solver))

    ########################
    #    Save Solutions    #
    ########################
    df_state_obs = pd.DataFrame({'state_obs': state_obs.flatten()})
    if generate_blob_flag == False:
        df_state_obs.to_csv(filepaths.state_obs + '.csv', index=False)
    else:
        df_state_obs.to_csv(filepaths.state_obs_blob + '.csv', index=False)

    return state_sample
