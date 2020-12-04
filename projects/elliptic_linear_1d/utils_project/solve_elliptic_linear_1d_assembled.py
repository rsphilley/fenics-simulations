import numpy as np
import pandas as pd
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                            Using Assembled Matrix                           #
###############################################################################
def solve_pde_assembled(options, filepaths,
                        parameters,
                        forward_operator):

    state = np.zeros((options.num_data, options.num_nodes))

    #=== Solving PDE ===#
    start_time_solver = time.time()

    for n in range(0, options.num_data):
        start_time_sample = time.time()
        state[n,:] = np.matmul(parameters[n,:], np.transpose(forward_operator))
        elapsed_time_sample = time.time() - start_time_sample
        print('Solved: %d of %d. Time taken: %4f'%(n, options.num_data, elapsed_time_sample))

    elapsed_time_solver = time.time() - start_time_solver
    print('All solutions computed. Total time taken: %4f'%(elapsed_time_solver))

    #=== Save Solution ===#
    df_state = pd.DataFrame({'state': state.flatten()})
    df_state.to_csv(filepaths.state_full + '.csv', index=False)

    return state
