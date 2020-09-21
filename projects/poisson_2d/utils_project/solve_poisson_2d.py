import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time
import warnings
warnings.filterwarnings('ignore')

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                               Using Prematrices                             #
###############################################################################
def solve_pde_prematrices(run_options, filepaths,
        parameters,
        prestiffness, boundary_matrix, load_vector):

    state = np.zeros((run_options.num_data, run_options.num_nodes))

    #=== Solving PDE ===#
    start_time_solver = time.time()
    prestiffness = sparse.csr_matrix.dot(prestiffness, parameters.T)
    prestiffness = sparse.csc_matrix(prestiffness)

    for n in range(0, run_options.num_data):
        start_time_sample = time.time()
        stiffness_matrix = np.reshape(prestiffness[:,n],
                (run_options.num_nodes, run_options.num_nodes))

        state[n,:] = sparse.linalg.spsolve(stiffness_matrix + boundary_matrix, load_vector).T
        elapsed_time_sample = time.time() - start_time_sample
        print('Solved: %d of %d. Time taken: %4f'%(n, run_options.num_data, elapsed_time_sample))

    elapsed_time_solver = time.time() - start_time_solver
    print('All solutions computed. Total time taken: %4f'%(elapsed_time_solver))

    #=== Save Solution ===#
    df_state = pd.DataFrame({'state': state.flatten()})
    df_state.to_csv(filepaths.state_full_gaussian_blobs + '.csv', index=False)

    return state
