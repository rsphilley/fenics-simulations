import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time
import warnings
warnings.filterwarnings('ignore')

from integrals_pwl_prestiffness import integrals_pwl_prestiffness

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                   Standard                                  #
###############################################################################
def solve_PDE_standard(run_options, filepaths,
                       parameters,
                       nodes, elements,
                       boundary_matrix, load_vector):

    state = np.zeros((run_options.num_data, run_options.num_nodes))

    #=== Solving PDE ===#
    start_time_solver = time.time()
    for n in range(0, run_options.num_data):
        start_time_sample = time.time()
        stiffness_matrix = np.zeros((run_options.num_nodes, run_options.num_nodes))
        for k in range(0, elements.shape[0]):
            ver = elements[k,:]
            vertices_coords = nodes[ver,:]
            p_k = parameters[n,ver]
            stiffness_matrix[np.ix_(ver,ver)] = stiffness_matrix[np.ix_(ver,ver)] +\
                    integrals_pwl_prestiffness(vertices_coords)*(p_k[0] + p_k[1] + p_k[2])
            stiffness_matrix = sparse.csc_matrix(stiffness_matrix)

        state[n,:] = sparse.linalg.spsolve(stiffness_matrix + boundary_matrix, load_vector).T
        elapsed_time_sample = time.time() - start_time_sample
        print('Solved: %d of %d. Time taken: %4f'%(n, run_options.num_data, elapsed_time_sample))

    elapsed_time_solver = time.time() - start_time_solver
    print('All solutions computed. Total time taken: %4f'%(elapsed_time_solver))

    #=== Save Solution ===#
    if run_options.solve_gaussian_blobs_problem == 0:
        df_state = pd.DataFrame({'state': state.flatten()})
        df_state.to_csv(filepaths.state_full + '.csv', index=False)
    else:
        df_state = pd.DataFrame({'state': state.flatten()})
        df_state.to_csv(filepaths.state_full_gaussian_blobs + '.csv', index=False)

    return state

###############################################################################
#                               Using Prematrices                             #
###############################################################################
def solve_PDE_prematrices(run_options, filepaths,
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
    if run_options.solve_gaussian_blobs_problem == 0:
        df_state = pd.DataFrame({'state': state.flatten()})
        df_state.to_csv(filepaths.state_full + '.csv', index=False)
    else:
        df_state = pd.DataFrame({'state': state.flatten()})
        df_state.to_csv(filepaths.state_full_gaussian_blobs + '.csv', index=False)

    return state
