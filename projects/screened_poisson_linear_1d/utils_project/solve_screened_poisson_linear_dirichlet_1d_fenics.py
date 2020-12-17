import numpy as np
import pandas as pd
import time

import dolfin as dl
from hippylib import *

# Import src code
from utils_fenics.convert_array_to_dolfin_function import convert_array_to_dolfin_function
from utils_hippylib.pde_varf_screened_poisson_source import pde_varf

# Import project utilities
from utils_project.pde_variational_problem import PDEVariationalProblem

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                            Using Fenics Solver                              #
###############################################################################
def u_boundary(x, on_boundary):
    return on_boundary and (x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS )

def solve_pde_fenics(options, filepaths,
                     parameters,
                     Vh):

    #=== Define boundary conditions and PDE ===#
    u_bdr = dl.Expression("1-x[0]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh, u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh, u_bdr0, u_boundary)

    #=== Set Variables ===#
    u = dl.TrialFunction(Vh)
    p = dl.TestFunction(Vh)

    #=== Variational Class ===#
    Vh_hippylib = [Vh, Vh, Vh]
    pde = PDEVariationalProblem(Vh_hippylib, pde_varf, bc, bc0, is_fwd_linear=True)
    state_dl_vector = pde.generate_state()

    #=== State Storage ===#
    state = np.zeros((options.num_data, options.num_nodes))

    #=== Solving PDE ===#
    start_time_solver = time.time()
    for n in range(0, options.num_data):
        start_time_sample = time.time()
        parameter_dl_function = convert_array_to_dolfin_function(Vh, parameters[n,:])
        parameter_dl_vector = parameter_dl_function.vector()
        x = [state_dl_vector, parameter_dl_vector, None]
        pde.solveFwd(x[STATE], x)
        state[n,:] = np.array(x[STATE])
        elapsed_time_sample = time.time() - start_time_sample
        print('Solved: %d of %d. Time taken: %4f'%(n, options.num_data, elapsed_time_sample))

    elapsed_time_solver = time.time() - start_time_solver
    print('All solutions computed. Total time taken: %4f'%(elapsed_time_solver))

    #=== Save Solution ===#
    df_state = pd.DataFrame({'state': state.flatten()})
    df_state.to_csv(filepaths.state_full + '.csv', index=False)

    return state
