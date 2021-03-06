import pandas as pd
from scipy import sparse

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def save_fem_operators(options, filepaths, pde_opt_problem):

    if options.time_stepping_erk4 == True or options.time_stepping_lserk4 == True:
        #=== Save Spatial Operator ===#
        df_fem_operator_spatial = pd.DataFrame({
            'fem_operator_spatial':pde_opt_problem.N.array().flatten()})
        df_fem_operator_spatial.to_csv(
                filepaths.fem_operator_spatial + '.csv', index=False)

    if options.time_stepping_implicit == True:
        #=== Save Implicit Time Stepping Operators ===#
        df_fem_operator_implicit_ts = pd.DataFrame({
            'fem_operator_implicit_ts':pde_opt_problem.L.array().flatten()})
        df_fem_operator_implicit_ts.to_csv(
                filepaths.fem_operator_implicit_ts + '.csv', index=False)
        df_fem_operator_implicit_ts_rhs = pd.DataFrame({
            'fem_operator_implicit_ts_rhs':pde_opt_problem.L_rhs.array().flatten()})
        df_fem_operator_implicit_ts_rhs.to_csv(
            filepaths.fem_operator_implicit_ts_rhs + '.csv', index=False)

    print('FEM operators saved')

def load_fem_operators(options, filepaths):

    if options.time_stepping_erk4 == True or options.time_stepping_lserk4 == True:
        #=== Load Spatial Operator ===#
        df_fem_operator_spatial = pd.read_csv(filepaths.fem_operator_spatial + '.csv')
        fem_operator_spatial = df_fem_operator_spatial.to_numpy()
        fem_operator_spatial = fem_operator_spatial.reshape(
                (options.num_nodes, options.num_nodes))
    else:
        fem_operator_spatial = 0

    if options.time_stepping_implicit == True:
        #=== Load Implicit Time Stepping Operators ===#
        df_fem_operator_implicit_ts = pd.read_csv(
                filepaths.fem_operator_implicit_ts + '.csv')
        fem_operator_implicit_ts = df_fem_operator_implicit_ts.to_numpy()
        fem_operator_implicit_ts = fem_operator_implicit_ts.reshape(
                (options.num_nodes, options.num_nodes))

        df_fem_operator_implicit_ts_rhs = pd.read_csv(
                filepaths.fem_operator_implicit_ts_rhs + '.csv')
        fem_operator_implicit_ts_rhs = df_fem_operator_implicit_ts_rhs.to_numpy()
        fem_operator_implicit_ts_rhs = fem_operator_implicit_ts_rhs.reshape(
                (options.num_nodes, options.num_nodes))
    else:
        fem_operator_implicit_ts = 0
        fem_operator_implicit_ts_rhs = 0

    return fem_operator_spatial, fem_operator_implicit_ts, fem_operator_implicit_ts_rhs
