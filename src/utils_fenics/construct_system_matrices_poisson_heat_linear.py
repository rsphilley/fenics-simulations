import numpy as np
import pandas as pd

import dolfin as dl

from scipy import sparse

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_system_matrices(filepaths, Vh, boundary_matrix):
    #=== Set Variables ===#
    u = dl.TrialFunction(Vh)
    p = dl.TestFunction(Vh)

    #=== Variational Form ===#
    a = dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx
    mass_varf = dl.inner(u,p)*dl.dx

    #=== Assemble ===#
    A = dl.assemble(a)
    mass = dl.assemble(mass_varf)

    #=== Forward Operator ===#
    forward_matrix = np.linalg.inv(A.array() + boundary_matrix)
    forward_matrix = np.asarray(forward_matrix)
    mass_matrix = mass.array()

    #=== Save Forward Operator ===#
    df_forward_matrix = pd.DataFrame({'forward_matrix':forward_matrix.flatten()})
    df_forward_matrix.to_csv(filepaths.forward_matrix + '.csv', index=False)
    df_mass_matrix = pd.DataFrame({'mass_matrix':mass_matrix.flatten()})
    df_mass_matrix.to_csv(filepaths.mass_matrix + '.csv', index=False)

def load_system_matrices(options, filepaths):
    #=== Load Spatial Operator ===#
    df_forward_matrix = pd.read_csv(filepaths.forward_matrix + '.csv')
    forward_matrix = df_forward_matrix.to_numpy()
    df_mass_matrix = pd.read_csv(filepaths.mass_matrix + '.csv')
    mass_matrix = df_mass_matrix.to_numpy()

    return forward_matrix.reshape((options.num_nodes, options.num_nodes)),\
           mass_matrix.reshape((options.num_nodes, options.num_nodes))
