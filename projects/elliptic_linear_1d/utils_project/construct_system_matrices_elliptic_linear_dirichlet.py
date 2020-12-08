import numpy as np
import pandas as pd

import dolfin as dl

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def u_boundary(x, on_boundary):
    return on_boundary and (x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS )

def construct_system_matrices(filepaths, Vh):
    #=== Define boundary conditions and PDE ===#
    u_bdr = dl.Expression("1-x[0]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh, u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh, u_bdr0, u_boundary)

    #=== Set Variables ===#
    u = dl.TrialFunction(Vh)
    p = dl.TestFunction(Vh)

    #=== Variational Form ===#
    a = dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx + dl.inner(u,p)*dl.dx
    mass_varf = dl.inner(u,p)*dl.dx

    #=== Assemble ===#
    A = dl.assemble(a)
    mass = dl.assemble(mass_varf)
    bc.apply(A)

    #=== Forward Operator ===#
    forward_matrix = np.linalg.inv(A.array())
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
