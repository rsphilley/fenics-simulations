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
    a = -dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx + dl.inner(u,p)*dl.dx
    L = dl.inner(u,p)*dl.dx

    #=== Assemble ===#
    A = dl.assemble(a)
    b = dl.assemble(L)
    bc.apply(A)
    bc.apply(b)

    #=== Forward Operator ===#
    forward_operator = np.matmul(np.linalg.inv(A.array()),b.array())

    #=== Save Forward Operator ===#
    df_forward_operator = pd.DataFrame({'forward_operator':forward_operator.flatten()})
    df_forward_operator.to_csv(filepaths.forward_operator + '.csv', index=False)

def load_system_matrices(options, filepaths):
    #=== Load Spatial Operator ===#
    df_forward_operator = pd.read_csv(filepaths.forward_operator + '.csv')
    forward_operator = df_forward_operator.to_numpy()

    return forward_operator.reshape((options.num_nodes+1, options.num_nodes+1))
