import dolfin as dl
import ufl

# Import project utilities
from utils_project.pde_variational_problem import PDEVariationalProblem

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def u_boundary(x, on_boundary):
    return on_boundary and (x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS )

def pde_varf(u, m, p):
    return dl.inner(dl.nabla_grad(u), dl.nabla_grad(p)) * dl.dx + u * p * dl.dx - m * p * dl.dx

def construct_system_matrices(Vh):
    #=== Define boundary conditions and PDE ===#
    u_bdr = dl.Expression("1-x[0]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh, u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh, u_bdr0, u_boundary)

    #=== Set Variables ===#
    u = dl.TrialFunction(Vh)
    m = u
    p = dl.TestFunction(Vh)

    #=== Assemble PDE ===#
    res_form = pde_varf(u, m, p)
    pdb.set_trace()
    A_form = ufl.lhs(res_form)
    b_form = ufl.rhs(res_form)
    A, b = dl.assemble_system(A_form, b_form, bcs=bc)
