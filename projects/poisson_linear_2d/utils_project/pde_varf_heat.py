import dolfin as dl
from fenics import *
import ufl

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def pde_varf_heat(u,m,p,
                  Vh,
                  boundary_matrix_constant, load_vector_constant):
    #=== Computational mesh ===#
    mesh = Vh.mesh()
    domains = MeshFunction("size_t", mesh, mesh.topology().dim())
    domains.set_all(0)

    #=== Marking boundaries for boundary conditions ===#
    exterior = CompiledSubDomain("!near(x[1], side) && on_boundary", side = -1)
    bottom = CompiledSubDomain("near(x[1], side) && on_boundary", side = -1)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    exterior.mark(boundaries, 1)
    bottom.mark(boundaries, 2)

    #=== Defining Measures ===#
    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    return m*ufl.inner(ufl.grad(u), ufl.grad(p))*dx +\
           ufl.inner(boundary_matrix_constant*u,p)*ds(1) +\
           load_vector_constant*p*ds(2)
