import dolfin as dl
from fenics import *
import ufl

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def pde_varf_elliptic_neumann(u,m,p,
                              Vh):
    return ufl.inner(ufl.grad(u), ufl.grad(p))*dx +\
           ufl.inner(u,p)*dx -\
           ufl.inner(m,p)*dx
