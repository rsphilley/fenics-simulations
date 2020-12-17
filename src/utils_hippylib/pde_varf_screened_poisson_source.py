import dolfin as dl
from fenics import *

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def pde_varf(u, m, p):
    return dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx + u*p*dl.dx - m*p*dl.dx
