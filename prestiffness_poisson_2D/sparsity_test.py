from fenics import *
import numpy as np
import scipy.sparse as sps

p1   = Point(-1, -1)
p2   = Point(1, 1)
nx   = 10
ny   = 10
mesh = RectangleMesh(p1, p2, nx, ny)
V    = FunctionSpace(mesh, 'P', 1)  

u    = TrialFunction(V)
v    = TestFunction(V)
sig  = Function(V)

L    = inner(grad(u), grad(v)) * dx
A    = assemble(L)
mat  = as_backend_type(A).mat() 
A_sparse = sps.csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)

