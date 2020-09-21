import numpy as np
from fenics import *
from scipy import sparse

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_boundary_matrices_and_load_vector(filepaths,
        fe_space,
        boundary_matrix_constant, load_vector_constant):

    #=== Computational mesh ===#
    mesh = fe_space.mesh()
    domains = MeshFunction("size_t", mesh, mesh.topology().dim())
    domains.set_all(0)

    #=== Trial and test functions for the weak forms ===#
    u = TrialFunction(fe_space)
    v = TestFunction(fe_space)

    #=== Marking boundaries for boundary conditions ===#
    bottom = CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
    exterior = CompiledSubDomain("!near(x[1], side) && on_boundary", side = 0.0)
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    exterior.mark(boundaries, 1)
    bottom.mark(boundaries, 2)

    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    #=== Boundary Matrix and Load Vector ===#
    boundary_matrix = assemble(inner(boundary_matrix_constant*u,v)*dx(1)).array()
    load_vector = np.array(assemble(v*ds(2)))

    #=== Save Boundary Matrices and Load Vector ===#
    print('saving boundary matrix and load vector')
    boundary_matrix = sparse.csc_matrix(boundary_matrix)
    sparse.save_npz(filepaths.boundary_matrix + '.npz', boundary_matrix)
    load_vector = sparse.csc_matrix(load_vector)
    sparse.save_npz(filepaths.load_vector + '.npz', load_vector)
    print('boundary matrix and load vector saved')
