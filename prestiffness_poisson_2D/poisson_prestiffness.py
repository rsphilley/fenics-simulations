'''
Created on Mon Aug 31 2020 
@author: jon
@description: Script to generate prestiffness matrix for the 2D poisson equation.
              Note that in this formulation, the fe_space for the standard fe problem
              does not have to be the same as the meta-fe space where we discretize the 
              spacially varying parameter, sigma(x,y).
              div(sigma grad(u)) = f
'''
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from fenics import *

def get_prestiffness(test=False):
    # define domain - rectangular in this case
    p1 = Point(-1, -1)
    p2 = Point(1, 1)
    nx = 9
    ny = 9
    mesh = RectangleMesh(p1, p2, nx, ny)
    
    # finite element space
    fe_space = FunctionSpace(mesh, 'P', 3)
    
    # meta-FE space - discretize the conductivity/parameter
    # same space as fe_space, but does not have to be
    meta_space = FunctionSpace(mesh, 'P', 2)
    
    # functions on the spaces
    u = TrialFunction(fe_space)
    v = TestFunction(fe_space)
    sig = Function(meta_space)
    
    # define fenics boundary conditions
    bc = DirichletBC(fe_space, Constant(0.0), 'on_boundary')
    
    # get the number of degrees of freedom for shaping matrices/vectors
    fe_dof = fe_space.dim()
    meta_dof = meta_space.dim()
    fe_shape = (fe_dof, fe_dof)
    pK_shape = (fe_dof**2, meta_dof)
    
    # for choosing which basis functions are active
    sig_vec = np.zeros((meta_dof))
    
    # lists for storing prestiffness coordinataes
    pK_rows = np.array([])
    pK_cols = np.array([])
    pK_data = np.array([])
    
    # get all the row/col/data pairs for CSR prestiffness matrix
    for k in range(meta_dof):
        sig_vec[k] = 1.0
        if k > 0:
            sig_vec[k-1] = 0.0
            
        sig.vector()[:] = sig_vec
            
        A = assemble(sig * inner(grad(u), grad(v)) * dx)
        A = as_backend_type(A).mat()
        A = sparse.csr_matrix(A.getValuesCSR()[::-1], shape=fe_shape)
        A.eliminate_zeros() # in-place operation, so does not return anything
        
        # extract row/column indices, we need coo format for this
        A = A.tocoo()
        rows = A.row
        cols = A.col
        data = A.data
        pK_rows = np.hstack((pK_rows, (fe_dof * cols + rows)))
        cols[:] = k
        pK_cols = np.hstack((pK_cols, cols))
        pK_data = np.hstack((pK_data, data))
        
    # construct prestiffness matrix
    pK = sparse.csr_matrix((pK_data, (pK_rows, pK_cols)), shape=pK_shape)

    print()
    print(f'Shape of prestiffness matrix: {pK.shape}')
    print(f'Number of non-zeros in pK:    {pK.nnz}')
    print(f'Percent of non-zeros in pK:   {pK.nnz / (pK.shape[0] * pK.shape[1]) * 100} %')


    
    if test == True:
        # need 2D sig_vec for sparse multiplication
        # test with random conductivity vector
        sig_vec = np.random.uniform(size=(meta_dof,1))

        # need 1-D sig_vec for this assignment
        sig.vector()[:] = sig_vec[:,0]
        
        A_assembled = assemble(sig * inner(grad(u), grad(v)) * dx).array()

        A_csr = sparse.csr_matrix(A_assembled)
        print()
        print(f'Numer of non-zeros in stiffness matrix: {A_csr.nnz}')
        print(f'Prestiffness matrix is {pK.nnz / A_csr.nnz} times larger than stiffness matrix in sparse format')
        print()
        
        A_prestffness = pK.dot(sparse.csr_matrix(sig_vec))

        # A_prestiffness is no longer sparse
        A_prestffness = np.reshape(A_prestffness, fe_shape)
        
        max_diff = np.amax(abs(A_assembled - A_prestffness))
        print(f'max diff between direct assembly of stiffness mat and using prestiffness: {max_diff}')
    
    return pK


if __name__ == '__main__':
    pK = get_prestiffness(True)
    
    sparse.save_npz('prestiffness_mat.npz', pK)
