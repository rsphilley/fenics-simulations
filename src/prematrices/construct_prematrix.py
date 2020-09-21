'''
Created on Mon Aug 31 2020
@author: Jon Wittmer
@description: Script to generate finite element prematrices.
              Note that in this formulation, the fe_space for the standard fe problem
              does not have to be the same as the meta-fe space where we discretize the
              spacially varying parameter, sigma(x,y).
'''
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from fenics import *

def construct_prematrix(option,
                        fe_space, meta_space,
                        dof_fe, dof_meta,
                        weak_form, test=False):

    # functions on the spaces
    u = TrialFunction(fe_space)
    v = TestFunction(fe_space)
    parameter = Function(meta_space)

    # for choosing which basis functions are active
    parameter_vec = np.zeros((dof_meta))

    # prematrix dimensions = n^2 x n
    prematrix_shape = (dof_fe**2, dof_meta)

    # lists for storing prestiffness coordinataes
    prematrix_rows = np.array([])
    prematrix_cols = np.array([])
    prematrix_data = np.array([])

    # get all the row/col/data pairs for CSR prestiffness matrix
    for k in range(dof_meta):
        parameter_vec[k] = 1.0
        if k > 0:
            parameter_vec[k-1] = 0.0

        parameter.vector()[:] = parameter_vec

        A = assemble(weak_form(parameter, u, v))
        A = as_backend_type(A).mat()
        A = sparse.csr_matrix(A.getValuesCSR()[::-1], shape=(dof_fe, dof_fe))
        A.eliminate_zeros() # in-place operation, so does not return anything

        # extract row/column indices, we need coo format for this
        A = A.tocoo()
        rows = A.row
        cols = A.col
        data = A.data
        prematrix_rows = np.hstack((prematrix_rows, (dof_fe * cols + rows)))
        cols[:] = k
        prematrix_cols = np.hstack((prematrix_cols, cols))
        prematrix_data = np.hstack((prematrix_data, data))

    # construct prestiffness matrix
    prematrix = sparse.csr_matrix(
            (prematrix_data, (prematrix_rows, prematrix_cols)), shape=prematrix_shape)
    print()
    print(f'Shape of prestiffness matrix: {prematrix.shape}')
    print(f'Number of non-zeros in prematrix: {prematrix.nnz}')
    print('Percent of non-zeros in prematrix: '\
            f'{prematrix.nnz / (prematrix.shape[0] * prematrix.shape[1]) * 100} %')

    if test == True:
        # need 2D parameter_vec for sparse multiplication
        # test with random conductivity vector
        parameter_vec = np.random.uniform(size=(dof_meta,1))

        # need 1-D parameter_vec for this assignment
        parameter.vector()[:] = parameter_vec[:,0]

        A_assembled = assemble(weak_form(parameter, u, v)).array()

        A_csr = sparse.csr_matrix(A_assembled)
        print()
        print(f'Number of non-zeros in stiffness matrix: {A_csr.nnz}')
        print('Prestiffness matrix is '
                f'{prematrix.nnz / A_csr.nnz} times larger than stiffness matrix in sparse format')
        print()

        A_prematrix = prematrix.dot(sparse.csr_matrix(parameter_vec))

        # A_prestiffness is no longer sparse
        A_prematrix = np.reshape(A_prematrix, (dof_fe, dof_fe))

        max_diff = np.amax(abs(A_assembled - A_prematrix))
        print('max diff between direct assembly of stiffness mat '
                f'and using prestiffness: {max_diff}')

    return prematrix
