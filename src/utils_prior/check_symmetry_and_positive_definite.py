import numpy as np

def check_symmetry_and_positive_definite(A):
    epsilon = 1e-9
    if np.allclose(A, A.T, rtol = epsilon, atol = epsilon):
        print("Matrix is symmetric")
        try:
            np.linalg.cholesky(A)
            print("Matrix is positive definite")
        except np.linalg.LinAlgError:
            print("Matrix is not positive definite\n")
    else:
        print("Matrix is not symmetric")
        print("Max. asymm. value: ", np.max(np.abs((A - A.T)/2)))
