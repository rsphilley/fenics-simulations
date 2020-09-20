import pandas as pd
from scipy import sparse

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def load_prematrices(filepaths, num_nodes):

    premass = sparse.load_npz(filepaths.premass + '.npz')
    prestiffness = sparse.load_npz(filepaths.prestiffness + '.npz')

    return premass, prestiffness

def load_boundary_matrices_and_load_vector(filepaths, num_nodes):

    boundary_matrix = sparse.load_npz(filepaths.boundary_matrix + '.npz')
    load_vector = sparse.load_npz(filepaths.load_vector + '.npz')

    return boundary_matrix, load_vector.T
