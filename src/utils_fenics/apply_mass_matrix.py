import os

import numpy as np
import pandas as pd

import dolfin as dl

# Import src codes
from utils_io.load_parameters import load_parameters

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def apply_mass_matrix(options, filepaths, Vh, dof):

    #=== Load Parameters ===#
    parameters = load_parameters(filepaths, dof, options.num_data)

    #=== Form Mass Matrix ===#
    u = dl.TrialFunction(Vh)
    p = dl.TestFunction(Vh)
    mass_varf = dl.inner(u,p)*dl.dx
    mass = dl.assemble(mass_varf)

    #=== Multiply Mass Matrix ===#
    parameters = np.matmul(parameters, np.transpose(mass.array()))

    #=== Save Samples ===#
    if not os.path.exists(filepaths.directory_dataset):
        os.makedirs(filepaths.directory_dataset)
    df_parameters= pd.DataFrame({'parameters': parameters.flatten()})
    df_parameters.to_csv(filepaths.parameter + '.csv', index=False)

    print('Samples saved')
