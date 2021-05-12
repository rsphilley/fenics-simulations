import os

import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Draw Samples                                #
###############################################################################
def draw_from_distribution(filepaths,
                           mean, L, num_nodes,
                           positivity_constraint, k,
                           flag_save_standard_gaussian_draws,
                           num_samples = 0):
    if num_samples == 0:
        samples=[]
    else:
        samples = np.zeros((num_samples, num_nodes))
        if flag_save_standard_gaussian_draws == True:
            samples_standard_gaussian_draws = np.zeros((num_samples, num_nodes))
        for n in range(0, num_samples):
            #=== Draw Sample from Prior Distribution ===#
            normal_draw = np.random.normal(0, 1, num_nodes)
            if flag_save_standard_gaussian_draws == True:
                samples_standard_gaussian_draws[n,:] = normal_draw
            samples[n,:] = np.matmul(L, normal_draw) + mean.T
            samples[n,:] = positivity_constraint(samples[n,:],k)
            print('Drawn: %d of %d' %(n, num_samples))
        print('Samples drawn')

        #=== Save Samples ===#
        if not os.path.exists(filepaths.directory_dataset):
            os.makedirs(filepaths.directory_dataset)
        df_samples= pd.DataFrame({'samples': samples.flatten()})
        df_samples.to_csv(filepaths.parameter + '.csv', index=False)
        if flag_save_standard_gaussian_draws == True:
            df_samples= pd.DataFrame({'samples': samples_standard_gaussian_draws.flatten()})
            df_samples.to_csv(filepaths.standard_gaussian + '.csv', index=False)

        print('Samples saved')
