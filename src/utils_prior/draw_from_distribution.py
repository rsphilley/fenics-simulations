import os

import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Draw Samples                                #
###############################################################################
def draw_from_distribution(filepaths,
                           mean, L, num_nodes,
                           num_samples = 0):

    if num_samples == 0:
        samples=[]
    else:
        samples = np.zeros((num_samples, num_nodes))
        k = 0.5;
        for n in range(0, num_samples):
            #=== Draw Sample from Prior Distribution ===#
            normal_draw = np.random.normal(0, 1, num_nodes)
            samples[n,:] = np.matmul(L, normal_draw) + mean.T
            # samples[n,:] = (1/k)*np.log(np.exp(k*samples[n,:])+1);
            samples[n,:] = np.exp(samples[n,:])
            # samples[samples<0] = 0 #Positivity constraint
            print('Drawn: %d of %d' %(n, num_samples))
        print('Samples drawn')

        #=== Save Samples ===#
        if not os.path.exists(filepaths.directory_dataset):
            os.makedirs(filepaths.directory_dataset)
        df_samples= pd.DataFrame({'samples': samples.flatten()})
        df_samples.to_csv(filepaths.parameter + '.csv', index=False)

        print('Samples saved')
