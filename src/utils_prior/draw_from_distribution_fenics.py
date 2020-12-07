import numpy as np

import dolfin as dl
from hippylib import *

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Draw Samples                                #
###############################################################################
def draw_from_distribution_fenics(filepaths,
                                  Vh, prior, num_nodes,
                                  num_samples = 0):
    if num_samples == 0:
        samples=[]
    else:
        samples = np.zeros((num_samples, num_nodes))
        for n in range(0, num_samples):
            noise = dl.Vector()
            prior.init_vector(noise,"noise")
            parRandom.normal(1., noise)
            s_prior = dl.Vector()
            prior.init_vector(s_prior, 0)
            prior.sample(noise, s_prior)
            samples[n,:] = np.array(s_prior)
            print('Drawn: %d of %d' %(n, num_samples))
        print('Samples drawn')

    return samples
