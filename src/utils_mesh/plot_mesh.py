# Obtained from:
# https://stackoverflow.com/questions/52202014/how-can-i-plot-2d-fem-results-using-matplotlib
import os

import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off
import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_mesh(filepaths,
              fig_size, plot_title,
              x_axis_limits, y_axis_limits,
              nodes, elements):

    #=== Creating Directory ===#
    if not os.path.exists(filepaths.directory_figures):
        os.makedirs(filepaths.directory_figures)

    #=== Nodes ===#
    nodes_x = nodes[:,0]
    nodes_y = nodes[:,1]

    #=== Observation Points ===#
    df_obs_indices = pd.read_csv(filepaths.obs_indices + '.csv')
    obs_indices = df_obs_indices.to_numpy().flatten()
    obs_coords = np.zeros((len(obs_indices), 2))
    for point in range(0, len(obs_indices)):
        obs_coords[point,:] = nodes[obs_indices[point],:]

    #=== Plot Mesh ===#
    plt.figure(figsize = fig_size)
    for element in elements:
        x = [nodes_x[element[i]] for i in range(len(element))]
        y = [nodes_y[element[i]] for i in range(len(element))]
        plt.fill(x, y, edgecolor='black', fill=False)

    for point in range(0, len(obs_indices)):
        plt.plot(obs_coords[point,0], obs_coords[point,1], 'r+', mew=5, ms=20)

    plt.xlim(x_axis_limits)
    plt.ylim(y_axis_limits)
    plt.savefig(filepaths.directory_figures + 'mesh_n%d'%(nodes.shape[0]),
                dpi=100, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
