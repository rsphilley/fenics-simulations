#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 21:41:12 2020
@author: hwan
"""
import sys
import os
sys.path.insert(0, os.path.realpath('../../src'))
sys.path.append('../')

import yaml
from attrdict import AttrDict
import scipy.sparse as sparse

# Import src code
from utils_io.load_dataset import load_dataset

# Import project utilities
from utils_project.filepaths import FilePaths
from utils_project.plot_paraview import plot_paraview

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":

    ##################
    #   Setting Up   #
    ##################
    #=== Options ===#
    with open('config_files/options.yaml') as f:
        options = yaml.safe_load(f)
    options = AttrDict(options)
    options.num_nodes = (options.num_nodes_x + 1) * (options.num_nodes_y + 1)\
                        *(options.num_nodes_z + 1)

    #=== File Paths ===#
    filepaths = FilePaths(options)

    #=== Load Parameters and State ===#
    parameters = load_dataset(filepaths.parameter, options.num_nodes, options.num_data)
    state = load_dataset(filepaths.state_full, options.num_nodes, options.num_data)

    #=== Colourbar Scale ===#
    cbar_RGB_parameter = [1.0576069802911363, 0.231373, 0.298039, 0.752941,
            3.6660661539477166, 0.865003, 0.865003, 0.865003, 6.274525327604296,
            0.705882, 0.0156863, 0.14902]
    cbar_RGB_state = [0.335898315086773, 0.231373, 0.298039, 0.752941,
            0.6068782815115794, 0.865003, 0.865003, 0.865003, 0.8778582479363859,
            0.705882, 0.0156863, 0.14902]

    #=== Plot and Save Paraview Figures ===#
    for n in range(0, options.num_data):
        plot_paraview(filepaths.figure_vtk_parameter + '_%d.pvd'%(n),
                      filepaths.figure_paraview_parameter + '_%d.png'%(n),
                      cbar_RGB_parameter)
        plot_paraview(filepaths.figure_vtk_state + '_%d.pvd'%(n),
                      filepaths.figure_paraview_state + '_%d.png'%(n),
                      cbar_RGB_state)
