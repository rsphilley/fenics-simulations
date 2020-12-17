#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:47:02 2019

@author: hwan
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off

from utils_fenics.convert_array_to_dolfin_function import\
        convert_array_to_dolfin_function

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_fem_function_fenics_1d(function_space, nodal_values,
                                title, filepath,
                                fig_size,
                                x_axis_limits, y_axis_limits):

    #=== Extract mesh and triangulate ===#
    mesh = function_space.mesh()
    coords = mesh.coordinates()
    elements = mesh.cells()

    #=== Plot figure ===#
    plt.figure(figsize = fig_size)
    ax = plt.gca()
    plt.title(title)
    plt.plot(coords, nodal_values, 'k-')
    plt.xlim(x_axis_limits)
    plt.ylim(y_axis_limits)
    plt.xlabel('x-coordinate')
    plt.ylabel('Value')

    #=== Save figure ===#
    plt.savefig(filepath, dpi=100, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
