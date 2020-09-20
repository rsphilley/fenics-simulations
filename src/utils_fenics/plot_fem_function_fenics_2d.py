#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:47:02 2019

@author: hwan - Took out relevant code from dolfin's plotting.py _plot_matplotlib code
              - To enter dolfin's own plotting code, use dl.plot(some_dolfin_object) wheresome_dolfin_object is a 3D object and an error will be thrown up
"""
import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off

from utils_fenics.convert_array_to_dolfin_function import\
        convert_array_to_dolfin_function

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_fem_function_fenics_2d(function_space, nodal_values,
                                title, filepath,
                                fig_size):

    #=== Convert array to dolfin function ===#
    nodal_values_fe = convert_array_to_dolfin_function(function_space, nodal_values)

    #=== Plot figure ===#
    fig, ax = plot(nodal_values_fe, title, fig_size)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(p_test_fig, cax = cax)

    #=== Save figure ===#
    plt.savefig(filepath, dpi=300, bbox_inches = 'tight', pad_inches = 0)
    print('Figure saved to ' + filepath)

def plot(obj, title, fig_size):
    if hasattr(obj, "cpp_object"):
        obj = obj.cpp_object()
    plt.figure(figsize = fig_size)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title(title)
    return my_mplot_function(ax, obj), ax

def my_mplot_function(ax, f, **kwargs):
    mesh = f.function_space().mesh()
    C = f.compute_vertex_values(mesh)
    mode = kwargs.pop("mode", "contourf")
    if mode == "contourf":
        levels = kwargs.pop("levels", 40)
    return ax.tricontourf(my_mesh2triang(mesh), C, levels, **kwargs)

def my_mesh2triang(mesh):
    import matplotlib.tri as tri
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())
