#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""
import os

from utils_io.value_to_string import value_to_string

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                               Prior Strings                                 #
###############################################################################
def prior_string_AC(prior_type, mean, variance, corr):
    mean_string = value_to_string(mean)
    variance_string = value_to_string(variance)
    corr_string = value_to_string(corr)

    return '%s_%s_%s_%s'%(prior_type, mean_string, variance_string, corr_string)

def prior_string_matern(prior_type, kern_type, cov_length):
    cov_length_string = value_to_string(cov_length)

    return '%s_%s_%s'%(prior_type, kern_type, cov_length)

def prior_string_blp(prior_type, mean, gamma, delta):
    mean_string = value_to_string(mean)
    gamma_string = value_to_string(gamma)
    delta_string = value_to_string(delta)

    return '%s_%s_%s_%s'%(prior_type, mean_string, gamma_string, delta_string)

###############################################################################
#                                 FilePaths                                   #
###############################################################################
class FilePaths():
    def __init__(self, options):

        #=== Key Strings ===#
        project_name = 'screened_poisson_linear_neumann_2d_'
        data_options = 'n%d'%(options.num_nodes)
        self.directory_dataset =\
                '../../../datasets/fenics/screened_poisson_linear_neumann_2d/' + data_options + '/'

        #=== File Name Properties ===#
        if options.generate_train_data == True:
            train_or_test = 'train_'
        if options.generate_test_data == True:
            train_or_test = 'test_'

        #=== Prior Properties ===#
        if options.prior_type_AC == True:
            prior_string = prior_string_AC('AC',
                    options.prior_mean_AC,
                    options.prior_variance_AC,
                    options.prior_corr_AC)
        if options.prior_type_matern == True:
            prior_string = prior_string_matern('matern',
                    options.prior_kern_type,
                    options.prior_cov_length)
        if options.prior_type_blp == True:
            prior_string = prior_string_blp('blp',
                                            options.prior_mean_blp,
                                            options.prior_gamma_blp,
                                            options.prior_delta_blp)

        #=== Mesh ===#
        mesh_name = 'mesh_square_2D_n%d' %(options.num_nodes)
        mesh_directory = '../../../Datasets/Mesh/' + mesh_name + '/'
        self.mesh_nodes = mesh_directory + mesh_name + '_nodes.csv'
        self.mesh_elements = mesh_directory + mesh_name + '_elements.csv'

        #=== FEM Matrices ===#
        self.forward_matrix = self.directory_dataset +\
                'forward_matrix_' + data_options
        self.mass_matrix = self.directory_dataset +\
                'mass_matrix_' + data_options

        #=== Prior ===#
        self.prior_mean = self.directory_dataset +\
                'prior_mean_' + data_options + '_' + prior_string
        self.prior_covariance = self.directory_dataset +\
                'prior_covariance_' + data_options + '_' + prior_string
        self.prior_covariance_inverse = self.directory_dataset +\
                'prior_covariance_inverse_' + data_options + '_' + prior_string
        self.prior_covariance_cholesky = self.directory_dataset +\
                'prior_covariance_cholesky_' + data_options + '_' + prior_string
        self.prior_covariance_cholesky_inverse = self.directory_dataset +\
                'prior_covariance_cholesky_inverse_' + data_options + '_' + prior_string

        #=== Parameters ===#
        self.parameter = self.directory_dataset +\
                project_name + 'parameter_' + train_or_test +\
                'd%d_'%(options.num_data) + data_options + '_' + prior_string
        self.standard_gaussian = self.directory_dataset +\
                project_name + 'standard_gaussian_' + train_or_test +\
                'd%d_'%(options.num_data) + data_options + '_' + prior_string
        self.input_specific = self.directory_dataset +\
                project_name + 'parameter_specific_' + data_options

        #=== Solution ===#
        self.obs_indices = self.directory_dataset +\
                project_name + 'obs_indices_' +\
                'o%d_'%(options.num_obs_points) + data_options
        self.state_full = self.directory_dataset +\
                project_name + 'state_full_' + train_or_test +\
                'd%d_'%(options.num_data) + data_options + '_' + prior_string
        self.state_obs = self.directory_dataset +\
                project_name + 'state_obs_' + train_or_test +\
                'o%d_d%d_'%(options.num_obs_points, options.num_data) +\
                data_options + '_' + prior_string

        #=== Figures ==#
        self.directory_figures = 'Figures/'
        if not os.path.exists(self.directory_figures):
            os.makedirs(self.directory_figures)
