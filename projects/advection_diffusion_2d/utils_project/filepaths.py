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
def prior_string_blp(prior_type, mean, gamma, delta):
    mean_string = value_to_string(mean)
    gamma_string = value_to_string(gamma)
    delta_string = value_to_string(delta)

    return '%s_%s_%s_%s'%(prior_type, mean_string, gamma_string, delta_string)

def prior_string_ac(prior_type, mean, variance, corr):
    mean_string = value_to_string(mean)
    variance_string = value_to_string(variance)
    corr_string = value_to_string(corr)

    return '%s_%s_%s_%s'%(prior_type, mean_string, variance_string, corr_string)

###############################################################################
#                                 FilePaths                                   #
###############################################################################
class FilePaths():
    def __init__(self, options):

        #=== Key Strings ===#
        project_name = 'advection_diffusion_2d_'
        if options.flow_navier_stokes == True:
            flow_string = 'navier_stokes'
        if options.flow_darcy == True:
            flow_string = 'darcy'
        if options.time_stepping_erk4 == True:
            time_stepping_string = 'erk4'
        if options.time_stepping_lserk4 == True:
            time_stepping_string = 'lserk4'
        if options.time_stepping_implicit == True:
            time_stepping_string = 'imp'
        num_nodes_string = 'n%d'%(options.num_nodes)
        data_options = num_nodes_string + '_' +\
                       flow_string + '_' +\
                       time_stepping_string
        self.directory_dataset = '../../../datasets/fenics/advection_diffusion_2d/' +\
            num_nodes_string + '/' + flow_string + '_' + time_stepping_string + '/'

        #=== File Name Properties ===#
        if options.generate_train_data == True:
            train_or_test = 'train_'
        if options.generate_test_data == True:
            train_or_test = 'test_'

        #=== Prior Properties ===#
        if options.prior_type_blp == True:
            prior_string = prior_string_blp('blp',
                                            options.prior_mean_blp,
                                            options.prior_gamma_blp,
                                            options.prior_delta_blp)
        if options.prior_type_ac == True:
            prior_string = prior_string_ac('ac',
                    options.prior_mean_ac,
                    options.prior_variance_ac,
                    options.prior_corr_ac)

        #=== Prior ===#
        self.prior_mean = self.directory_dataset +\
                'prior_mean_' + num_nodes_string + '_' + prior_string
        self.prior_covariance = self.directory_dataset +\
                'prior_covariance_' + num_nodes_string + '_' + prior_string
        self.prior_covariance_inverse = self.directory_dataset +\
                'prior_covariance_inverse_' + data_options + '_' + prior_string
        self.prior_covariance_cholesky = self.directory_dataset +\
                'prior_covariance_cholesky_' + num_nodes_string + '_' + prior_string
        self.prior_covariance_cholesky_inverse = self.directory_dataset +\
                'prior_covariance_cholesky_inverse_' + num_nodes_string + '_' + prior_string

        #=== FEM Operators ===#
        self.fem_operator_spatial = self.directory_dataset +\
                'fem_operator_spatial_' + num_nodes_string
        self.fem_operator_implicit_ts = self.directory_dataset +\
                'fem_operator_implicit_ts_' + num_nodes_string
        self.fem_operator_implicit_ts_rhs = self.directory_dataset +\
                'fem_operator_implicit_ts_rhs_' + num_nodes_string

        #=== Parameters ===#
        self.parameter = self.directory_dataset +\
                project_name + 'parameter_' + train_or_test +\
                'd%d_'%(options.num_data) + num_nodes_string + '_' + prior_string
        self.standard_gaussian = self.directory_dataset +\
                project_name + 'standard_gaussian_' + train_or_test +\
                'd%d_'%(options.num_data) + data_options + '_' + prior_string
        self.input_specific = self.directory_dataset +\
                project_name + 'parameter_specific_' + data_options
        self.parameter_blob = self.directory_dataset +\
                project_name + 'parameter_blob_' + num_nodes_string

        #=== Solution ===#
        self.obs_indices = self.directory_dataset +\
                project_name + 'obs_indices_' +\
                'o%d_'%(options.num_obs_points) + num_nodes_string
        self.state_full = self.directory_dataset +\
                project_name + 'state_full_' + train_or_test +\
                'd%d_'%(options.num_data) + data_options + '_' + prior_string
        self.state_obs = self.directory_dataset +\
                project_name + 'state_obs_' + train_or_test +\
                'o%d_d%d_'%(options.num_obs_points, options.num_data) +\
                data_options + '_' + prior_string
        self.output_specific = self.directory_dataset +\
                project_name + 'state_obs_specific_' +\
                'o%d_'%(options.num_obs_points) +\
                data_options
        self.state_obs_blob = self.directory_dataset +\
                        project_name + 'state_obs_blob_' +\
                        'o%d_'%(options.num_obs_points) +\
                        data_options

        #=== Figures ==#
        self.directory_figures = 'Figures/'
        if not os.path.exists(self.directory_figures):
            os.makedirs(self.directory_figures)
