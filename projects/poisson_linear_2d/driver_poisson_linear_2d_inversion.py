import sys
import os
sys.path.insert(0, os.path.realpath('../../src'))
sys.path.append('../')
import yaml
from attrdict import AttrDict

import numpy as np
import pandas as pd
import time

import dolfin as dl
import ufl
import matplotlib.pyplot as plt
import argparse

sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *
import math

# Import src code
from utils_mesh.construct_mesh_rectangular import construct_mesh
from utils_mesh.plot_mesh import plot_mesh
from utils_prior.bilaplacian_prior import construct_bilaplacian_prior
from utils_fenics.convert_array_to_dolfin_function import convert_array_to_dolfin_function
from utils_mesh.observation_points import load_observation_points
from utils_fenics.plot_fem_function_fenics_2d import plot_fem_function_fenics_2d
from utils_hippylib.pde_varf_poisson_heat_source import pde_varf
from utils_hippylib.pde_variational_problem_heat import PDEVariationalProblem
from utils_fenics.plot_cross_section import plot_cross_section

# Import project utilities
from utils_project.filepaths import FilePaths

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                  Utilities                                  #
###############################################################################
def true_model(prior):
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    parRandom.normal(1., noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise,mtrue)
    return mtrue

if __name__ == "__main__":

###############################################################################
#                              Inversion Options                              #
###############################################################################
    #=== True Parameter Options ===#
    true_parameter_prior = True
    true_parameter_specific = False

    #=== Prior Options ===#
    prior_scalar_yaml = True
    prior_scalar_set = False
    prior_scalar_value = 0

    #=== Noise Options ===#
    noise_level = 0.05

    #=== Uncertainty Quantification Options ===#
    compute_trace = False

    #=== Plotting Options ===#
    colourbar_limit_parameter_min = -4
    colourbar_limit_parameter_max = 4
    colourbar_limit_state = 4
    colourbar_limit_prior_variance = 1.2
    colourbar_limit_posterior_variance = 1.2
    cross_section_y = 0.0
    cross_section_y_limit_min = -4
    cross_section_y_limit_max = 4

###############################################################################
#                                  Setting Up                                 #
###############################################################################
    #=== Separation for Print Statements ===#
    sep = "\n"+"#"*80+"\n"

    #=== Options ===#
    with open('config_files/options.yaml') as f:
        options = yaml.safe_load(f)
    options = AttrDict(options)
    options.num_nodes = (options.nx + 1) * (options.ny + 1)

    #=== File Paths ===#
    filepaths = FilePaths(options)

    #=== Construct Mesh ===#
    fe_space, meta_space,\
    nodes, dof_fe, dof_meta = construct_mesh(options)

    #=== Plot Mesh ===#
    mesh = fe_space.mesh()
    if options.plot_mesh == True:
        plot_mesh(filepaths,
                  (5,5), '',
                  (-1,1), (-1,1),
                  mesh.coordinates(), mesh.cells())

    #=== MPI ===#
    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())

    #=== Function Spaces ===#
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
    if rank == 0:
        print (sep, "Set up the mesh and finite element spaces", sep)
        print ("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs))

###############################################################################
#                            Prior and True Parameter                         #
###############################################################################
    #=== Prior ===#
    if prior_scalar_yaml == True:
        if options.prior_mean_blp == 0:
            mean_array = 0*np.ones(Vh[PARAMETER].dim())
        else:
            mean_array = options.prior_mean_blp*np.ones(Vh[PARAMETER].dim())
    if prior_scalar_set == True:
        if prior_scalar_value == 0:
            mean_array = 0*np.ones(Vh[PARAMETER].dim())
        else:
            mean_array = prior_scalar_value*np.ones(Vh[PARAMETER].dim())
    mean_dl = convert_array_to_dolfin_function(Vh[PARAMETER], mean_array)
    mean = mean_dl.vector()
    prior = BiLaplacianPrior(Vh[PARAMETER],
                             options.prior_gamma_blp,
                             options.prior_delta_blp,
                             mean = mean,
                             robin_bc=True)

    #=== True Parameter ===#
    if true_parameter_prior == True:
        mtrue = true_model(prior)
        mtrue_dl = convert_array_to_dolfin_function(Vh[PARAMETER], np.array(mtrue))
        mtrue = mtrue_dl.vector()
    if true_parameter_specific == True:
        df_mtrue = pd.read_csv(filepaths.input_specific + '.csv')
        mtrue_array = df_mtrue.to_numpy()
        mtrue_dl = convert_array_to_dolfin_function(Vh[PARAMETER], mtrue_array)
        mtrue = mtrue_dl.vector()

    #=== Plot Estimation ===#
    plot_fem_function_fenics_2d(Vh[PARAMETER], np.array(mtrue),
                                '',
                                filepaths.directory_figures + 'parameter_test.png',
                                (5,5),
                                (colourbar_limit_parameter_min,colourbar_limit_parameter_max))

###############################################################################
#                                  PDE Problem                                #
###############################################################################
    #=== Variational Form ===#
    pde = PDEVariationalProblem(options, Vh, pde_varf, is_fwd_linear=True)

    #=== PDE Solver ===#
    pde.solver = PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
    pde.solver_fwd_inc = PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())
    pde.solver_adj_inc = PETScKrylovSolver(mesh.mpi_comm(), "cg", amg_method())

    pde.solver.parameters["relative_tolerance"] = 1e-15
    pde.solver.parameters["absolute_tolerance"] = 1e-20
    pde.solver_fwd_inc.parameters = pde.solver.parameters
    pde.solver_adj_inc.parameters = pde.solver.parameters

    #=== Observation Points ===#
    _, targets = load_observation_points(filepaths.obs_indices, Vh1)

    #=== Misfit Functional ===#
    misfit = PointwiseStateObservation(Vh[STATE], targets)

###############################################################################
#                        Generate Synthetic Observations                      #
###############################################################################
    #=== Generate True State and Observations ===#
    utrue = pde.generate_state()
    x = [utrue, mtrue, None]
    pde.solveFwd(x[STATE], x)
    misfit.B.mult(x[STATE], misfit.d)

    #=== Noise Model ===#
    rel_noise = noise_level
    MAX = misfit.d.norm("linf")
    noise_std_dev = rel_noise * MAX
    parRandom.normal_perturb(noise_std_dev, misfit.d)
    misfit.noise_variance = noise_std_dev*noise_std_dev

    #=== Plot True State ===#
    plot_fem_function_fenics_2d(Vh[STATE], np.array(utrue),
                                '',
                                filepaths.directory_figures + 'state_test.png',
                                (5,5), (0,colourbar_limit_state))

###############################################################################
#                               Model and Solver                              #
###############################################################################
    #=== Form Model ===#
    model = Model(pde, prior, misfit)

    #=== Initial Guess ===#
    m0_array = options.prior_mean_blp*np.ones(Vh[PARAMETER].dim())
    m0 = convert_array_to_dolfin_function(Vh[PARAMETER], m0_array)

    #=== Perform Gradient and Hessian Test ===#
    if rank == 0:
        print( sep, "Test the gradient and the Hessian of the model", sep )
    modelVerify(model, m0.vector(), is_quadratic = False, verbose = (rank == 0) )

    #=== Evaluate Gradient ===#
    [u,m,p] = model.generate_vector()
    model.solveFwd(u, [u,m,p])
    model.solveAdj(p, [u,m,p])
    mg = model.generate_vector(PARAMETER)
    grad_norm = model.evalGradientParameter([u,m,p], mg)

    #=== Compute Gaussian Posterior ===#
    H = ReducedHessian(model, misfit_only=True)
    k = 80
    p = 20
    print( "Single Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )
    Omega = MultiVector(x[PARAMETER], k+p)
    parRandom.normal(1., Omega)
    lmbda, V = doublePassG(H, prior.R, prior.Rsolver, Omega, k)
    posterior = GaussianLRPosterior(prior, lmbda, V)

    #=== Compute MAP Point ===#
    H.misfit_only = False

    solver = CGSolverSteihaug()
    solver.set_operator(H)
    solver.set_preconditioner( posterior.Hlr )
    solver.parameters["print_level"] = 1
    solver.parameters["rel_tolerance"] = 1e-6
    solver.solve(m, -mg)
    model.solveFwd(u, [u,m,p])

    total_cost, reg_cost, misfit_cost = model.cost([u,m,p])
    print("Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}"\
                    .format(total_cost, reg_cost, misfit_cost))

    posterior.mean = m

    #=== Plot Estimation ===#
    plot_fem_function_fenics_2d(Vh[PARAMETER], np.array(m),
                                '',
                                filepaths.directory_figures + 'parameter_pred.png',
                                (5,5),
                                (colourbar_limit_parameter_min,colourbar_limit_parameter_max))

###############################################################################
#                          Uncertainty Quantification                         #
###############################################################################
    #=== Compute Trace ===#
    if compute_trace:
        post_tr, prior_tr, corr_tr = posterior.trace(method="Randomized", r=300)
        print("Posterior trace {0:5g}; Prior trace {1:5g}; Correction trace {2:5g}"\
                        .format(post_tr, prior_tr, corr_tr))

    #=== Compute Variances ===#
    post_pw_variance, pr_pw_variance, corr_pw_variance =\
            posterior.pointwise_variance(method="Randomized", r=300)

    #=== Plot Variances ===#
    plot_fem_function_fenics_2d(Vh[PARAMETER], np.array(pr_pw_variance),
                                '',
                                filepaths.directory_figures + 'prior_variance.png',
                                (5,5), (0,colourbar_limit_prior_variance))
    plot_fem_function_fenics_2d(Vh[PARAMETER], np.array(post_pw_variance),
                                '',
                                filepaths.directory_figures + 'posterior_covariance.png',
                                (5,5), (0,colourbar_limit_posterior_variance))

    #=== Plot Cross-Section with Error Bounds ===#
    plot_cross_section(Vh[PARAMETER],
                       np.array(mtrue),
                       np.array(m), np.array(post_pw_variance),
                       (-1,1), cross_section_y,
                       '',
                       filepaths.directory_figures + 'parameter_cross_section.png',
                       (cross_section_y_limit_min,cross_section_y_limit_max))

    #=== Relative Error ===#
    relative_error = np.linalg.norm(
            np.array(mtrue) - np.array(m), ord=2)/\
                    np.linalg.norm(np.array(mtrue), ord=2)
    print('Relative Error: %4f' %(relative_error))
