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
from utils_mesh.construct_mesh_1d import construct_mesh
from utils_prior.laplacian_prior import construct_laplacian_prior
from utils_fenics.convert_array_to_dolfin_function import convert_array_to_dolfin_function
from utils_mesh.observation_points import load_observation_points
from utils_fenics.plot_fem_function_fenics_2d import plot_fem_function_fenics_2d
from utils_fenics.plot_cross_section import plot_cross_section

# Import project utilities
from utils_project.filepaths import FilePaths

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                  Utilities                                  #
###############################################################################
def u_boundary(x, on_boundary):
    return on_boundary and (x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

def pde_varf(u, m, p):
    return dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx + u*p*dl.dx - m*p*dl.dx

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
    noise_level = 0.01

    #=== Uncertainty Quantification Options ===#
    compute_trace = False

###############################################################################
#                                  Setting Up                                 #
###############################################################################
    #=== Separation for Print Statements ===#
    sep = "\n"+"#"*80+"\n"

    #=== Options ===#
    with open('config_files/options.yaml') as f:
        options = yaml.safe_load(f)
    options = AttrDict(options)
    options.num_nodes = options.nx + 1

    #=== File Paths ===#
    filepaths = FilePaths(options)

    #=== Construct Mesh ===#
    Vh, nodes, dof = construct_mesh(options)
    mesh = Vh.mesh()

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
        if options.prior_mean_lp == 0:
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
    prior = construct_laplacian_prior(filepaths,
                                      Vh[PARAMETER], options.prior_mean_lp,
                                      options.prior_gamma_lp,
                                      options.prior_delta_lp)

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

###############################################################################
#                                  PDE Problem                                #
###############################################################################
    #=== Variational Form ===#
    u_bdr = dl.Expression("1-x[0]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)
    pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

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

###############################################################################
#                               Model and Solver                              #
###############################################################################
    #=== Form Model ===#
    model = Model(pde, prior, misfit)

    #=== Initial Guess ===#
    m0_array = options.prior_mean_lp*np.ones(Vh[PARAMETER].dim())
    m0 = convert_array_to_dolfin_function(Vh[PARAMETER], m0_array)

    #=== Perform Gradient and Hessian Test ===#
    if rank == 0:
        print( sep, "Test the gradient and the Hessian of the model", sep )
    modelVerify(model, m0.vector(), is_quadratic = False, verbose = (rank == 0) )

    #=== Solver Parameters ===#
    if rank == 0:
        print( sep, "Find the MAP point", sep)
    m = prior.mean.copy()
    parameters = ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = 1e-9
    parameters["abs_tolerance"] = 1e-12
    parameters["max_iter"]      = 25
    parameters["globalization"] = "LS"
    parameters["GN_iter"] = 5
    if rank != 0:
        parameters["print_level"] = -1

    #=== Solve ===#
    if rank == 0:
        parameters.showMe()
    solver = ReducedSpaceNewtonCG(model, parameters)

    start_time_solver = time.time()
    x = solver.solve([None, m, None])
    elapsed_time_solver = time.time() - start_time_solver
    print('Time taken to solve: %.4f' %(elapsed_time_solver))

    #=== Print Solver Information ===#
    if rank == 0:
        if solver.converged:
            print( "\nConverged in ", solver.it, " iterations.")
        else:
            print( "\nNot Converged")

        print ("Termination reason: ", solver.termination_reasons[solver.reason])
        print ("Final gradient norm: ", solver.final_grad_norm)
        print ("Final cost: ", solver.final_cost)

###############################################################################
#                           Uncertainty Quantification                        #
###############################################################################
    if rank == 0:
        print (sep, "Compute the low rank Gaussian Approximation of the posterior", sep)

    model.setPointForHessianEvaluations(x, gauss_newton_approx = False)
    Hmisfit = ReducedHessian(model, misfit_only=True)
    k = 50
    p = 20
    if rank == 0:
        print ("Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )

    Omega = MultiVector(x[PARAMETER], k+p)
    parRandom.normal(1., Omega)

    start_time_laplace = time.time()
    d, U = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
    posterior = GaussianLRPosterior(prior, d, U)
    posterior.mean = x[PARAMETER]
    elapsed_time_laplace = time.time() - start_time_laplace
    print('Time taken to form Laplacian approximation: %.4f' %(elapsed_time_laplace))

    post_tr, prior_tr, corr_tr = posterior.trace(method="Randomized", r=200)
    if rank == 0:
        print("Posterior trace {0:5e}; Prior trace {1:5e}; Correction trace {2:5e}"\
                .format(post_tr, prior_tr, corr_tr))

    post_pw_variance, pr_pw_variance, corr_pw_variance =\
            posterior.pointwise_variance(method="Randomized", r=200)

    kl_dist = posterior.klDistanceFromPrior()
    if rank == 0:
        print ("KL-Distance from prior: ", kl_dist)

    with dl.XDMFFile(mesh.mpi_comm(), "results/pointwise_variance.xdmf") as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False

        fid.write(vector2Function(post_pw_variance, Vh[PARAMETER], name="Posterior"), 0)
        fid.write(vector2Function(pr_pw_variance, Vh[PARAMETER], name="Prior"), 0)
        fid.write(vector2Function(corr_pw_variance, Vh[PARAMETER], name="Correction"), 0)

    if rank == 0:
        print (sep, "Save State, Parameter, Adjoint, and observation in paraview", sep)
    xxname = ["state", "parameter", "adjoint"]
    xx = [vector2Function(x[i], Vh[i], name=xxname[i]) for i in range(len(Vh))]

    with dl.XDMFFile(mesh.mpi_comm(), "results/results.xdmf") as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False

        fid.write(xx[STATE],0)
        fid.write(vector2Function(utrue, Vh[STATE], name = "true state"), 0)
        fid.write(xx[PARAMETER],0)
        fid.write(vector2Function(mtrue, Vh[PARAMETER], name = "true parameter"), 0)
        fid.write(vector2Function(prior.mean, Vh[PARAMETER], name = "prior mean"), 0)
        fid.write(xx[ADJOINT],0)

    exportPointwiseObservation(Vh[STATE], misfit.B, misfit.d, "results/poisson_observation")

    if rank == 0:
        print(sep,
              "Generate samples from Prior and Posterior\n","Export generalized Eigenpairs",
              sep)

    nsamples = 50
    noise = dl.Vector()
    posterior.init_vector(noise,"noise")
    s_prior = dl.Function(Vh[PARAMETER], name="sample_prior")
    s_post = dl.Function(Vh[PARAMETER], name="sample_post")
    with dl.XDMFFile(mesh.mpi_comm(), "results/samples.xdmf") as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False
        for i in range(nsamples):
            parRandom.normal(1., noise)
            posterior.sample(noise, s_prior.vector(), s_post.vector())
            fid.write(s_prior, i)
            fid.write(s_post, i)

    #=== Save eigenvalues for printing ===#
    U.export(Vh[PARAMETER], "results/evect.xdmf", varname = "gen_evects", normalize = True)
    if rank == 0:
        np.savetxt("results/eigevalues.dat", d)

    if rank == 0:
        plt.figure()
        plt.plot(range(0,k), d, 'b*', range(0,k), np.ones(k), '-r')
        plt.yscale('log')
        plt.show()

    #=== Compute Trace ===#
    if compute_trace:
        post_tr, prior_tr, corr_tr = posterior.trace(method="Randomized", r=200)
        print("Posterior trace {0:5e}; Prior trace {1:5e}; Correction trace {2:5e}"\
                .format(post_tr, prior_tr, corr_tr) )

    #=== Compute Variances ===#
    post_pw_variance, pr_pw_variance, corr_pw_variance =\
            posterior.pointwise_variance(method="Randomized", r=200)

    #=== Relative Error ===#
    relative_error = np.linalg.norm(
            np.array(mtrue) - np.array(x[PARAMETER]), ord=2)/\
                    np.linalg.norm(np.array(mtrue), ord=2)
    print('Relative Error: %4f' %(relative_error))
