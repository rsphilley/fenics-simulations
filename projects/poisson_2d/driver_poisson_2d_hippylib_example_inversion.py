import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys
import os
sys.path.insert(0, os.path.realpath('../../src'))
sys.path.append('../')
import yaml
from attrdict import AttrDict

sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *
import math

# Import src code
from utils_mesh.construct_mesh_rectangular import construct_mesh
from utils_mesh.plot_mesh import plot_mesh
from utils_prior.bilaplacian_prior import construct_bilaplacian_prior
from utils_mesh.observation_points import load_observation_points
from utils_fenics.plot_fem_function_fenics_2d import plot_fem_function_fenics_2d
from utils_hippylib.pde_variational_problem_dirichlet import PDEVariationalProblem
from utils_fenics.plot_cross_section import plot_cross_section

# Import project utilities
from utils_project.filepaths import FilePaths

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                  Utilities                                  #
###############################################################################
def u_boundary(x, on_boundary):
    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def v_boundary(x, on_boundary):
    return on_boundary and ( x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)

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
    #=== Noise Options ===#
    noise_level = 0.01

    #=== Uncertainty Quantification Options ===#
    compute_trace = True

    #=== Plotting Options ===#
    use_hippylib_plotting = False
    use_my_plotting = True
    colourbar_limit_parameter = 4
    colourbar_limit_state = 2
    colourbar_limit_prior_variance = 1.16
    colourbar_limit_posterior_variance = 1.16
    cross_section_y_limit_min = 0.0
    cross_section_y_limit_max = 5.0

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
    if options.plot_mesh == True:
        mesh = fe_space.mesh()
        plot_mesh(filepaths,
                  (5,5), '',
                  (-1,1), (-1,1),
                  mesh.coordinates(), mesh.cells())

    #=== Hippylib Example Mesh ===#
    ndim = 2
    nx = 64
    ny = 64
    mesh = dl.UnitSquareMesh(nx, ny)

    #=== MPI ===#
    rank = dl.MPI.rank(mesh.mpi_comm())
    nproc = dl.MPI.size(mesh.mpi_comm())

    #=== Function Spaces ===#
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    ndofs = [Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()]
    if rank == 0:
        print (sep, "Set up the mesh and finite element spaces", sep)
        print ("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs))

    #=== Forcing Term ===#
    f = dl.Constant(0.0)

    #=== Boundary Conditions ===#
    u_bdr = dl.Expression("x[1]", element = Vh[STATE].ufl_element() )
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)

###############################################################################
#                            Prior and True Parameter                         #
###############################################################################
    #=== Prior ===#
    gamma = .1
    delta = .5

    theta0 = 2.
    theta1 = .5
    alpha  = math.pi/4

    anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
    anis_diff.set(theta0, theta1, alpha)

    prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, anis_diff, robin_bc=True )

    #=== True Parameter ===#
    mtrue = true_model(prior)

    #=== Plot Prior and True Parameter ===#
    if use_my_plotting == True:
        plot_fem_function_fenics_2d(Vh[PARAMETER], np.array(mtrue),
                                    '',
                                    filepaths.directory_figures + 'parameter_test.png',
                                    (5,5), (0,colourbar_limit_parameter))
    if use_hippylib_plotting == True:
        vmax_parameter = max(mtrue.max(), misfit.d.max())
        vmin_parameter = min(mtrue.min(), misfit.d.min())
        plt.figure(figsize=(9,3))
        nb.plot(dl.Function(Vh[PARAMETER], mtrue),
                mytitle="True Parameter", subplot_loc=121,
                vmin=vmin_parameter, vmax=vmax_parameter)
        nb.plot(dl.Function(Vh[PARAMETER], prior.mean),
                mytitle="Prior Mean", subplot_loc=122,
                vmin=vmin_parameter, vmax=vmax_parameter)
        plt.show()

###############################################################################
#                                  PDE Problem                                #
###############################################################################
    #=== Variational Form ===#
    def pde_varf(u,m,p):
        return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx
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
    # _, targets = load_observation_points(filepaths.obs_indices, Vh1)

    #=== Observation Points hIPPYLib ===#
    ntargets = 50
    ndim = 2
    targets_x = np.random.uniform(0.1,0.9, [ntargets] )
    targets_y = np.random.uniform(0.1,0.5, [ntargets] )
    targets = np.zeros([ntargets, ndim])
    targets[:,0] = targets_x
    targets[:,1] = targets_y
    print( "Number of observation points: {0}".format(ntargets) )

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

    #=== Plot True State and Observation Points ===#
    if use_my_plotting == True:
        plot_fem_function_fenics_2d(Vh[STATE], np.array(utrue),
                                    '',
                                    filepaths.directory_figures + 'state_test.png',
                                    (5,5), (0,1.2))
    if use_hippylib_plotting == True:
        vmax_state = max(utrue.max(), misfit.d.max())
        vmin_state = min(utrue.min(), misfit.d.min())
        plt.figure(figsize=(9,3))
        nb.plot(dl.Function(Vh[STATE], utrue),
                mytitle="True State", subplot_loc=121,
                vmin=vmin_state, vmax=vmax_state)
        nb.plot_pts(targets, misfit.d, mytitle="Observations", subplot_loc=122,
                vmin=vmin_state, vmax=vmax_state)
    print("Number of observation points: {0}".format(len(targets)))

###############################################################################
#                                Model and Solver                             #
###############################################################################
    #=== Form Model ===#
    model = Model(pde,prior, misfit)

    #=== Initial Guess ===#
    m0 = dl.interpolate(
            dl.Expression("sin(x[0])", element=Vh[PARAMETER].ufl_element() ), Vh[PARAMETER])

    #=== Test Gradient ===#
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

    x = solver.solve([None, m, None])

    #=== Print Solver Information ===#
    if rank == 0:
        if solver.converged:
            print( "\nConverged in ", solver.it, " iterations.")
        else:
            print( "\nNot Converged")

        print ("Termination reason: ", solver.termination_reasons[solver.reason])
        print ("Final gradient norm: ", solver.final_grad_norm)
        print ("Final cost: ", solver.final_cost)

    #=== Print Estimation ===#
    if use_my_plotting == True:
        plot_fem_function_fenics_2d(Vh[PARAMETER], np.array(x[PARAMETER]),
                                    '',
                                    filepaths.directory_figures + 'parameter_pred.png',
                                    (5,5), (0,colourbar_limit_parameter))
        plot_fem_function_fenics_2d(Vh[STATE], np.array(x[STATE]),
                                    '',
                                    filepaths.directory_figures + 'state_pred.png',
                                    (5,5), (0,colourbar_limit_state))
    if use_hippylib_plotting == True:
        plt.figure(figsize=(9,3))
        nb.plot(dl.Function(Vh[PARAMETER], x[PARAMETER]),
                subplot_loc=122,mytitle="Parameter Pred",
                vmin=vmin_parameter, vmax=vmax_parameter)
        nb.plot(dl.Function(Vh[PARAMETER], mtrue),
                mytitle="True Parameter", subplot_loc=121,
                vmin=vmin_parameter, vmax=vmax_parameter)
        # nb.plot(dl.Function(Vh[STATE], x[STATE]),
        #         subplot_loc=121,mytitle="State Pred",
        #         vmin=vmin_state, vmax=vmax_state)
        plt.show()

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

    d, U = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1, check=False)
    posterior = GaussianLRPosterior(prior, d, U)
    posterior.mean = x[PARAMETER]


    post_tr, prior_tr, corr_tr = posterior.trace(method="Randomized", r=200)
    if rank == 0:
        print ("Posterior trace {0:5e}; Prior trace {1:5e}; Correction trace {2:5e}"\
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

    #=== Plot Variance ===#
    if use_my_plotting == True:
        plot_fem_function_fenics_2d(Vh[PARAMETER], np.array(pr_pw_variance),
                                    '',
                                    filepaths.directory_figures + 'prior_variance.png',
                                    (5,5), (0,colourbar_limit_prior_variance))
        plot_fem_function_fenics_2d(Vh[PARAMETER], np.array(post_pw_variance),
                                    '',
                                    filepaths.directory_figures + 'posterior_covariance.png',
                                    (5,5), (0,colourbar_limit_posterior_variance))
    if use_hippylib_plotting == True:
        vmin = 0.74
        vmax = colourbar_limit_posterior_variance
        plt.figure(figsize=(9,3))
        nb.plot(dl.Function(Vh[PARAMETER], pr_pw_variance),
                mytitle="Prior Variance", subplot_loc=121, vmin=vmin, vmax=vmax)
        nb.plot(dl.Function(Vh[PARAMETER], post_pw_variance),
                mytitle="Posterior Variance", subplot_loc=122, vmin=vmin, vmax=vmax)
        plt.show()

    #=== Plot Cross-Section with Error Bounds ===#
    cross_section_y = 0.5
    plot_cross_section(Vh[PARAMETER],
                       np.exp(np.array(mtrue)),
                       np.exp(np.array(x[PARAMETER])), np.array(post_pw_variance),
                       (-1,1), cross_section_y,
                       '',
                       filepaths.directory_figures + 'parameter_cross_section.png',
                       (cross_section_y_limit_min,cross_section_y_limit_max))

    #=== Relative Error ===#
    relative_error = np.linalg.norm(
            np.exp(np.array(mtrue)) - np.exp(np.array(x[PARAMETER])), ord=2)/\
                    np.linalg.norm(np.exp(np.array(mtrue)), ord=2)
    print('Relative Error: %4f' %(relative_error))
