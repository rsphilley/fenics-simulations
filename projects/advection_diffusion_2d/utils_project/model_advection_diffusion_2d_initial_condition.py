import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt
import argparse
from hippylib import *

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class TimeDependentAdvectionDiffusionInitialCondition:
    def __init__(self, options,
                 mesh, Vh,
                 prior, misfit,
                 simulation_times, velocity,
                 gls_stab):

        '''Initialize a time-dependent advection diffusion problem with the
           initial condition as the parameter

           Inputs:
               mesh - mesh generated using mshr
               Vh - tuple of three fenics function spaces for state, parameter,
                    and adjoint variables
               prior - hippylib prior for regularization of the inverse problem
               misfit - misfit class which models observations with the ability
                    to evaluate gradients
               simulation_times - array describing simulation time steps
               velocity - velocity function for advection-diffusion
               u_0 - initial condition of concentrate
               gls_stab - Set true to turn on Galerkin Least-Squares stabilization.
                   Currently unsupported
        '''
        # Set member variables describing the problem
        self.mesh = mesh
        self.Vh = Vh
        self.prior = prior
        self.misfit = misfit

        # Assume constant timestepping
        self.simulation_times = simulation_times
        dt = options.time_dt_imp
        dt_expr = dl.Constant(dt)

        # Trial and Test functions for the weak forms
        u_trial = dl.TrialFunction(Vh[STATE])
        p_trial = dl.TrialFunction(Vh[ADJOINT])
        u_test = dl.TestFunction(Vh[STATE])
        p_test = dl.TestFunction(Vh[ADJOINT])

        # Functions to be populated for time stepping
        self.u_old = dl.Function(Vh[STATE])
        self.p_old = dl.Function(Vh[ADJOINT])

        # Diffusion coefficient
        self.kappa = dl.Constant(options.kappa)

        # Galerkin Least Squares stabilization terms
        r_trial = u_trial + dt_expr*(-ufl.div(self.kappa*ufl.grad(u_trial)) +\
                  ufl.inner(velocity, ufl.grad(u_trial)))
        r_test  = u_test + dt_expr*(-ufl.div(self.kappa*ufl.grad(u_test)) +\
                  ufl.inner(velocity, ufl.grad(u_test)))

        h = dl.CellDiameter(mesh)
        vnorm = ufl.sqrt(ufl.inner(velocity, velocity))
        if gls_stab:
            tau = ufl.min_value((h*h)/(dl.Constant(2.)*self.kappa), h/vnorm )
        else:
            tau = dl.Constant(0.)

        # Mass matrix variational forms and their assembled matrices
        M_varf = ufl.inner(u_trial, u_test)*ufl.dx
        self.M = dl.assemble(M_varf)

        M_stab_varf = ufl.inner(u_trial, u_test + tau * r_test) * ufl.dx
        self.M_stab = dl.assemble(M_stab_varf)

        Mt_stab_varf = ufl.inner(u_trial + tau * r_trial, u_test) * ufl.dx
        self.Mt_stab = dl.assemble(Mt_stab_varf)

        # Variational form for time-stepping
        N_varf = (ufl.inner(self.kappa * ufl.grad(u_trial), ufl.grad(u_test)) \
                + ufl.inner(velocity, ufl.grad(u_trial)) * u_test) * ufl.dx
        self.N = dl.assemble(N_varf)

        Nt_varf = (ufl.inner(self.kappa * ufl.grad(p_test), ufl.grad(p_trial)) \
                + ufl.inner(velocity, ufl.grad(p_test)) * p_trial) * ufl.dx
        self.Nt = dl.assemble(Nt_varf)

        stab_varf = tau*ufl.inner(r_trial, r_test) * ufl.dx
        self.stab = dl.assemble(stab_varf)

        # Implicit Time-Stepping: LHS variational form to be solved at each time step
        self.L_varf = M_varf + dt_expr * N_varf + stab_varf
        self.L = dl.assemble(self.L_varf)

        self.L_rhs_varf = ufl.inner(u_trial, u_test) * ufl.dx
        self.L_rhs = dl.assemble(self.L_rhs_varf)

        self.Lt_varf = M_varf + dt_expr * Nt_varf + stab_varf
        self.Lt = dl.assemble(self.Lt_varf)

        self.Lt_rhs_varf = ufl.inner(p_trial, p_test) * ufl.dx
        self.Lt_rhs = dl.assemble(self.Lt_rhs_varf)

        # Solvers
        self.solver = PETScLUSolver(self.mesh.mpi_comm())
        self.solver.set_operator(dl.as_backend_type(self.L))
        self.solvert = PETScLUSolver(self.mesh.mpi_comm())
        self.solvert.set_operator(dl.as_backend_type(self.Lt))

        # Part of model public API for hippylib
        self.gauss_newton_approx = False

        # Setup variational forms for gradient evaluation
        self.solved_u = dl.Function(Vh[STATE])
        self.solved_p = dl.Function(Vh[ADJOINT])
        self.solved_u_tilde = dl.Function(Vh[STATE])
        self.solved_p_tilde = dl.Function(Vh[ADJOINT])

    def generate_vector(self, component = "ALL"):
        '''Generates an appropriately initialized PETSc vector for appropriate variables'''
        if component == "ALL":
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.M, 0)
            m = dl.Vector()
            self.prior.init_vector(m,0)
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.M, 0)
            return [u, m, p]
        elif component == STATE:
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.M, 0)
            return u
        elif component == PARAMETER:
            m = dl.Vector()
            self.prior.init_vector(m,0)
            return m
        elif component == ADJOINT:
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.M, 0)
            return p
        else:
            raise

    def init_parameter(self, m):
        '''Initialize parameter to be compatible with the prior'''
        self.prior.init_vector(m,0)

    def cost(self, x):
        '''Evaluate the cost functional to be optimized for the inverse problem'''
        Rdx = dl.Vector()
        self.prior.init_vector(Rdx,0)
        dx = x[PARAMETER] - self.prior.mean
        self.prior.R.mult(dx, Rdx)
        reg = .5*Rdx.inner(dx)

        misfit = self.misfit.cost(x)

        return [reg+misfit, reg, misfit]

    def solveFwd(self, out, x):
        '''Perform implicit time-stepping and solve the forward problem '''
        out.zero()
        uold = x[PARAMETER]
        u = dl.Vector()
        rhs = dl.Vector()
        self.M.init_vector(rhs, 0)
        self.M.init_vector(u, 0)
        for t in self.simulation_times[1::]:
            pdb.set_trace()
            self.M_stab.mult(uold, rhs)
            self.solver.solve(u, rhs)
            out.store(u,t)
            uold = u

    def solveAdj(self, out, x):
        '''Solve adjoint problem backwards in time and store in out '''
        grad_state = TimeDependentVector(self.simulation_times)
        grad_state.initialize(self.M, 0)
        self.misfit.grad(STATE, x, grad_state)

        out.zero()

        pold = dl.Vector()
        self.M.init_vector(pold,0)

        p = dl.Vector()
        self.M.init_vector(p,0)

        rhs = dl.Vector()
        self.M.init_vector(rhs,0)

        grad_state_snap = dl.Vector()
        self.M.init_vector(grad_state_snap,0)

        for t in self.simulation_times[::-1]:
            self.Mt_stab.mult(pold,rhs)
            grad_state.retrieve(grad_state_snap, t)
            rhs.axpy(-1., grad_state_snap)
            self.solvert.solve(p, rhs)
            pold = p
            out.store(p, t)

    def evalGradientParameter(self,x, mg, misfit_only=False):
        self.prior.init_vector(mg,1)
        if misfit_only == False:
            dm = x[PARAMETER] - self.prior.mean
            self.prior.R.mult(dm, mg)
        else:
            mg.zero()

        p0 = dl.Vector()
        self.M.init_vector(p0,0)
        x[ADJOINT].retrieve(p0, self.simulation_times[1])

        mg.axpy(-1., self.Mt_stab*p0)

        g = dl.Vector()
        self.M.init_vector(g,1)

        self.prior.Msolver.solve(g,mg)

        grad_norm = g.inner(mg)

        return grad_norm

    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):
        """
        Specify the point x = [u,a,p] at which the Hessian operator
        (or the Gauss-Newton approximation)
        need to be evaluated.

        Nothing to do since the problem is linear
        """
        self.gauss_newton_approx = gauss_newton_approx
        return

    def solveFwdIncremental(self, sol, rhs):
        sol.zero()
        uold = dl.Vector()
        u = dl.Vector()
        Muold = dl.Vector()
        myrhs = dl.Vector()
        self.M.init_vector(uold, 0)
        self.M.init_vector(u, 0)
        self.M.init_vector(Muold, 0)
        self.M.init_vector(myrhs, 0)

        for t in self.simulation_times[1::]:
            self.M_stab.mult(uold, Muold)
            rhs.retrieve(myrhs, t)
            myrhs.axpy(1., Muold)
            self.solver.solve(u, myrhs)
            sol.store(u,t)
            uold = u

    def solveAdjIncremental(self, sol, rhs):
        sol.zero()
        pold = dl.Vector()
        p = dl.Vector()
        Mpold = dl.Vector()
        myrhs = dl.Vector()
        self.M.init_vector(pold, 0)
        self.M.init_vector(p, 0)
        self.M.init_vector(Mpold, 0)
        self.M.init_vector(myrhs, 0)

        for t in self.simulation_times[::-1]:
            self.Mt_stab.mult(pold,Mpold)
            rhs.retrieve(myrhs, t)
            Mpold.axpy(1., myrhs)
            self.solvert.solve(p, Mpold)
            pold = p
            sol.store(p, t)

    def applyC(self, dm, out):
        out.zero()
        myout = dl.Vector()
        self.M.init_vector(myout, 0)
        self.M_stab.mult(dm,myout)
        myout *= -1.
        t = self.simulation_times[1]
        out.store(myout,t)

        myout.zero()
        for t in self.simulation_times[2:]:
            out.store(myout,t)

    def applyCt(self, dp, out):
        t = self.simulation_times[1]
        dp0 = dl.Vector()
        self.M.init_vector(dp0,0)
        dp.retrieve(dp0, t)
        dp0 *= -1.
        self.Mt_stab.mult(dp0, out)

    def applyWuu(self, du, out):
        out.zero()
        self.misfit.apply_ij(STATE, STATE, du, out)

    def applyWum(self, dm, out):
        out.zero()

    def applyWmu(self, du, out):
        out.zero()

    def applyR(self, dm, out):
        self.prior.R.mult(dm,out)

    def applyWmm(self, dm, out):
        out.zero()

    def exportState(self, x, filename, varname):
        out_file = dl.XDMFFile(self.Vh[STATE].mesh().mpi_comm(), filename)
        out_file.parameters["functions_share_mesh"] = True
        out_file.parameters["rewrite_function_mesh"] = False
        ufunc = dl.Function(self.Vh[STATE], name=varname)
        t = self.simulation_times[0]
        out_file.write(vector2Function(x[PARAMETER], self.Vh[STATE], name=varname),t)
        for t in self.simulation_times[1:]:
            x[STATE].retrieve(ufunc.vector(), t)
            out_file.write(ufunc, t)
