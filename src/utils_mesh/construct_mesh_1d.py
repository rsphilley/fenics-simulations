from fenics import *
import dolfin as dl

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_mesh(options):

    # construct mesh
    mesh = dl.IntervalMesh(options.nx,
                           options.left_boundary, options.right_boundary)

    # finite element space
    Vh = FunctionSpace(mesh, 'P', options.order_fe_space)

    # functions on the spaces
    u = TrialFunction(Vh)
    v = TestFunction(Vh)
    parameter = Function(Vh)

    # get the mesh topology
    dof = Vh.dim()
    nodes = Vh.tabulate_dof_coordinates()

    return Vh, nodes, dof
