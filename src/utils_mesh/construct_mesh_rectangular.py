from fenics import *

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_mesh(options):

    # define mesh via the diagonal
    if len(options.mesh_point_1) == 2:
        mesh = RectangleMesh(Point(options.mesh_point_1[0], options.mesh_point_1[1]),
                             Point(options.mesh_point_2[0], options.mesh_point_2[1]),
                             options.num_nodes_x, options.num_nodes_y)
    if len(options.mesh_point_1) == 3:
        mesh = BoxMesh(Point(options.mesh_point_1[0], options.mesh_point_1[1],
                             options.mesh_point_1[2]),
                       Point(options.mesh_point_2[0], options.mesh_point_2[1],
                             options.mesh_point_2[2]),
                       options.num_nodes_x, options.num_nodes_y, options.num_nodes_z)

    # finite element space
    fe_space = FunctionSpace(mesh, 'P', options.order_fe_space)

    # meta-FE space - discretize the conductivity/parameter
    # same space as fe_space, but does not have to be
    meta_space = FunctionSpace(mesh, 'P', options.order_meta_space)

    # functions on the spaces
    u = TrialFunction(fe_space)
    v = TestFunction(fe_space)
    parameter = Function(meta_space)

    # get the mesh topology
    dof_fe = fe_space.dim()
    dof_meta = meta_space.dim()
    nodes = meta_space.tabulate_dof_coordinates()

    return fe_space, meta_space,\
            nodes, dof_fe, dof_meta
