from fenics import *

def stiffness(sig, u, v):
    return sig * inner(grad(u), grad(v)) * dx
