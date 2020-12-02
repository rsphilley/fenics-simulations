import numpy as np

def positivity_constraint_identity(x, dummy):
    return x

def positivity_constraint_exp(x, dummy):
    return np.exp(x)

def positivity_constraint_log_exp(x,k):
    return (1/k)*np.log(np.exp(k*x)+1)
