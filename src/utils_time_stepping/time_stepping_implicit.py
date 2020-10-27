import numpy as np

def time_stepping_implicit(operator_lhs, operator_rhs, state_current):

    state_current = np.linalg.solve(operator_lhs, np.matmul(operator_rhs, state_current))

    return state_current
