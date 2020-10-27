import numpy as np

def time_stepping_erk4(options, operator, state_current):
    dt = options.time_dt

    k_1 = np.matmul(operator, state_current)
    k_2 = np.matmul(operator, state_current + (1/2)*dt*k_1)
    k_3 = np.matmul(operator, state_current + (1/2)*dt*k_2)
    k_4 = np.matmul(operator, state_current + dt*k_3)

    state_current += (1/6)*dt*(k_1 + 2*k_2 + 2*k_3 + k_4)

    return state_current
