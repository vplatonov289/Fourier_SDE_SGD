'''
We put here the functions to simulate the Brownian motion increments and a function to transform them with the new frequency.
'''

import numpy as np
import matplotlib.pyplot as plt

def simulate_dW(N,T):
    return np.random.normal(size = N, scale = np.sqrt(T/N[1]))

def simulate_dW_1d(N,T):
    return np.random.normal(size = N, scale = np.sqrt(T/N))

def transform_dW(dW_t,n_discr, n):
    dW_t_new = np.zeros(n)
    k = 0
    local = 0
    print(dW_t.shape[0])
    for i in range(dW_t.shape[0]):
        local += dW_t[i]
        if (i + 1) % (n_discr // n) == 0:
            dW_t_new[k] = local
            local = 0
            k += 1
    return dW_t_new