import numpy as np
import matplotlib.pyplot as plt

import time

from brownian_motion import simulate_dW, simulate_dW_1d, transform_dW
from kuramoto_appr import SDE_Kuramoto_MV_appr
from linear_appr import SDE_Linear_MV_appr
'''
Main algorithm here
'''

def SGD_MV(n_discr, eta, gamma, T, eps, N_iter, key):
    t_0 = time.time()
    gamma_aver = 0
    i = 0
    err = np.inf
    x_0 = 1
    sigma = 1
    
    alpha = -0.5
    beta = 0.3
    
#     eta_base = np.copy(eta)
    if key == "kuramoto":
        while (err > eps) and (i < N_iter):

            dW_t_1 = simulate_dW_1d(n_discr,T)
            dW_t_2 = simulate_dW_1d(n_discr,T)

            X_1 = SDE_Kuramoto_MV_appr(x_0, sigma, gamma, dW_t_1, T, n_discr)
            X_2 = SDE_Kuramoto_MV_appr(x_0, sigma, gamma, dW_t_2, T, n_discr)
            #print(f'initial condition = {np.sin(X_1.x[0])}')
            #print(f'initial condition = {-np.cos(X_1.x[0])}')
            
            #print(f'gamma_0 = {gamma[0,0]}')
            #print(f'gamma_1 = {gamma[1,0]}')
            
            grad_first_part = np.zeros((2,n_discr))
            grad_first_part[0] = 2 * (np.sin(X_1.x) - gamma[0])

            grad_first_part[1] = 2 * (-np.cos(X_1.x) - gamma[1])

            grad_first_part = grad_first_part.reshape(2 * n_discr)

            jacobian = np.zeros((2 * n_discr, 2 * n_discr))

            for i_1 in range(n_discr):
                for k_1 in range(2):
                    for i_2 in range(n_discr):
                        for k_2 in range(2):
                            if k_1 == 0:
                                jacobian[k_1 * n_discr + i_1][k_2 * n_discr + i_2] = np.cos(X_2.x[i_1]) * X_2.get_path_for_gradient_SDE(k_2,i_2)[i_1] - (i_1 == i_2 and k_1 == k_2)
                            elif k_1 == 1:
                                jacobian[k_1 * n_discr + i_1][k_2 * n_discr + i_2] = np.sin(X_2.x[i_1]) * X_2.get_path_for_gradient_SDE(k_2,i_2)[i_1] - (i_1 == i_2 and k_1 == k_2)

            #print(jacobian)
    #         eta = eta * 0.9
    #         if i % (N_iter // 200):
    #             eta = eta_base

            gamma_old = np.copy(gamma)

            gamma = gamma.reshape(2 * n_discr)
            gamma = gamma - eta * np.matmul(grad_first_part,jacobian)
            gamma = gamma.reshape(2, n_discr)
            err = abs(gamma_old - gamma).max()
            
#             print(f'Gradient first part is {grad_first_part}')
#             print(f'Jacobian is {jacobian}')
            
#             print(f'SGD correction is {eta * np.matmul(grad_first_part,jacobian)}')
            
            #err = max(abs(sin(X) - gamma[0]))
            #print(gamma.shape)

            i += 1
        # ToDo: save the gamma for later to calculate the weighted average further on 

            gamma_aver = i / (i + 1) * gamma_aver + 1 / (i + 1) * gamma
            #print(f'Gamma average for {i} iteration is: \n {gamma_aver}')
            #print(f'Step {i} completed.')
            if ((i + 1) % (N_iter // 10))  == 0 and (time.time() - t_0 > 60):
                print(f'|{((i+1) // (N_iter // 10))*10}% of the iterations completed.|')
    
    elif key == "linear":
        while (err > eps) and (i < N_iter):

            dW_t_1 = simulate_dW_1d(n_discr,T)
            dW_t_2 = simulate_dW_1d(n_discr,T)

            X_1 = SDE_Linear_MV_appr(x_0 = 0, alpha = alpha, beta = beta, sigma = 1, 
                 gamma = np.random.uniform(low = -0.3, high = 0.3, size = 100), 
                 dW_t = dW_t_1, T = T, n_discr = n_discr)
            X_2 = SDE_Linear_MV_appr(x_0 = 0, alpha = alpha, beta = beta, sigma = 1, 
                 gamma = np.random.uniform(low = -0.3, high = 0.3, size = 100), 
                 dW_t = dW_t_2, T = T, n_discr = n_discr)
            #print(f'initial condition = {np.sin(X_1.x[0])}')
            #print(f'initial condition = {-np.cos(X_1.x[0])}')
            
            #print(f'gamma_0 = {gamma[0,0]}')
            #print(f'gamma_1 = {gamma[1,0]}')
            
            grad_first_part = np.zeros(n_discr)
            grad_first_part = 2 * (X_1.x - gamma)

            jacobian = np.zeros((n_discr, n_discr))

            for i_1 in range(n_discr):
                for i_2 in range(n_discr):
                    jacobian[i_1][i_2] = X_2.get_path_for_gradient_SDE(i_2)[i_1] - (i_1 == i_2)
            #print(jacobian)
    #         eta = eta * 0.9
    #         if i % (N_iter // 200):
    #             eta = eta_base

            gamma_old = np.copy(gamma)
            gamma = gamma - eta * np.matmul(grad_first_part,jacobian)

            err = abs(gamma_old - gamma).max()
            
#             print(f'Gradient first part is {grad_first_part}')
#             print(f'Jacobian is {jacobian}')
            
#             print(f'SGD correction is {eta * np.matmul(grad_first_part,jacobian)}')
            
            #err = max(abs(sin(X) - gamma[0]))
            #print(gamma.shape)

            i += 1
        # ToDo: save the gamma for later to calculate the weighted average further on 

            gamma_aver = i / (i + 1) * gamma_aver + 1 / (i + 1) * gamma
            #print(f'Gamma average for {i} iteration is: \n {gamma_aver}')
            #print(f'Step {i} completed.')
            if ((i + 1) % (N_iter // 10))  == 0 and (time.time() - t_0 > 60):
                print(f'|{((i+1) // (N_iter // 10))*10}% of the iterations completed.|')
#     print(f'The solution of the SGD algorithm is {gamma_aver}.')
    print(f'Solved for {time.time() - t_0:.{4}} seconds.')
    return gamma_aver