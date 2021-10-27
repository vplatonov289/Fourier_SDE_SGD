import numpy as np
import matplotlib.pyplot as plt

import time

from brownian_motion import simulate_dW, simulate_dW_1d, transform_dW
from kuramoto_appr import SDE_Kuramoto_MV_appr
from linear_appr import SDE_Linear_MV_appr
'''
Main algorithm here
'''

class SGD_MV():
       
    def __init__(self, x_0 = 1, sigma = 1, alpha = - 0.5, beta = 0.3, T = 0.2):
        self.x_0 = x_0
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.T = T

    def kuramoto_get_gamma(self, n_discr = 10, eta = 0.01, gamma = np.random.uniform(low = - 0.3, high = 0.3, size = 10), eps = 1e-4, N_iter = 10000, cyclic_lr = False):
        
        t_0 = time.time()
        gamma_aver = 0
        i = 0
        err = np.inf

        eta_base = np.copy(eta)
        while (err > eps) and (i < N_iter):

            dW_t_1 = simulate_dW_1d(n_discr,self.T)
            dW_t_2 = simulate_dW_1d(n_discr,self.T)

            X_1 = SDE_Kuramoto_MV_appr(x_0 = self.x_0, sigma = self.sigma, gamma = gamma, dW_t = dW_t_1,
                                       T = self.T, n_discr = n_discr)
            X_2 = SDE_Kuramoto_MV_appr(x_0 = self.x_0, sigma = self.sigma, gamma = gamma, dW_t = dW_t_2,
                                       T = self.T, n_discr = n_discr)
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
                    gradient_path = X_2.get_path_for_gradient_SDE(k_1,i_1)
                    for i_2 in range(n_discr):
                        for k_2 in range(2):
                            if k_1 == 0:
                                jacobian[k_2 * n_discr + i_2][k_1 * n_discr + i_1] = np.cos(X_2.x[i_2]) * gradient_path[i_2]
                            elif k_1 == 1:
                                jacobian[k_2 * n_discr + i_2][k_1 * n_discr + i_1] = np.sin(X_2.x[i_2]) * gradient_path[i_2] 
            
            jacobian = jacobian - np.eye(2 * n_discr)
            
            
            gamma_old = np.copy(gamma)

            gamma = gamma.reshape(2 * n_discr)
            gamma = gamma - eta * np.matmul(grad_first_part,jacobian) ## add L_2 regularisation
            gamma = gamma.reshape(2, n_discr)
            err = abs(gamma_old - gamma).max()

            # For debugging:
#             print(f'Gradient first part is {grad_first_part}')
#             print(f'Jacobian is {jacobian}')
#             print(f'SGD correction is {eta * np.matmul(grad_first_part,jacobian)}')

            #err = max(abs(sin(X) - gamma[0]))
            #print(gamma.shape)

            i += 1
        # ToDo: save the gamma for later to calculate the weighted average further on 
            
            if cyclic_lr == True:
                eta = eta * 0.99
                if (((i + 1) % (N_iter // 10)) == 0) or (eta < 0.0001):
                    eta = eta_base
                    
            gamma_aver = i / (i + 1) * gamma_aver + 1 / (i + 1) * gamma
            #gamma_aver = 0.9 * gamma_aver + 0.1 * gamma
            
#             if i % 100 == 0:
#                 print(gamma)
            
            
            if (err <= eps):
                print(f'Algorithm stopped due to reached tolerance at iteration {i}.')
            
            #print(f'Gamma average for {i} iteration is: \n {gamma_aver}')
            #print(f'Step {i} completed.')
            if ((i + 1) % (N_iter // 10)) == 0 and (time.time() - t_0 > 60):
                print(f'|{((i+1) // (N_iter // 10)) * 10}% of the iterations completed.|')
        print(f'Solved for {time.time() - t_0:.{4}} seconds.')
        return gamma_aver

    
    def linear_get_gamma(self, n_discr = 10, eta = 0.05, gamma = np.random.uniform(low = - 0.3, high = 0.3, size = 10), eps = 1e-4, N_iter = 10000, cyclic_lr = False):
        
        t_0 = time.time()
        gamma_aver = 0
        i = 0
        err = np.inf
        errors = []
        
        eta_base = np.copy(eta)
        
        while (err > eps) and (i < N_iter):
            
            dW_t_1 = simulate_dW_1d(n_discr, self.T)
            dW_t_2 = simulate_dW_1d(n_discr, self.T)

            X_1 = SDE_Linear_MV_appr(x_0 = self.x_0, alpha = self.alpha, beta = self.beta, sigma = self.sigma, 
                 gamma = gamma, dW_t = dW_t_1, T = self.T, n_discr = n_discr)

#             X_2 = SDE_Linear_MV_appr(self.x_0 = 0, alpha = alpha, beta = beta, sigma = 1, 
#                  gamma = np.random.uniform(low = -0.3, high = 0.3, size = 100), 
#                  dW_t = dW_t_2, T = T, n_discr = n_discr)

            grad_first_part = np.zeros(n_discr)
            grad_first_part = 2 * (X_1.x - gamma)

            jacobian = np.zeros((n_discr, n_discr))

            # rewrite to avoid extra calculations.
            
            for i_1 in range(n_discr):
                gradient_path = X_1.get_path_for_gradient_SDE(i_1)
                for i_2 in range(n_discr):
                    jacobian[i_2][i_1] = gradient_path[i_2] 
            
            #print(jacobian)
            
            jacobian = jacobian - np.eye(n_discr)
            
            #print(jacobian)
            
            gamma_old = np.copy(gamma)
            gamma = gamma - eta * np.matmul(grad_first_part,jacobian)
            #print(f'Gradient is: {np.matmul(grad_first_part,jacobian)})')
            err = abs(gamma_old - gamma).max()
            errors.append(err)
            
            if cyclic_lr == True:
                eta = eta * 0.99
                if (((i + 1) % (N_iter // 10)) == 0) or (eta < 0.0001):
                    eta = eta_base
            i += 1
            gamma_aver = i / (i + 1) * gamma_aver + 1 / (i + 1) * gamma
            #print(gamma)
            if (err <= eps):
                print(f'Algorithm stopped due to reached tolerance at iteration {i}.')

            #print(f'Gamma average for {i} iteration is: \n {gamma_aver}')
            #print(f'Step {i} completed.')
            
            if ((i + 1) % (N_iter // 10))  == 0 and (time.time() - t_0 > 60):
                print(f'|{((i+1) // (N_iter // 10)) * 10}% of the iterations completed.|')
#     print(f'The solution of the SGD algorithm is {gamma_aver}.')
        print(f'Solved for {time.time() - t_0:.{4}} seconds.')
        return gamma_aver#, errors