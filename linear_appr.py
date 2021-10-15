import numpy as np
import matplotlib.pyplot as plt

from brownian_motion import simulate_dW, simulate_dW_1d, transform_dW


'''
We initialise the main class for approximation of the MV-SDE of Kuramoto type with one harmonic.
'''


class SDE_Linear_MV_appr ():
    
    ### Check the correct passing of the arguments
    
    def __init__(self, x_0 = 0, alpha = -0.5, beta = 0.3, sigma = 1, 
                 gamma = np.random.uniform(low = -0.3, high = 0.3, size = 100), 
                 dW_t = simulate_dW_1d(100,1), T = 1, n_discr = 100):
        self.x_0 = x_0
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.T = T
        self.n_discr = n_discr
        self.dt = self.T / self.n_discr
        self.gamma = gamma
        self.dW_t = dW_t
        self.x = self.get_path()
        
    #### Simulates the path according to Euler algorithm    
    def get_path(self):
        x = np.zeros(self.n_discr)
        x[0] = self.x_0 
        for i in range(1,self.n_discr):
            x[i] = x[i - 1] + (self.alpha * x[i - 1] + self.beta * self.gamma[i - 1]) * self.dt + self.sigma * self.dW_t[i - 1]
        return x

    def get_path_for_gradient_SDE(self,n):
        ksi = np.zeros(self.n_discr)
        
        for i in range(1,self.n_discr):
            if (i == n):
                    ksi[i] = ksi[i - 1] + (self.alpha * ksi[i - 1] + self.beta) * self.dt 
            elif (i != n):
                    ksi[i] = ksi[i - 1] + (self.alpha * ksi[i - 1]) * self.dt
        return ksi
    
    #### Plots the path
    def plot_path(self):
        t = np.linspace(0, self.T, self.n_discr)
        
        fig, ax = plt.subplots(1,1,figsize=(15, 10), tight_layout=True)

        ax.set_title(r"Dynamics of the SDE", fontsize = 15)
        ax.set_xlabel(r'$t$',fontsize=15)
        ax.set_ylabel(r'$X_t$',fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize = 20)
        ax.tick_params(axis='both', which='minor', labelsize = 20)
        ax.plot(t, self.x)
        plt.show()

