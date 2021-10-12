import numpy as np
import matplotlib.pyplot as plt

from brownian_motion import simulate_dW, simulate_dW_1d, transform_dW

'''
We initialise the main class for MV-SDE of Kuramoto type with one harmonic.
'''

class SDE_Linear_MV():
    
    #### Check the correct passing of the arguments
    
    def __init__(self, x_0 = 1, alpha = -0.5, beta = 0.4, sigma = 1, dW_t = simulate_dW_1d(100,1),
                 T = 1, n_discr = 100):
        #self.x_0 = np.random.normal(size = n_part, scale = 0.2)
        self.x_0 = x_0
        
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        
        self.T = T
        self.n_discr = n_discr
        self.dt = self.T / self.n_discr
        self.dW_t = dW_t
        self.x = self.get_path()
        
    #### Simulates the path according to Euler algorithm    
    def get_path(self):
        x = np.zeros(self.n_discr)
        
        x[0] = self.x_0
        
        
        for i in range(1,self.n_discr):
            x[i] = x[i - 1] + (self.alpha * x[i - 1] + self.beta * np.exp((self.beta + self.alpha) * self.dt * (i - 1))) * self.dt + self.sigma * self.dW_t[i - 1]
        return x
    
    #### Plots the path
    def plot_path(self):
        t = np.linspace(0, self.T, self.n_discr)
        
        fig, ax = plt.subplots(1,1,figsize = (15, 10), tight_layout=True)

        ax.set_title(r"Dynamics of the SDE", fontsize = 15)
        ax.set_xlabel(r'$t$',fontsize = 15)
        ax.set_ylabel(r'$X_t$',fontsize = 15)
        ax.tick_params(axis='both', which='major', labelsize = 20)
        ax.tick_params(axis='both', which='minor', labelsize = 20)
        ax.plot(t, self.x)
        #ax.plot(t,[0]*t.shape[0],label = 'asymptote')
        plt.show()

        #ksi = self.get_path_for_gradient_SDE()