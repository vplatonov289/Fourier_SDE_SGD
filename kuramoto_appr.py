import numpy as np
import matplotlib.pyplot as plt

from brownian_motion import simulate_dW, simulate_dW_1d, transform_dW

'''
We initialise the main class for approximation of the MV-SDE of Kuramoto type with one harmonic.
'''


class SDE_Kuramoto_MV_appr ():
    
    ### Check the correct passing of the arguments
    
    def __init__(self, x_0 = 0, sigma = 1, 
                 gamma = np.random.uniform(low = -1, high = 1, size = (2,100)), 
                 dW_t = simulate_dW_1d(100,1),  T = 1, n_discr = 100):
        self.x_0 = x_0
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

            x[i] = x[i - 1] + (self.gamma[0][i - 1] * np.sin(x[i - 1]) - self.gamma[1][i - 1] * np.cos(x[i - 1])) * self.dt + self.sigma * self.dW_t[i - 1]
        return x

    def get_path_for_gradient_SDE(self,k,n):
        ksi = np.zeros(self.n_discr)
        
        for i in range(1,self.n_discr):
            ## check if i == n or n + 1
            if k == 0:
                if (i == n):
                    ksi[i] = ksi[i - 1] + (np.sin(self.x[i - 1]) + 
                                                 self.gamma[0][i - 1] * np.cos(self.x[i - 1]) * ksi[i - 1] + 
                                                 self.gamma[1][i - 1] * np.sin(self.x[i - 1]) * ksi[i - 1]) * self.dt + self.sigma * self.dW_t[i - 1] 
                elif (i != n):
                    ksi[i] = ksi[i - 1] + (self.gamma[0][i - 1] * np.cos(self.x[i - 1]) * ksi[i - 1] + 
                                                 self.gamma[1][i - 1] * np.sin(self.x[i - 1]) * ksi[i - 1]) * self.dt + self.sigma * self.dW_t[i - 1]
            ### CHECK the signs 
            
            elif k == 1:
                if (i == n):
                    ksi[i] = ksi[i - 1] + (-np.cos(self.x[i - 1]) + 
                                                     self.gamma[0][i - 1] * np.cos(self.x[i - 1]) * ksi[i - 1] + 
                                                     self.gamma[1][i - 1] * np.sin(self.x[i - 1]) * ksi[i - 1]) * self.dt + self.sigma * self.dW_t[i - 1]   
                elif (i != n):
                    ksi[i] = ksi[i - 1] + (self.gamma[0][i - 1] * np.cos(self.x[i - 1]) * ksi[i - 1] + 
                                                 self.gamma[1][i - 1] * np.sin(self.x[i - 1]) * ksi[i - 1]) * self.dt + self.sigma * self.dW_t[i - 1]
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

