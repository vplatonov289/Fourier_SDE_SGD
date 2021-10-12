import numpy as np
import matplotlib.pyplot as plt

from brownian_motion import simulate_dW, simulate_dW_1d, transform_dW

'''
We initialise the main class for MV-SDE of Kuramoto type with one harmonic.
'''

class SDE_Kuramoto_MV():
    
    #### Check the correct passing of the arguments
    
    def __init__(self, x_0 = 1, sigma = 1, dW_t = simulate_dW((100,100),1),
                 T = 1, n_discr = 100, n_part = 100):
        #self.x_0 = np.random.normal(size = n_part, scale = 0.2)
        self.x_0 = np.zeros(n_part) + x_0
        
        self.sigma = sigma
        self.T = T
        self.n_discr = n_discr
        self.n_part = n_part
        self.dt = self.T / self.n_discr
        self.dW_t = dW_t
        self.x = self.get_path()
        
    #### Simulates the path according to Euler algorithm    
    def get_path(self):
        x = np.zeros((self.n_part,self.n_discr))

        sum_sin = 0
        sum_cos = 0
        
        for j in range(self.n_part):
            x[j][0] = self.x_0[j]
            sum_sin += np.sin(x[j][0])
            sum_cos += np.cos(x[j][0])
        aver_sin = sum_sin / self.n_part
        aver_cos = sum_cos / self.n_part
        
        for i in range(1,self.n_discr):
            for j in range(self.n_part):
                
                x[j][i] = x[j][i - 1] + (np.cos(x[j][i - 1]) * aver_sin 
                                      - np.sin(x[j][i - 1]) * aver_cos) * self.dt + self.sigma * self.dW_t[j][i - 1]
                sum_sin += np.sin(x[j][i])
                sum_cos += np.cos(x[j][i])
            
            aver_sin = sum_sin / self.n_part
            aver_cos = sum_cos / self.n_part
            
            sum_sin = 0
            sum_cos = 0
            
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
        for i in range(self.n_part):
            ax.plot(t, self.x[i][:])
        #ax.plot(t,[0]*t.shape[0],label = 'asymptote')
        plt.show()

        #ksi = self.get_path_for_gradient_SDE()