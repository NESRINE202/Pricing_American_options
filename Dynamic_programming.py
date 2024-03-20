import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from Monte_carlo import MonteCarlo_simulator 


    

class Dynamic_pricing(MonteCarlo_simulator):
    def __init__(self, S0, L, n, m, Projection_base, payoff_function, r=None, sigma=None, a=None, b=None, q=None, model_type="GBM"):
        super().__init__(S0, L, n, r, sigma, a, b, q, model_type)
        self.m = m #taille de la base de projection
        self.Projection_base  = Projection_base  # Améliorer : donner des noms aux polynomes à utiliser 
        self.payoff_function= payoff_function # pareil il y a dans ce cas call ou  put 
    
    
    #Z : payoff
    #X: price

    def least_square_minimizer(self, payoff_simulation, Tau_i_1, Price_simulation_i, Projection_base): #tau_i_1 = tau_i+1
        Y = np.array([payoff_simulation[int(Tau_i_1[path] - 1), path] for path in range(self.n)])
        X = np.array([Projection_base(self.m, Price_simulation_i[path]) for path in range(self.n)])
        model = LinearRegression()
        model.fit(X, Y)
        alpha = model.coef_
        return alpha

    def dynamic_prog_price(self):

        payoff_0 = self.payoff_function(self.S0)
        n = self.n # nombre des simulaions 
        m = self.m #taille de la base
        L = self.L

        payoff_simulation = self.monte_carlo_payoff_simulator(self.payoff_function)

        Tau = np.zeros((L - 1, n))
        Tau[L - 2, :] = L * np.ones(n)
        for i in range(L - 3, -1, -1):
            alpha_i = self.least_square_minimizer(payoff_simulation, Tau[i + 1, :], payoff_simulation[i, :], self.Projection_base)
            for path in range(n):
                approx_ = alpha_i.T @ self.Projection_base(m,payoff_simulation[i, path])
                if payoff_simulation[i, path] >= approx_:
                    Tau[i, path] = i
                else:
                    Tau[i, path] = Tau[i + 1, path]

        monte_carlo_approx = sum([payoff_simulation[int(Tau[0, i])- 1,0] for i in range(n)]) / n
        U_0 = max(payoff_0, monte_carlo_approx)

        return U_0


