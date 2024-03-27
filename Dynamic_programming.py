import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from Monte_carlo import MonteCarlo_simulator 
from scipy.optimize import minimize

    
import numpy as np

class DynamicPricing(MonteCarlo_simulator):
    """
    A class for dynamic pricing using Monte Carlo simulation.
    """
    def __init__(self, S0, L, n, m, projection_base, payoff_function, r=None, sigma=None, a=None, b=None, q=None, model_type="GBM")-> None:
        super().__init__(S0, L, n, r, sigma, a, b, q, model_type)
      
    
        """
        Initialize the DynamicPricing object.

        Parameters:
            r: risk free rate, type = float
            sigma: volatilty of the asset , type = float
            s_0 : the price at time 0, type = float
            L : time de maturity / number of divisions of time  , type = int
            n: number of simulation/paths we want , type = int
            m : Number of basis functions. type =int
            projection_base (str): Type of projection base ('poly' or 'Laguerre').
            payoff_function (Callable): Payoff function for options (call or put).
        """
        
        self.m = m 
        self.payoff_function = payoff_function
        self.projection_base = self._set_projection_base(projection_base)
        
    def _set_projection_base(self, projection_base: str):
        """
        Set the projection base function.

        Parameters:
            projection_base (str): Type of projection base ('poly' or 'Laguerre').

        Returns:
            Callable: Projection base function.
        """
        if projection_base == 'poly': 
            return self.polynomial_base
        elif projection_base == 'Laguerre': 
            return self.laguerre_base
        else: 
            return self.polynomial_base

    @staticmethod
    def polynomial_base(m: int, x: np.ndarray) -> np.ndarray:
        """
        Polynomial projection base function.

        Parameters:
            m: Number of basis functions. type= int
            x: Input array. type= np.ndarray
            
        Returns:
            np.ndarray: Polynomial base.
        """
        return x ** np.arange(1, m + 1)
    
    @staticmethod
    def laguerre_base(m: int, x: np.ndarray) -> np.ndarray:
        """
        Laguerre projection base function.

        Parameters:
            m: Number of basis functions. type= int
            x: Input array. type= np.ndarray

        Returns:
            np.ndarray: Laguerre base.
        """
        laguerre_polynomials = np.zeros(m)
        for i in range(m):
            laguerre_polynomials[i] = np.exp(-x / 2) * x**i * np.sqrt(np.math.factorial(i) / np.math.factorial(i + 1))
        
        return laguerre_polynomials

    
    
    def least_square_minimizer(self,payoff_simulation, Tau_i_1, Price_simulation_i, projection_base,n,m):  
        Y = np.zeros(len(Tau_i_1))
        
        # Fill the array Y by selecting elements from Z based on indices
        for k in range(n):
            Y[k]=payoff_simulation[int(Tau_i_1[k]),k]
        #X denotes the matrix of Projection base applied to Price simulation i
        X = np.array([projection_base(m, Price_simulation_i[path]) for path in range(n)])
        #We perform a linear regression in order to find the coefficient alpha
        model = LinearRegression()
        model.fit(X, Y)
        alpha = model.coef_
        return alpha
    

    def dynamic_prog_price(self):

        payoff_0 = self.payoff_function(self.S0)
        n = self.n # number of simulations
        m = self.m # size of the projection base
        L = self.L # maturity
        
        price_simulation = self.monte_carlo_price_simulator()
        payoff_simulation = self.monte_carlo_payoff_simulator(self.payoff_function,price_simulation)
        
        Tau = np.zeros((L, n))
        Tau[L - 1, :] = L * np.ones(n)
        for i in range(L - 2, -1, -1):
            alpha_i = self.least_square_minimizer(payoff_simulation, Tau[i + 1, :], price_simulation[i+1, :], self.projection_base,n,m)
            for path in range(n):
                approx_ = alpha_i.T @ self.projection_base(m,price_simulation[i+1, path])
                if payoff_simulation[i+1, path] >= approx_:
                    Tau[i, path] = i+1
                else:
                    Tau[i, path] = Tau[i + 1, path]
        monte_carlo_approx = sum([payoff_simulation[int(Tau[0, i]),i] for i in range(n)]) / n
        U_0 = max(payoff_0, monte_carlo_approx)

        return U_0