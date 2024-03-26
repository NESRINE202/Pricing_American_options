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
    def __init__(
        self, r: float, sigma: float, S0: float, L: float,
        m: int,n:int, projection_base: str, payoff_function) -> None:
        """
        Initialize the DynamicPricing object.

        Parameters:
            r (float): Risk-free interest rate.
            sigma (float): Volatility.
            S0 (float): Initial stock price.
            L (float): Stock price at maturity.
            m (int): Number of basis functions.
            projection_base (str): Type of projection base ('poly' or 'Laguerre').
            payoff_function (Callable): Payoff function for options (call or put).
        """
        super().__init__(r, sigma, S0, L,n)
        self.m = m 
        self.payoff_function = payoff_function
        self.projection_base = self._set_projection_base(projection_base)
        # self.projection_base = projection_base


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
            m (int): Number of basis functions.
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Polynomial base.
        """
        return x ** np.arange(1, m + 1)
    
    @staticmethod
    def laguerre_base(m: int, x: np.ndarray) -> np.ndarray:
        """
        Laguerre projection base function.

        Parameters:
            m (int): Number of basis functions.
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Laguerre base.
        """
        laguerre_polynomials = np.zeros(m)
        for i in range(m):
            laguerre_polynomials[i] = np.exp(-x / 2) * x**i * np.sqrt(np.math.factorial(i) / np.math.factorial(i + 1))
        
        return laguerre_polynomials

    
    
    def least_square_minimizer(self,payoff_simulation, Tau_i_1, Price_simulation_i, projection_base,m,n):
        Y = np.zeros(len(Tau_i_1))
        
        # Fill the array Y by selecting elements from Z based on indices
        for k in range(n):
            Y[k]=payoff_simulation[int(Tau_i_1[k]-1),k]
        X = np.array([projection_base(m, Price_simulation_i[path]) for path in range(n)])
        model = LinearRegression()
        model.fit(X, Y)
        alpha = model.coef_
        return alpha

    def dynamic_prog_price(self):

        payoff_0 = self.payoff_function(self.S0)
        price_simulation=self.monte_carlo_price_simulator()
        payoff_simulation = self.monte_carlo_payoff_simulator(self.payoff_function,price_simulation)
        
        n  = self.n
        m  = self.m
        L = self.L
        Tau = np.zeros((L, n))
        Tau[L - 1, :] = L * np.ones(n)
        for i in range(L - 2, -1, -1):
            alpha_i = self.least_square_minimizer(payoff_simulation, Tau[i + 1, :], price_simulation[i, :], self.projection_base,m,n)
            for path in range(n):
                approx_ = alpha_i.T @ self.projection_base(m,price_simulation[i, path])
                if payoff_simulation[i, path] >= approx_:
                    Tau[i, path] = i
                else:
                    Tau[i, path] = Tau[i + 1, path]


        monte_carlo_approx = sum([payoff_simulation[int(Tau[0, i])- 1,0] for i in range(n)]) / n
        U_0 = max(payoff_0, monte_carlo_approx)

        return U_0

