
import numpy as np 
import matplotlib.pyplot as plt 


class MonteCarlo_simulator(): 
    def __init__(self,S0,L,n,r=None,sigma=None,a=None, b=None,q=None, model_type="GBM"):
        
        """
        ARGS: 
        r: risk free rate, type = float
        sigma: volatilty of the asset , type = float
        s_0 : the price at time 0, type = float
        L : time de maturity / number of divisions of time  , type = int
        n: number of simulation/paths we want , type = int
        a,b = interest rate binomial model , type = float in [-1,1]
        q : proba of having a, type = float
        model_type : type of the model ("GBM" , "Binomial" )
        """
        self.r = r
        self.sigma = sigma
        self.S0 = S0
        self.L = L
        self.n = n
        self.a = a
        self.b = b
        self.q=q
        self.model_type = model_type
        
    def monte_carlo_price_simulator(self):
        Price_simulation = np.zeros((self.L+1, self.n))  # Fixed the size to include initial price
        if self.model_type == "GBM":
            for path in range(self.n):
                Price_simulation[0, path] = self.S0  # We initialize the Price at time 0
                for i in range(1, self.L + 1):  
                    z_i = np.random.normal(0, 1)
                    Price_simulation[i, path] = Price_simulation[i - 1, path] * np.exp((self.r - self.sigma**2 / 2) * (1 / self.L) + self.sigma * np.sqrt(1 / self.L) * z_i)

        elif self.model_type == "Binomial":
            for path in range(self.n):
                Price_simulation[0, path] = self.S0 # We initialize the Price at time 0
                r = np.random.choice([self.a, self.b], size=self.L, p=[self.q, 1-self.q])
                for i in range(1, self.L+1 ):
                    Price_simulation[i, path] = Price_simulation[i-1, path] * (1+r[i-1])
                    
        return Price_simulation

    def visualisation_price(self): 
        Price_simulation = self.monte_carlo_price_simulator()
        plt.figure(figsize=(10, 6))
        for i in range(self.n): #for each simulation, we plot the price evolution.
            plt.plot(range(self.L+1), Price_simulation, label=f"Path {i+1}")
        plt.title('Monte Carlo Simulation of Asset Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()
    
    def monte_carlo_payoff_simulator(self, payoff_function,price_simulation):
        return np.vectorize(payoff_function)(price_simulation)
    
