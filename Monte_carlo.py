import numpy as np 

    
class MonteCarlo_simulator(): 
    def __init__(self,r,sigma,time_step,call = True,S0,L,n):
        """
        ARGS: 
        r : risk free rate 
        sigma : volatilty of the asset 
        s_0 : the price at time 0
        L : time de maturity / number of divisions of time  
        n: number of simulation/paths we want 
    

        """
        self.r= r 
        self.sigma = sigma
        self.time_step = time_step
        self.S0 = S0
        self.L = L 
        self.n = n 

    def monte_carlo_price_simulator(self):

        Price_simulation = np.zeros((self.L,self.n))
        for path in range(self.n):
            # We initialize the price at time 1 
            z_0 = np.random.normal(0,1)
            Price_simulation[0,path] = self.s_0* np.exp((self.r-self.sigma**2/2)+self.sigma*z_0)
            for i in range(1,self.L): 
                z_i = np.random.normal(0,1)
                Price_simulation[i,path] = Price_simulation[i-1,path]*np.exp((self.r-self.sigma**2/2)+self.sigma*z_i)

        
        return Price_simulation
    

    
    def monte_carlo_payoff_simulation(self, payoff_function):

        price_simulation = MonteCarlo_simulator(self)

        return np.vectorize(payoff_function)(price_simulation)
    

