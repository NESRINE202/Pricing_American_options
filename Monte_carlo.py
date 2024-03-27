import numpy as np 
import matplotlib.pyplot as plt 


class MonteCarlo_simulator(): 
    def __init__(self,S0,L,n,r=None,sigma=None,a=None, b=None,q=None, model_type="GBM"):
        """
        ARGS: 
        r : risk free rate 
        sigma : volatilty of the asset 
        s_0 : the price at time 0
        L : time de maturity / number of divisions of time  
        n: number of simulation/paths we want 
        a,b = interest rate binomial model
        q : proba of having a
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
        
#  use the GeometricBrownianMotion class to simulation some paths : sigma, r 
# or binomial tree
    def monte_carlo_price_simulator(self):
        Price_simulation = np.zeros((self.L +1, self.n))  # X0  X=price
       
        if self.model_type == "GBM":
            for path in range(self.n):
                Price_simulation[0, path] = self.S0
            # We initialize the price at time 1 
                
                for i in range(1,self.L+1): 
                    z_i = np.random.normal(0,1) #Wt+1 -Wt
                    Price_simulation[i,path] = Price_simulation[i-1,path]* np.exp((self.r - self.sigma**2 / 2) * (1 / self.L) + self.sigma * np.sqrt(1 / self.L) * z_i)
                    #np.exp((self.r-self.sigma**2/2)+self.sigma*z_i)
                    
                    
        elif self.model_type == "Binomial":
            for path in range(self.n):
                Price_simulation[0, path] = self.S0
                r = np.random.choice([self.a, self.b], size=self.L-1, p=[self.q, 1-self.q])
                for i in range(1, self.L +1):
                    Price_simulation[i, path] = Price_simulation[i-1, path] * r[i-1]
                    # z = np.random.rand()  # Tirage aléatoire entre 0 et 1
                    # if z < self.q:
                    #     Price_simulation[i, path] = Price_simulation[i-1, path] * self.a 
                    # else:
                    #     Price_simulation[i, path] = Price_simulation[i-1, path] * self.b  

        return Price_simulation
    # def monte_carlo_price_simulator(self):

    #     Price_simulation = np.zeros((self.L,self.n))  #X0  X=price
    #     for path in range(self.n):
    #         # We initialize the price at time 1 
    #         z_0 = np.random.normal(0,1)  # W1-W0
    #         Price_simulation[0,path] = self.S0* np.exp((self.r-self.sigma**2/2)+self.sigma*z_0) # GeometricBrownianMotion   # X1
    #         for i in range(1,self.L): 
    #             z_i = np.random.normal(0,1) #Wt+1 -Wt
    #             Price_simulation[i,path] = Price_simulation[i-1,path]*np.exp((self.r-self.sigma**2/2)+self.sigma*z_i)
        
        
    #     return Price_simulation
    
    def visualisation_price(self): 
        Price_simulation = self.monte_carlo_price_simulator()
        plt.figure(figsize=(10, 6))
        for i in range(self.n): #pour chaque simulation on va tracer l'évolution du prix
            price = np.insert(Price_simulation[:, i],0,self.S0)
            plt.plot(range(self.L+2), price, label=f"Path {i+1}")

        plt.title('Monte Carlo Simulation of Asset Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        #plt.legend()
        plt.grid(True)
        plt.show()

    
    def monte_carlo_payoff_simulator(self, payoff_function,price_simulation):


        return np.vectorize(payoff_function)(price_simulation)
    

