# imports
import numpy as np 


class Option:
    """
    Representation of an option derivative
    

    s0: float
    T: int
    K: int
    
    call: bool = True
    """
    def __init__(self,s0,T,K,call = True) -> None:
        self.s0 = s0
        self.T = T 
        self.K = K 
        self.call = call 
    def payoff(self, s: np.ndarray):#-> np.ndarray:
        payoff = np.maximum(s - self.K, 0) if self.call else np.maximum(self.K - s, 0)
        return payoff


