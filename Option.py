# imports
import numpy as np 


class Option:
    def __init__(self, underlying_price, strike_price, expiration_date):
        self.underlying_price = underlying_price
        self.strike_price = strike_price
        self.expiration_date = expiration_date
    
    def payoff(self, spot_price):
        
        pass


