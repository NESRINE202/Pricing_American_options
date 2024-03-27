import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price.

    Parameters:
        S : Current stock price. type = float
        K : Option strike price. type = float
        T : Time to expiration (in years). type = float
        r : Risk-free interest rate. type = float
        sigma : Volatility. type = float
        option_type (str): Type of option ('call' or 'put').

    Returns:
        float: Option price.
    """
    T=T/365
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Please choose 'call' or 'put'.")
    
    return option_price
