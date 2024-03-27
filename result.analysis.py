import numpy as np
import matplotlib.pyplot as plt
from Pricing import Pricing

def analyse_n(S0,k,L, listn,m,base,payoff,r,sigma,model_type):
    frequences=[]
    elements=[]
    for n in listn:
        pricing=Pricing(S0, L, n,m,base,payoff,r,sigma,model_type)
        e,f=Pricing.pricer(k)
        frequences.append(f)
        elements.append(e)
    