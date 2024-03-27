from Dynamic_programming import Dynamic_pricing
from collections import Counter

    

class Pricing(Dynamic_pricing):
    def __init__(self, S0, L, n, m, Projection_base, payoff_function, r=None, sigma=None, a=None, b=None, q=None, model_type="GBM"):
        super().__init__( S0, L, n, m, Projection_base, payoff_function, r, sigma, a, b, q, model_type)
        #self.k = k #nombre de répétitions de dynamic programming
    def pricer(self,k):
        l=[]
        for i in range(k):
            dynamic=Dynamic_pricing(self.S0, self.L, self.n,self.m,self.Projection_base ,self.payoff_function,self.r, self.sigma,self.model_type)
            u=dynamic.dynamic_prog_price()
            #print("Dynamic Pricing:", u)
            l.append(u)
        compteur = Counter(l)
    
        # Trouver l'élément le plus fréquent et sa fréquence
        element, frequence = compteur.most_common(1)[0]
        print('u le plus fréquent =:',element)
        print('sa fréquence =',frequence)
        return(element,frequence)