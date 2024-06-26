# Pricing of American Options

### Project Overview:
The objective of this project is to implement the Longstaff and Schwartz algorithm for pricing American options. This algorithm involves two main approximations: replacing conditional expectations in the dynamic programming principle and using Monte Carlo simulations with least squares regression to compute the option value function.

### File structure:
- **Option.py** : Used to define the option's parameters as well as its nature (call or put).
- **Monte_carlo.py** : Implements Monte Carlo simulations for option pricing.
- **Dynamic_Programming.py** : Implements the dynamic programming algorithm to estimate the stopping times and thus the option's value at time 0.
- **Black_Scholes.py**: Computes the Black Scholes price.
- **Least_squares.ipynb** : Notebook to execute the American option pricing with least squares method.
- **quantization.ipynb**: Notebook to excute and test the American pricing with quantization. 
- **Tests_least_squares.ipynb**: Notebook to test the least square algorithm and analyse results.
### Supervisor:
Mr. Gaoyue Guo, Assistant Professor of *Laboratory in Mathematics and Computer Science at CentraleSupélec*.

### References: 
- Clément, E., Lamberton, D., & Protter, P. "An analysis of a least squares regression method for American option pricing," *Finance and Stochastics*, 6(4), 449–471, 2002. Published by Springer-Verlag
- Bally, V., Pagès, G., & Printems, J. "A Quantization Tree Method for Pricing and Hedging Multidimensional American Options," *Mathematical Finance*, Vol. 15, No. 1, pp. 119–168, January 2005.

### Authors:
CHAFIK Hala, CHIBA Nessrine, DEMRI Lina, ELOMARI Chaimae, NAGAZ Sarra