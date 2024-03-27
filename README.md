# Pricing_American_options

### Project Overview:
This project focuses on implementing the Longstaff and Schwartz algorithm for pricing American options. The algorithm utilizes Monte Carlo simulations and least squares regression to approximate the option's value function, allowing for efficient pricing of American options.

### File structure:
- Option.py : Used to define the option's parameters as well as its nature ( call or put ).
- Monte_carlo.py : Implements Monte Carlo simulations for option pricing.
- Dynamic_Programming.py : Implements the dynamic programming algorithm to estimate the stopping times and thus the option's value at time 0.
- main.py : Main script to execute the American option pricing.
