from scipy.special import genlaguerre
import numpy as np 




def laguerre_basis(m, x):
    basis = []
    for i in range(m):
        basis.append(genlaguerre(i, 0))  # Les polynômes de Laguerre de degré i
    return np.array([basis[i](x) for i in range(m)])


def GD_least_square(X, y, learning_rate=0.01, num_iterations=1000, tolerance=1e-4):
    # Initialisation des poids
    num_samples, num_features = X.shape
    theta = np.zeros((num_features, 1))

    # Gradient descent
    for i in range(num_iterations):
        # Calcul de la prédiction
        y_pred = np.dot(X, theta)

        # Calcul de l'erreur
        error = y_pred - y

        # Calcul du gradient
        gradient = np.dot(X.T, error)
        # print(np.linalg.norm(error))

        # Mise à jour des poids
        theta -= learning_rate * gradient

        # Vérification de la convergence
        if np.linalg.norm(gradient) < tolerance:
            break 

    return theta 

def least_square_minimizer(payoff_simulation,Tau_i_1,Price_simulation_i,Projection_base,m): 
    
    n= len(payoff_simulation.shape[1])


    
    Y = np.array([payoff_simulation[Tau_i_1[path],path]] for path in range(n))

    X = np.array([Projection_base(m,Price_simulation_i[path]) for path in range(n)])

    # Methode_1: Calcul matricielle [beaucoup de temps de calcul ]
    """Methode eliminé on n'est pas toujours sur que XX.T est ineversible"""
    alpha_1 = np.linalg.inv(X@X.T) @ Y.T @ X

    
    # Methode_2: descente de gradient 
    alpha_2 = GD_least_square(X,Y,learning_rate = 0.01,n_iterations=1000 ,epsilon= 1e-5)


    #Methode_3 : descente de gradient stochastique 

    

    
    return alpha_2


def dynamic_prog_price(Price_simulation,payoff_simulation,Projection_base,payoff_0,n,m,L): 

    """
    Price_simulation : array shape ((L-1),n) containing for each time step n paths simulated by monte carlo 

    Payoff_simulation : array shape ((L-1),n) containing for each time step n paths of the payoff of the option simulated by monte carlo 
    Projection_base : une famille generatrice et libre de fonctions d'un sous-espace de L2 de dim m 
    price_0: price at time 0 
    n = n_path 
    m : Dimension of the projection space 
    L = maturity 

    """

    Tau = np.zeros((L-1,n))
    Tau[L-2,:] = L * np.ones(n)

    # for i in range(L-3, -1, -1): 
    #     alpha_i = least_square_minimizer(payoff_simulation[i,:], Tau[i+1,:], Price_simulation[i,:], Projection_base) 
    #     approximations = alpha_i.T @ Projection_base(Price_simulation[i])
    #     Tau[i] = np.where(payoff_simulation[i] >= approximations, i, Tau[i+1])
    #     Tau = np.zeros((L-1,n))

    # Intialisation de tau_L = L pour chaque chemin 
    Tau[L-2,:] = L*np.ones(n)
    for i in range(L-3,-1,-1): 
        alpha_i = least_square_minimizer(payoff_simulation,Tau[i+1,:],Price_simulation[i,:],Projection_base) 
        for path in range(n): 
            approx_ = alpha_i.T @ Projection_base(Price_simulation[i,path])
            if payoff_simulation[i,path]>= approx_: 
                Tau[i,path] = i
            else: 
                Tau[i,path] = Tau[i+1,path]

    #price_of_the_option 
    monte_carlo_approx = sum([payoff_simulation[0,Tau[0,i]] for i in range(n)])/n
    U_0 = max(payoff_0,monte_carlo_approx)
    
    return U_0


