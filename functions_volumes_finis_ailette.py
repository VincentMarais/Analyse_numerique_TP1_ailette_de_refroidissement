import numpy as np
import matplotlib.pyplot as plt
from typing import Any

def thomas_algorithm(a, b, c, d):
    """
    Résout un système tridiagonal de la forme Ax = d en utilisant l'algorithme de Thomas.

    Paramètres :
    a (array): Vecteur des coefficients sub-diagonaux de la matrice tridiagonale.
    b (array): Vecteur des coefficients diagonaux de la matrice tridiagonale.
    c (array): Vecteur des coefficients super-diagonaux de la matrice tridiagonale.
    d (array): Vecteur du terme constant.

    Retourne :
    x (array): Solution du système tridiagonal.

    """
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)

    # Précalcule les coefficients c' et d' pour la méthode de Thomas
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        temp = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / temp
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / temp

    x = np.zeros(n)
    x[-1] = d_prime[-1]

    # Réalise la substitution arrière pour obtenir la solution x
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x

def volume_finis_thomas(sigma, n, T_c, T_a):
    """
    Calcule la distribution de température en utilisant la méthode des volumes finis.

    Paramètres :
    sigma (float): Coefficient de conductivité thermique du matériau.
    n (int): Nombre de volumes de contrôle.
    T_c (float): Température de la condition initiale.
    T_a (float): Température de la condition aux limites.

    Retourne :
    T_tilde (array): Distribution de température calculée.

    """
    # Calcul de delta_x
    delta_x = 1 / n

    # Initialise les vecteurs diagonaux a, b et c pour la matrice tridiagonale, ainsi que le vecteur d
    a = np.full(n, -1)
    b = np.full(n, 2 + sigma * (delta_x**2))
    c = np.full(n, -1)
    d = np.full(n, sigma * (delta_x**2) * T_a)

    # Ajustements pour la première et dernière lignes de la matrice tridiagonale
    b[0] = sigma * (delta_x**2) + 3
    d[0] = 2 * T_c + sigma * (delta_x**2) * T_a
    b[n - 1] = sigma * (delta_x**2) + 3
    d[n - 1] = (2 + sigma * (delta_x**2)) * T_a

    # Résolution du système tridiagonal en utilisant l'algorithme de Thomas
    T_tilde = thomas_algorithm(a, b, c, d)

    return T_tilde

def calculate_MSE(n: int, 
    T_c: float, 
    T_a: float, 
    sigma: float, 
    L: float,
    y_exact: np.ndarray[float, Any]):
    """
    Calcule l'erreur quadratique moyenne (MSE) entre la solution numérique et la solution exacte.

    Paramètres :
    sigma (float): Coefficient de conductivité thermique du matériau.
    n (int): Nombre de volumes de contrôle.
    T_c (float): Température de la condition initiale.
    T_a (float): Température de la condition aux limites.
    y_exact (array): Solution exacte du problème.

    Retourne :
    delta_x (float): Pas d'espace (delta x).
    mse (float): Erreur quadratique moyenne.

    """
    # Calcul de delta_x pour la valeur donnée de n
    delta_x = 1 / n    
    # Calcul de la solution approximative en utilisant la méthode des volumes finis
    approximate_solution = volume_finis_thomas(sigma, n, T_c, T_a)

    # Calcul de l'erreur quadratique moyenne (MSE)
    mse = np.mean((y_exact - approximate_solution)**2)
    
    return delta_x, mse


def evolution_Tnum_k(n: int, 
    T_c: float,
    T_a: float,
    R: float, 
    L: float,
    h: float,
    k: np.ndarray[float, Any]):
    """
    Affiche l'évolution de la distribution de température numérique pour différentes valeurs de conductivité thermique k.

    Cette fonction effectue une simulation numérique de la distribution de température à l'intérieur d'un matériau pour différentes valeurs de conductivité thermique k. Elle utilise la méthode des volumes finis pour résoudre le problème de conduction de la chaleur.

    Paramètres :
    n (int) : Nombre de volumes de contrôle.
    T_c (float) : Température de la condition initiale.
    T_a (float) : Température de la condition aux limites.
    k (array) : Tableau des valeurs de conductivité thermique à tester.

    Remarque :
    - La fonction affiche les résultats sous forme de graphiques, montrant l'évolution de la distribution de température numérique pour différentes valeurs de k, ainsi que la distribution de température pour une valeur spécifique k=420.

    """
    n_1=0.5/n # On commence au milieu du volume K_1 (x=n_1) et on termine à x=(1-1/n) 
    x_k = np.arange(n_1, L, 1/n)
    x_k = np.insert(x_k, [0, n], [0, L])
    S = np.pi * (R)**2 # section de l'ailette
    P = 2 * np.pi *R

    # Affichage de la conduction pour différentes valeurs de k
    for k_value in k:
        sigma_conduction = (P*h) / (k_value * S)
        y_th = volume_finis_thomas(sigma_conduction, n, T_c, T_a)
        y_th = np.insert(y_th, [0, n], [T_c, T_a])        
        plt.plot(x_k, y_th, label=f"$T_{{num}}$ pour k={k_value}")

    # Conduction pour k=420
    k_specific = 420
    sigma_conduction = (P*h) / (k_specific * S)
    y_th = volume_finis_thomas(sigma_conduction, n, T_c, T_a)
    y_th = np.insert(y_th, [0, n], [T_c, T_a])

    plt.plot(x_k, y_th, label=f"$T_{{num}}$ pour k={k_specific}")
    plt.xlabel("x (m)")
    plt.ylabel("T (°C)")
    plt.grid(True)
    plt.legend()
    plt.show()

def MSE_delta_x(sigma, T_c, T_a, y_exact, n_maxi, L):
    """
    Cette fonction calcule l'erreur en fonction de delta_x.

    Paramètres :
        sigma (float): Coefficient de conductivité thermique du matériau.
        T_c (float): Température de la condition initiale.
        T_a (float): Température de la condition aux limites.
        y_exact (array): Solution exacte du problème.
        n_maxi (int): Valeur maximale de n (nombre de volumes de contrôle) à tester.
        L (float): Longueur du matériau.

        Retourne :
        deltax_values (list): Liste des pas d'espace (delta x) testés.
        mse_values (list): Liste des erreurs quadratiques moyennes correspondantes.
    
    """
    n_values = range(5, n_maxi, 1)
    deltax_values = []
    mse_values = []

    for i in n_values:
        x = np.linspace(0, 1, i)
        y_exact = 20 + 80.00363216 * np.exp(-x / 0.2) - 0.0036321593 * np.exp(x / 0.2)
        deltax, mse = calculate_MSE(sigma=sigma, n=i, T_c=T_c, T_a=T_a, y_exact=y_exact, L=L)
        deltax_values.append(deltax)
        mse_values.append(mse)
    return deltax_values, mse_values
