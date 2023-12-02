import numpy as np
import matplotlib.pyplot as plt
from founctions_volumes_finis_ailette import  MSE_delta_x, volume_finis_thomas, evolution_Tnum_k

# Définition des paramètres
R=4 * 10**-3 # Rayon de la section de l'ailette
P = 2 * np.pi *R  # Périmètre de la section de l'ailette
S = np.pi * (R)**2 # section de l'ailette
k=np.arange(16, 420, 50) # liste des conductivités thermique allant de 16 à 420 avec un pas de 50

# Calcul de sigma_test
H = 10
K_ENONCE=200
HAUT = P * H
BAS = S * K_ENONCE
sigma_test = HAUT / BAS

# Paramètres de la simulation
T_c = 100
T_a = 20
n = 500
n_1 = 0.5 / n # Dans nos calculs, nous avons calculé nos températures aux centres des volumes de contrôles
L=1 # Longueur de l'ailette 1m
delta_x=1/n
# Dans nos calculs, nous avons calculé nos températures aux centres des volumes de contrôles
x = np.arange(n_1, L, delta_x) 

# Calcul de la solution numérique
y_th = volume_finis_thomas(sigma_test, n, T_c, T_a)

# Correction des valeurs de x et y_th
x = np.insert(x, [0, n], [0, L])
y_th = np.insert(y_th, [0, n], [T_c, T_a])

# Calcul de la solution exacte
y_exact = 20 + 80.00363216 * np.exp(-x / 0.2) - 0.0036321593 * np.exp(x / 0.2)

# Calcul de l'erreur relative en tout point
erreur_relative = np.abs((y_th - y_exact) / y_exact)

# Affichage des graphiques
# Graphe question 4-5
plt.figure(figsize=(8, 6))  # Adjust the figure size
plt.plot(x, y_th, label="$T_{num}$")
plt.plot(x, y_exact, label="$T_{ex}$")
plt.xlabel("x (m)")
plt.ylabel("T (°C)")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(x, erreur_relative, label="Erreur relative")
plt.xlabel("x (m)")
plt.ylabel("Erreur relative")
plt.grid(True)
plt.legend()
plt.show()


[deltax_values, mse_values]=MSE_delta_x(sigma_test, T_c, T_a, y_exact, n_maxi=50, L=L)
# Graphe question 5.2
plt.figure(figsize=(10, 6))
plt.plot(np.flip(deltax_values), np.flip(mse_values), marker='o')
plt.xlabel('$\delta x$')
plt.ylabel('$E(\delta x)$')
plt.title('$E(\delta x)$')
plt.grid(True)
plt.show()


# Graphe question 7
evolution_Tnum_k(R=R, L=L, h=H, n=500, T_c=100, T_a=20, k=k)
