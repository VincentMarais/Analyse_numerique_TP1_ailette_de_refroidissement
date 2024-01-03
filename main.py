"""
Programme TP1 analyse numérique
"""
import numpy as np
import matplotlib.pyplot as plt
from core.functions_volumes_finis_ailette import  MSE_delta_x, volume_finis_thomas, evolution_Tnum_k
from core.bonus_simulation import TemperatureSimulation
# Définition des paramètres de l'ailette

R=4 * 10**-3 # Rayon de la section de l'ailette
P = 2 * np.pi *R  # Périmètre de la section de l'ailette
S = np.pi * (R)**2 # section de l'ailette
k=np.arange(16, 420, 50) # liste des conductivités thermique allant de 16 à 420 avec un pas de 50
H = 10 # Coefficient de convecto-convection
K_ENONCE=200 # Conductivité thermique
SIGMA_TEST = (P * H) / (S * K_ENONCE)

# Paramètres de la simulation
T_C = 100
T_A = 20
n = input("Entrer la valeur de n : ")
n=int(n)
n_1 = 0.5 / n # Dans nos calculs, nous avons calculé nos températures aux centres des volumes de contrôles
L=1 # Longueur de l'ailette 1m
delta_x=1/n
# Dans nos calculs, nous avons calculé nos températures aux centres des volumes de contrôles
x = np.arange(n_1, L, delta_x)
# Paramètre de la solution exate avec les conditions aux limites du prof
A_1 = 80.00363216
B_1 = - 0.0036321593


# Calcul de la solution numérique
y_th = volume_finis_thomas(SIGMA_TEST, n, T_C, T_A)

# Correction des valeurs de x et y_th
x = np.insert(x, [0, n], [0, L])
y_th = np.insert(y_th, [0, n], [T_C, T_A])

# Calcul de la solution exacte
y_exact = T_A + A_1 * np.exp(-x / 0.2) + B_1 * np.exp(x / 0.2)

# Calcul de l'erreur relative en tout point
erreur_relative = np.abs((y_th - y_exact) / y_exact)

# Affichage des graphiques
# Graphe question 4-5
# Créez une figure avec 2 sous-graphiques
plt.figure(figsize=(12, 6))  # Ajustez la taille de la figure

# Créez une figure avec 2 sous-graphiques côte à côte
plt.figure(figsize=(12, 6))  # Ajustez la taille de la figure

# Sous-plot 1 (à gauche)
plt.subplot(1, 2, 1)  # 1 ligne, 2 colonnes, sous-plot 1
plt.plot(x, y_th, label="$T_{num}$")
plt.plot(x, y_exact, label="$T_{ex}$")
plt.xlabel("x (m)")
plt.ylabel("T (°C)")
plt.legend()
plt.grid(True)

# Sous-plot 2 (à droite)
plt.subplot(1, 2, 2)  # 1 ligne, 2 colonnes, sous-plot 2
plt.plot(x, erreur_relative, label="Erreur relative")
plt.xlabel("x (m)")
plt.ylabel("Erreur relative")
plt.grid(True)
plt.legend()

# Affichez la figure avec les sous-graphiques
plt.tight_layout()  # Pour éviter que les labels se chevauchent
plt.show()


[deltax_values, mse_values]=MSE_delta_x(SIGMA_TEST, T_C, T_A, y_exact, n_maxi=50, L=L)
# Graphe question 5.2
plt.figure(figsize=(10, 6))
plt.plot(np.flip(deltax_values), np.flip(mse_values), marker='o')
plt.xlabel('$\delta x$')
plt.ylabel('$E(\delta x)$')
plt.title('$E(\delta x)$')
plt.grid(True)
plt.show()


# Graphe question 7
evolution_Tnum_k(R=R, L=L, h=H, n=100, T_c=100, T_a=20, k=k)

# Bonus simulation
print("Simulation bonus")
# Paramètre de la solution exate mes conditions aux limites :
A_2=79.99644008
B_2=0.0035599153

# Choix utilisateur
choise=input("Veut tu y simulation adaptative ? 'Oui' ou 'Non' : ")
while choise not in ['Oui', 'Non']:
    solution = input("Veuillez écrire Oui ou Non : ")
if choise=='Oui':
    x_error=[i/n for i in range(n+1)]
else:
    N_DEFAUT=5
    x_error=[i/N_DEFAUT for i in range(N_DEFAUT+1)]

# Première simulation
sim_1 = TemperatureSimulation(A_2, B_2, n)
sim_1.plot_T(1, "red", "T(x)", "green")

# Deuxième simulation
sim_2 = TemperatureSimulation(A_1, B_1, n)
sim_2.plot_T(1, "red", "T(x)", "green")

# Calcul de T''(x) et visualisation des points et de l'erreur intégrale
sim_2.visualize_derivative(sim_2.calculate_second_derivative(), "T''")
sim_2.plot_error_intervals(x_error, sim_2.calculate_second_derivative(), "Erreur intégrale", "T''", sim_2.error_integrale(x_error, sim_2.calculate_second_derivative()))

# Calcul de T'''(x) et visualisation
sim_2.visualize_derivative(sim_2.calculate_third_derivative(), "T'''")

# Visualisation de l'erreur de la dérivée centrée et de la dérivée gauche pour chaque intervalle
sim_2.plot_error_intervals(x_error, sim_2.calculate_third_derivative(), "Erreur dérivée centrée", "T'''", sim_2.error_derive_center(x_error, sim_2.calculate_third_derivative()))
sim_2.plot_error_intervals(x_error, sim_2.calculate_second_derivative(), "Erreur dérivée gauche", "T''", sim_2.plot_error_derive_gauche(x_error, sim_2.calculate_second_derivative()))
