import numpy as np
import matplotlib.pyplot as plt
# Définition des paramètres de l'ailette

R=4 * 10**-3 # Rayon de la section de l'ailette
P = 2 * np.pi *R  # Périmètre de la section de l'ailette
S = np.pi * (R)**2 # section de l'ailette
k=np.arange(16, 420, 50) # liste des conductivités thermique allant de 16 à 420 avec un pas de 50
H = 10 # Coefficient de convecto-convection
K_ENONCE=200 # Conductivité thermique
SIGMA_TEST = (P * H) / (S * K_ENONCE)
A=1/np.sqrt(SIGMA_TEST)
class TemperatureSimulation:
    def __init__(self, a, b, n):
        """Initialisation des paramètres de la simulation"""
        self.a = a
        self.b = b
        self.x = np.linspace(0, 1, 100)
        self.a_2 = A**2
        self.a_3 = A**3
        self.h = 1/n
        self.num_integrale = (1/n)**3
        self.dem_integrale = 24*(n**2)

    def calculate_T(self, x):
        """Calcul de la température en fonction de la position"""
        return self.a * np.exp(-x / A) + self.b * np.exp(x / A) + 20

    def calculate_second_derivative(self):
        """Calcul de la dérivée seconde la température"""
        return self.a/self.a_2 * np.exp(-self.x / A) + self.b/self.a_2 * np.exp(self.x / A)

    def calculate_third_derivative(self):
        """Calcul de la dérivée troisième de la température"""
        return -self.a/self.a_3 * np.exp(-self.x / A) + self.b/self.a_3 * np.exp(self.x / A)

    def plot_T(self, x_value, color, label, text_color):
        """Affichage du graphe de la température en fonction de la position"""
        T_value = self.calculate_T(x_value)

        plt.plot(self.x, self.calculate_T(self.x), color=color, label=label)
        plt.xlabel("x (m)")
        plt.ylabel("T (°C)")

        plt.text(0.3, T_value, f'T(x={x_value}) = {T_value:.2f} °C', ha='right', va='bottom', color=text_color)

        plt.axhline(y=T_value, color=text_color, linestyle='--', label="Asymptote horizontale")

        plt.grid(True)
        plt.legend()
        plt.show()

    def visualize_derivative(self, derivative, label):
        """Affichage du graphe de la dérivée spécifiée"""
        plt.plot(self.x, derivative, color="green", label=label)
        plt.xlabel("x (m)")
        plt.ylabel(label + " (°C)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_error_derive_gauche(self, points_x, derivative):
        """Calcul et affichage de l'erreur pour la dérivée gauche"""
        error_values = []
        sup_T_second=[]
        for i in range(len(points_x)-1):
            y_value = np.interp(points_x[i], self.x, derivative)

            plt.scatter(points_x[i], y_value, color='red', marker='o')
            rounded_y_value = round(y_value, 2)

            plt.text(points_x[i]+0.01, y_value, f'$\sup_{{t \in [{points_x[i]}, {points_x[i+1]}]}} T\'\'(t) = {rounded_y_value}$', fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='blue')
            sup_T_second.append(y_value)
            error = rounded_y_value * (self.h/2)
            error_values.append(error)
        print("Erreur sur la dérivée à gauche :" , error_values)
        print("Borne sup T'' sur chaque intervalle :",  sup_T_second)
        return error_values

    def error_integrale(self, points_x, derivative):
        """Calcul et affichage de l'erreur sur intégrale """
        error_values = []
        for i in range(len(points_x)-1):
            y_value = np.interp(points_x[i], self.x, derivative)

            plt.scatter(points_x[i], y_value, color='red', marker='o')
            rounded_y_value = round(y_value, 2)

            plt.text(points_x[i]+0.01, y_value, f'$\sup_{{t \in [{points_x[i]}, {points_x[i+1]}]}} T\'\'(t) = {rounded_y_value}$', fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='blue')

            error = rounded_y_value * (self.num_integrale/self.dem_integrale)
            error_values.append(error)
        print("Erreur sur l'intégrale:" , error_values)

        return error_values

    def error_derive_center(self, points_x, derivative):
        """Calcul et affichage de l'erreur pour la dérivée centrée"""
        error_values = []
        sup_T_third = []
        for i in range(len(points_x)-1):
            y_value = np.interp(points_x[i+1], self.x, derivative)

            plt.scatter(points_x[i+1], y_value, color='red', marker='o')
            rounded_y_value = round(y_value, 2)

            plt.text(points_x[i]+0.1, y_value, f'$\sup_{{t \in [{points_x[i]}, {points_x[i+1]}]}}  T\'\'\'(t) = {rounded_y_value}$', fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='blue')
            sup_T_third.append(y_value)
            error =  (self.h**2/6) * np.abs(rounded_y_value)
            error_values.append(error)
        print("Erreur sur la dérivée centrée :" , error_values)
        print("Borne sup T''' sur chaque intervalle :",  sup_T_third)            
        return error_values

    def plot_error_intervals(self, points_x, derivative, error_label, label, error_values):
        """Affichage du graphe avec les intervalles et les erreurs"""
        plt.xlabel("x (m)")
        plt.ylabel(f"{label} (°C)")
        plt.plot(self.x, derivative, color="green", label=label)
        plt.grid(True)
        plt.legend()
        plt.show()

        # Visualisation de l'erreur de la dérivée pour chaque intervalle
        interval_labels = [f'[{points_x[i]}, {points_x[i+1]}]' for i in range(len(points_x)-1)]
        plt.plot(interval_labels, error_values, marker='o', linestyle='-', color='purple', label=error_label)
        plt.xlabel('Intervalles')
        plt.ylabel('Valeurs')
        plt.grid(True)
        plt.legend()
        plt.show()

