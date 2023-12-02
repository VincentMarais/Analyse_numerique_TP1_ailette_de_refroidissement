import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
T_1 = 20 + 80 * np.exp(-x / 0.0565) + 0.0000000003 * np.exp(x / 0.05265)
plt.plot(x, T_1, color="red", label="T(x)")
plt.xlabel("x(m)")
plt.ylabel("T (°C)")
# Ajouter la valeur de T en x=1
x_value = 1
T_value = 20 + 79.99644008 * np.exp(-x_value / 0.2) + 0.0035599153 * np.exp(x_value / 0.2)
plt.text(0.3, T_value, f'T(x=1) = {T_value:.2f} °C', ha='right', va='bottom', color="green")
plt.axhline(y=T_value, color='green', linestyle='--', label="Asymptote horizontale")

plt.axhline(y=20, color='blue', linestyle='--', label="Température ambiante")
plt.legend()



plt.show()


# Simulation seconde fonction :
x = np.linspace(0, 1, 100)
T_1 = 20 + 80.00363216 * np.exp(-x / 0.2) -0.0036321593 * np.exp(x / 0.2)
plt.plot(x, T_1, color="red", label="T(x)")
plt.xlabel("x(m)")
plt.ylabel("T (°C)")
x_value = 1
T_value = 20 + 80.00363216 * np.exp(-x_value / 0.2) -0.0036321593 * np.exp(x_value / 0.2)
plt.text(0.3, T_value, f'T(x=1) = {T_value:.2f} °C', ha='right', va='bottom', color="green")
plt.axhline(y=T_value, color='green', linestyle='--', label="Asymptote horizontale")

plt.legend()
plt.show()
