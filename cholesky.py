import numpy as np
import matplotlib.pyplot as plt


def cholesky_solve(A, b):
    # Vérification de la symétrie de A
    if not np.allclose(A, A.T): # fonction qui compare la matrice transposé et la matrice initiale 
        raise ValueError("La matrice que vous avez entrée n'est pas symétrique")

    # Décomposition de Cholesky
    L = np.linalg.cholesky(A)

    # Résolution de L y = b
    y = np.linalg.solve(L, b)

    # Résolution de L.T x = y pour trouver x
    x = np.linalg.solve(L.T, y)

    return x


# J'ai une équation discrétisé du volume K_1 vaut T_1 (sigma*delta_x**2 + 1 ) - T_2 = sigma*demt
def maillage_volume_fini(n,delta_x,sigma):
    
    pass
# Exemple de système d'équations

A = np.array([[4, -1,0,0,0], [-1, 3,-1,0,0], [0, -1,1+2,-1,0], [0, 0,-1,1+2,-1] ,[0, 0,0,-1,4]], dtype=float)
b = np.array([220, 20, 20, 20, 60], dtype=float)

# Résolution du système
T = cholesky_solve(A, b)
print("Solution T:", T)

x=np.linspace(0,1,5)
y_th=T
y_exate= 20 + 80.00363216 * np.exp(-x / 0.2) -0.0036321593 * np.exp(x / 0.2)

plt.plot(x, y_th, label="Température méthode des éléments finis")
plt.plot(x, y_exate, label="Solution exate de la température")
plt.xlabel("x(m)")
plt.ylabel("T (°C)")

plt.show()



