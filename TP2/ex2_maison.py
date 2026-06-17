import numpy as np
import matplotlib.pyplot as plt

# Data
d = np.loadtxt('Maisons.txt', delimiter=',')
X, y, m = d[:, :2], d[:, 2], len(d)

print("Aperçu des données (taille, chambres, prix):")
print(f"{'Taille':>8} {'Chambres':>10} {'Prix':>12}")
for row in d[:5]: print(f"{row[0]:>8.0f} {row[1]:>10.0f} {row[2]:>12.0f}")
print(f"... ({m} exemples au total)\n")

# Normalize
mu, sigma = X.mean(axis=0), X.std(axis=0)
Xn = (X - mu) / sigma
Xb = np.column_stack([np.ones(m), Xn])

# Gradient descent
t, costs = np.zeros(3), []
for _ in range(1000):
    e = Xb @ t - y
    t -= 0.01 * Xb.T @ e / m
    costs.append(e @ e / (2 * m))

print(f"θ₀={t[0]:.2f}  θ₁={t[1]:.2f}  θ₂={t[2]:.2f}")

# Metrics
yp = Xb @ t
r2  = 1 - ((y - yp)**2).sum() / ((y - y.mean())**2).sum()
mse = ((y - yp)**2).mean()
print(f"R²={r2:.4f}  MSE={mse:.2f}  RMSE={mse**0.5:.2f}\n")

# Predict function
predict = lambda size, rooms: t[0] + t[1]*(size-mu[0])/sigma[0] + t[2]*(rooms-mu[1])/sigma[1]

print("--- Prédictions ---")
for size, rooms in [(1650,3),(2000,3),(3000,4),(1200,2),(4000,5)]:
    print(f"Taille={size} m², {rooms} chambres => Prix estimé: ${predict(size,rooms):,.0f}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(X[:,0], y, c='steelblue', s=40, edgecolors='white', lw=0.5)
xl = np.linspace(X[:,0].min(), X[:,0].max(), 200)
axes[0].plot(xl, predict(xl, X[:,1].mean()), 'tomato', lw=2)
axes[0].set(xlabel='Taille (m²)', ylabel='Prix ($)', title='Taille vs Prix')

axes[1].scatter(X[:,1], y, c='mediumpurple', s=40, edgecolors='white', lw=0.5)
axes[1].set(xlabel='Nb chambres', ylabel='Prix ($)', title='Chambres vs Prix')

axes[2].plot(costs, 'seagreen', lw=2)
axes[2].set(xlabel='Itérations', ylabel='J(θ)', title='Convergence')

plt.tight_layout()
plt.savefig('maisons_result.png', dpi=150)
plt.show()