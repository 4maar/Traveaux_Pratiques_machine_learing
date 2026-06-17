import numpy as np
import matplotlib.pyplot as plt

e = np.linspace(-5, 5, 200) #

mae = np.abs(e)
plt.figure(figsize=(6, 4))
plt.plot(e, mae, label='MAE', color='blue')
plt.title('MAE (Mean Absolute Error)')
plt.xlabel('Erreur (e)')
plt.ylabel('L(e)')
plt.grid(True)
plt.show()

mse = e**2
plt.figure(figsize=(6, 4))
plt.plot(e, mse, label='MSE', color='red')
plt.title('MSE (Mean Squared Error)')
plt.xlabel('Erreur (e)')
plt.ylabel('L(e)')
plt.grid(True)
plt.show()

# 3. 0/1 Loss : L(e) = 0 si e=0, 1 si e!=0
zero_one_loss = np.where(e == 0, 0, 1)
plt.figure(figsize=(6, 4))
plt.plot(e, zero_one_loss, label='0/1 Loss', color='green')
plt.title('0/1 Loss')
plt.xlabel('Erreur (e)')
plt.ylabel('L(e)')
plt.grid(True)
plt.show()

hinge_loss = np.maximum(0, 1 - e)
plt.figure(figsize=(6, 4))
plt.plot(e, hinge_loss, label='Hinge Loss', color='purple')
plt.title('Hinge Loss')
plt.xlabel('Erreur (e)')
plt.ylabel('L(e)')
plt.grid(True)
plt.show()