import pandas as pd
import numpy as np

# ==========================================
# 1. Chargement et aperçu
# ==========================================
# a. Chargement du fichier et affichage des 5 premiers jours
df = pd.read_csv('donnees_meteo.csv')
print("--- 1.a. Les 5 premiers jours ---")
print(df.head())

# b. Séparer les colonnes dans des variables (utilisation de tableaux Numpy)
jours = df['Jour'].values
temp = df['Température (°C)'].values
hum = df['Humidité (%)'].values
vent = df['Vitesse du vent (km/h)'].values

# ==========================================
# 2. Statistiques de base
# ==========================================
# a. Calculer la température moyenne du mois
temp_moyenne = np.mean(temp)
print(f"\n--- 2.a. Température moyenne : {temp_moyenne:.2f} °C")

# b. Trouver le jour le plus chaud et le jour le plus froid
jour_plus_chaud = jours[np.argmax(temp)]
jour_plus_froid = jours[np.argmin(temp)]
print(f"--- 2.b. Jour le plus chaud : Jour {jour_plus_chaud} ({np.max(temp)} °C)")
print(f"--- 2.b. Jour le plus froid : Jour {jour_plus_froid} ({np.min(temp)} °C)")

# c. Calculer l'humidité moyenne
hum_moyenne = np.mean(hum)
print(f"--- 2.c. Humidité moyenne : {hum_moyenne:.2f} %")

# ==========================================
# 3. Filtrage des données
# ==========================================
# a. Jours où la température dépasse 24°C
jours_chauds = jours[temp > 24]
print(f"\n--- 3.a. Jours avec température > 24°C : {jours_chauds}")

# b. Jours où l'humidité est < 50% ET la vitesse du vent > 10 km/h
condition_filtrage = (hum < 50) & (vent > 10)
jours_cond = jours[condition_filtrage]
print(f"--- 3.b. Jours (Hum < 50% et Vent > 10 km/h) : {jours_cond}")

# ==========================================
# 4. Vectorisation
# ==========================================
# a. Créer un tableau indice_chaleur
indice_chaleur = temp + 0.1 * hum - 0.05 * vent

# b. Afficher les jours où indice_chaleur dépasse 30
jours_indice_eleve = jours[indice_chaleur > 30]
print(f"\n--- 4.b. Jours avec indice de chaleur > 30 : {jours_indice_eleve}")

# ==========================================
# 5. Transformation
# ==========================================
# a. Ajouter 1°C aux jours où la température est inférieure à 21°C
temp_corrigee = np.where(temp < 21, temp + 1, temp)

# b. Convertir toutes les températures corrigées en Fahrenheit (F = C * 9/5 + 32)
temp_fahrenheit = temp_corrigee * 9/5 + 32
print(f"\n--- 5.b. Températures en Fahrenheit (5 premiers jours) : {temp_fahrenheit[:5]}")

# ==========================================
# 6. Classement
# ==========================================
# a. Trier les jours selon la température croissante (sans boucle)
indices_tri_temp = np.argsort(temp_corrigee)
jours_tries_temp = jours[indices_tri_temp]
print(f"\n--- 6.a. Jours triés par température croissante : \n{jours_tries_temp}")

# b. Extraire les 5 jours les plus humides
indices_tri_hum = np.argsort(hum)
# On prend les 5 derniers indices du tableau trié de façon croissante
jours_plus_humides = jours[indices_tri_hum[-5:]] 
print(f"--- 6.b. Les 5 jours les plus humides : {jours_plus_humides}")

# ==========================================
# 7. Tableaux 2D
# ==========================================
# Création d'un tableau 2D avec Température, Humidité et Vent
donnees_2d = np.column_stack((temp_corrigee, hum, vent))

# a. Normaliser chaque colonne (Min-Max scaling : (x - min) / (max - min))
donnees_min = np.min(donnees_2d, axis=0)
donnees_max = np.max(donnees_2d, axis=0)
donnees_normalisees = (donnees_2d - donnees_min) / (donnees_max - donnees_min)
print(f"\n--- 7.a. Données normalisées (3 premiers jours) : \n{donnees_normalisees[:3]}")

# b. Calculer une corrélation entre température et humidité
# corrcoef renvoie une matrice 2x2, la valeur de corrélation est à l'indice [0, 1] ou [1, 0]
correlation = np.corrcoef(temp_corrigee, hum)[0, 1]
print(f"--- 7.b. Corrélation entre température et humidité : {correlation:.4f}")