import streamlit as st
import joblib
import numpy as np

# 1. Charger les paramètres calculés par ton script
sauvegarde = joblib.load('TP2/model_maison.pkl')
t = sauvegarde['t']
mu = sauvegarde['mu']
sigma = sauvegarde['sigma']

# 2. Interface Web
st.title("🏡 Simulateur de Prix Immobilier")
st.write("Ce modèle utilise une régression linéaire avec descente de gradient pour estimer le prix.")

# Formulaire
taille = st.number_input("Superficie de la maison (en m² / sqft)", min_value=50, max_value=10000, value=1650)
chambres = st.number_input("Nombre de chambres", min_value=1, max_value=10, value=3)

if st.button("Calculer l'estimation"):
    # Application de TA formule mathématique exacte
    prix_estime = t[0] + t[1] * (taille - mu[0]) / sigma[0] + t[2] * (chambres - mu[1]) / sigma[1]
    
    # Affichage du résultat
    st.success(f"Le prix estimé pour cette propriété est de : ${prix_estime:,.0f}")