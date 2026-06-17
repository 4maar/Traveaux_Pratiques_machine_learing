import pandas as pd # [cite: 54]

# Création du DataFrame fourni dans l'énoncé [cite: 55-61]
data = {
    "Nom": ["Ali", "Sara", "Youssef", "Meriem", "Nadia"],
    "Age": [20, 21, 19, 22, 20],
    "Note": [14, 17, 12, 15, 18],
    "Absences": [2, 0, 5, 1, 0]
}
df = pd.DataFrame(data)

# 1. Afficher la table complète [cite: 62]
print("--- 1. Table complète ---")
print(df)

# 2. Afficher les trois premières lignes [cite: 63]
print("\n--- 2. Trois premières lignes ---")
print(df.head(3))

# 3. Afficher uniquement les colonnes: Nom et Note [cite: 64]
print("\n--- 3. Colonnes Nom et Note ---")
print(df[['Nom', 'Note']])

# 4. Afficher les étudiants ayant une note supérieure ou égale à 15 [cite: 65]
print("\n--- 4. Étudiants avec Note >= 15 ---")
print(df[df['Note'] >= 15])

# 5. Afficher les étudiants ayant note >= 15 et absences = 0 [cite: 66-69]
print("\n--- 5. Étudiants avec Note >= 15 ET Absences == 0 ---")
print(df[(df['Note'] >= 15) & (df['Absences'] == 0)])

# 6. Calculer moyenne des notes, note maximale, note minimale [cite: 70-73]
moyenne_notes = df['Note'].mean()
note_max = df['Note'].max()
note_min = df['Note'].min()

print("\n--- 6. Statistiques des notes ---")
print(f"Moyenne des notes : {moyenne_notes}")
print(f"Note maximale : {note_max}")
print(f"Note minimale : {note_min}")

# 7. Ajouter une nouvelle colonne: Résultat ("Admis" si Note >= 10, "Ajourné" sinon) [cite: 74-78]
# On utilise une compréhension de liste ou la fonction apply de Pandas
df['Résultat'] = df['Note'].apply(lambda x: "Admis" if x >= 10 else "Ajourné")

print("\n--- 7. Table avec la colonne Résultat ---")
print(df)