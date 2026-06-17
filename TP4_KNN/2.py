import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('teleCust1000t.csv')
df.head()
df['custcat'].value_counts()
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
X = df.drop('custcat', axis=1)
y = df['custcat']
X_norm = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)
mean_acc = []
std_acc = []

for k in range(1, 100):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_model = knn_classifier.fit(X_train, y_train)
    yhat = knn_model.predict(X_test)
    mean_acc[k - 1] = accuracy_score(y_test, yhat)
    


plt.xlabel("Valeur de k")
plt.ylabel("Précision")
plt.title("Précision du modèle KNN pour k de 1 à 99")
plt.show()



