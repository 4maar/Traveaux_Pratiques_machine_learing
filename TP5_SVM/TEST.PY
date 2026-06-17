import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

# Lire les données
raw_data = pd.read_csv('creditcard.csv')
print(raw_data.head(10))

# Analyse de données
print(raw_data['Class'].value_counts().values)

# L'ensemble de classes
labels = raw_data['Class'].unique()

# Le nombre de chaque classe
sizes = raw_data['Class'].value_counts().values

# Plot
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Nombre de valeurs de chaque classe')
plt.show()

correlation_values = raw_data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10, 6))


# Evaluer le modèle SVM
y_pred_svm = svm.decision_function(X_test)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))