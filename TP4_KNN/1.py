import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('teleCust1000t.csv', encoding = 'latin' )
print(df.head())
print(df['custcat'].value_counts())
correlation_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap="coolwrm",fmt=".2f",linewidths=0.5)
plt.show()
x = df.drop('custcat' , axis=1)
y = df['custcat']
x_norm = StandardScaler().fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_norm,y,test_size=0.2,random_state=4)
k=6
knn=KNeighborsClassifier(n_neighbors=k)
knn_model= knn.fit(x_train,y_train)
yhit= knn_model.predict(x_test)
print(accuracy_score(x_test,yhit))
