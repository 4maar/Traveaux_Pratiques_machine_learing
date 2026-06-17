import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#explore les données

df=pd.read_csv('FoodTruck.txt', encoding='latin')
print("\n",df.head())
print("\n",df.describe())
plt.scatter(df['population'], df['benefice'], color='blue')
plt.xlabel("population (en milliers)")
plt.ylabel("benefice (en milliers de dollars)")

#separe les donnees en donnes d'apprentissage et de test

from sklearn.model_selection import train_test_split
x=df[['population']]
y=df['benefice']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#entrainemant de modele

from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(x_train,y_train)
print("\n",model.intercept_,model.coef_)
teta=np.concatenate(([model.intercept_],model.coef_))
plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,teta[1]*x_train+teta[0],'-r')
plt.show()

#evaluation du modele
from sklearn.metrics import mean_absolute_error , r2_score
y_pred=model.predict(x_test)
print("\nl'erreur absolute moyenne : ",mean_absolute_error(y_pred,y_test))
print("\nscore r2 : " , r2_score(y_pred,y_test)) 

