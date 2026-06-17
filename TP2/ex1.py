import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('FuelConsumption.csv',encoding='latin1') # encoding='latin1' pour les accents 
print(df.head())
print (df.describe())
cdf = df[['Engine size (L)','Cylinders','Combined (L/100 km)','CO2 emissions (g/km)']]
print(cdf.head(9))
print(cdf.hist())
plt.show()
plt.scatter(cdf['Combined (L/100 km)'], cdf['CO2 emissions (g/km)'], color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("emissions")
plt.title("Fuel Consumption vs CO2 Emissions")
plt.show()
plt.scatter(cdf['Engine size (L)'], cdf['CO2 emissions (g/km)'], color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emissions")
plt.title("Engine Size vs CO2 Emissions")
plt.show()


print("---------------############------------------")
print("Séparer les données en données d/'apprentissageet autres de test\n")


from sklearn.model_selection import train_test_split
X=cdf[['Engine size (L)']] # la caractéristique (Dataframe)
y=cdf['CO2 emissions (g/km)'] # l'etiquette (Series)
X_train,X_test,y_Train,y__Test=train_test_split(X,y,test_size=0.2)


print("\n---------------############------------------\n")
print("Entraîner le modèle de régression linéaire sur les données d'apprentissage\n")


from sklearn.linear_model import LinearRegression
modele=LinearRegression().fit(X_train,y_Train)
# afficher les coefficients
print(modele.intercept_,modele.coef_)
teta = np.concatenate(([modele.intercept_], modele.coef_))
print(teta)
plt.scatter(X_train, y_Train, color='blue')
plt.plot(X_train, teta[1]*X_train + teta[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Regression Lineaire")
plt.show()


print("\n---------------############------------------\n")
print("Évaluer le modèle sur les données de test\n")


from sklearn.metrics import mean_absolute_error, r2_score
y_pred=modele.predict(X_test)
print("L'erreur quadratique moyenne : ", mean_absolute_error(y_pred,y__Test))
print("Score R2 : ",r2_score(y_pred,y__Test))


print("\n---------------############------------------\n")
print("\n regression lineare multiple\n")


df = pd.read_csv('FuelConsumption.csv', encoding='latin1')
print(df.head())
cdf = df[['Engine size (L)', 'Cylinders', 'Combined (L/100 km)', 'CO2 emissions (g/km)']]
print(cdf.head(9))
X = cdf[['Engine size (L)', 'Cylinders', 'Combined (L/100 km)']]# la caractéristiqu
y=cdf['CO2 emissions (g/km)'] # l'étiquette
from sklearn.model_selection import train_test_split
X_train,X_test,y_Train,y__Test=train_test_split(X,y,test_size=0.2)

print("\n---------------############------------------\n")
print("\n entrainer le modele\n")


from sklearn import linear_model
regr= linear_model.LinearRegression()
regr.fit(X_train, y_Train)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
from sklearn.metrics import mean_absolute_error, r2_score
y_pred=regr.predict(X_test)
print("L'erreur quadratique moyenne : ", mean_absolute_error(y_pred,y__Test))
print("Score R2 : ",r2_score(y_pred,y__Test))
