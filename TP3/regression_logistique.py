import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

df = pd.read_csv("ChurnData.csv")
df.head()
cdf = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
cdf['churn'] = cdf['churn'].astype('int')
print(cdf)
cdf = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
cdf['churn'] = cdf['churn'].astype('int')
print(cdf)
X = cdf[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']]
y = cdf['churn']
X_norm = StandardScaler().fit_transform(X)
print(X_norm[0:5])
X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)
print("\n=================================================================\n")
print("Classificateur de Regression Logistique\n")
LR = LogisticRegression().fit(X_train,y_train)
yhat = LR.predict(X_test)
print(yhat[:10])
yhat_prob = LR.predict_proba(X_test)
print(yhat_prob[:10])
coefficients = pd.Series(LR.coef_[0], index=cdf.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()
log_loss(y_test, yhat_prob)
print("\n log_loss is " , log_loss(y_test,yhat_prob))
