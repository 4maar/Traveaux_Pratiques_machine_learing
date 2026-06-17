import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
df = pd.read_csv("ChurnData.csv" ,encoding="latin")
cdf=df[["tenure","age","address","income","ed","employ","equip","callcard","wireless","churn"]]
cdf["churn"]=cdf["churn"].astype('int')
x=cdf[["tenure","age","address","income","ed","employ","equip","callcard","wireless"]]
y=cdf["churn"]
x_norm=StandardScaler().fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_norm,y,test_size=0.2)
lr=LogisticRegression().fit(x_train , y_train)
yhit=lr.predict(x_test)
yhit_pro=lr.predict_proba(x_test)
coefficient=pd.Series(lr.coef_[0],index=cdf.columns[:-1])
coefficient.sort_values().plot(kind="barh")
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.title("Logistic Regression Coefficients")
plt.show()
log_loss(y_test,yhit_pro)
print("\n log_loss is " , log_loss(y_test,yhit_pro))