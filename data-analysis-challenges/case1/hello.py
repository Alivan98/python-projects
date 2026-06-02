import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn as sl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score
import statsmodels.api as sm
import os

df_Total_Accel=pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/TotalAcceleration.csv',index_col=False)
df_Accel=pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Accelerometer.csv',index_col=False)
df_Gyro=pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Gyroscope.csv',index_col=False)
df_Light=pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Light.csv',index_col=False)
df_Magnet=pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Magnetometer.csv',index_col=False)

print("===== Total Acceleration =====")
print(df_Total_Accel)

print("===== Accelerometer =====")
print(df_Accel)

print("===== Gyroscope =====")
print(df_Gyro)

print("===== Light =====")
print(df_Light)

print("===== Magnetometer =====")
print(df_Magnet)

x1=df_Total_Accel.x.values
y1=df_Light.lux.values
x1=x1.reshape(200,1)
y1=y1.reshape(200,1)
lr1=linear_model.LinearRegression().fit(x1,y1)

print(lr1.coef_)
print(lr1.intercept_)
X12 = sm.add_constant(x1)
est = sm.OLS(y1, X12)
est2 = est.fit()
print(est2.summary())

y_pred1=lr1.predict(x1)
plt.scatter(x1,y1)
plt.plot(x1,y_pred1,color='red')
plt.ylabel("Total Acceleration (X)")
plt.xlabel("Light (lux)")
plt.title(f"Γραμμική Παλινδρόμηση: Total Acceleration(X) ~ Light(lux)")
plt.show()