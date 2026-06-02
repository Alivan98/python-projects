import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df_Total_Accel = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/TotalAcceleration.csv', index_col=False)
df_Accel = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Accelerometer.csv', index_col=False)
df_Gyro = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Gyroscope.csv', index_col=False)
df_Light = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Light.csv', index_col=False)
df_Magnet = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Magnetometer.csv', index_col=False)

data = pd.DataFrame({
    'Accelerometer X': df_Accel['x'].values,
    'Gyroscope X': df_Gyro['x'].values,
    'Magnetometer X': df_Magnet['x'].values,
    'Light Lux': df_Light['lux'].values
})

y = df_Total_Accel['x'].values

degrees = [1, 2, 3]

for feature in data.columns:
    X = data[[feature]].values 

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label="Δεδομένα")

    for d in degrees:
        poly = PolynomialFeatures(degree=d)
        X_poly = poly.fit_transform(X)

        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)

        r2 = r2_score(y, y_pred)
        print(f"Μεταβλητή: {feature}, Βαθμός: {d}, R²: {r2:.4f}")
        
        X_poly_const = sm.add_constant(X_poly)
        ols_model = sm.OLS(y, X_poly_const).fit()
        print(ols_model.summary())

        sorted_idx = np.argsort(X[:, 0])  
        plt.plot(X[sorted_idx], y_pred[sorted_idx], label=f"Βαθμός {d} (R²={r2:.2f})")

    plt.title(f"Πολυωνυμική Παλινδρόμηση για {feature}")
    plt.xlabel(feature)
    plt.ylabel("Total Acceleration X")
    plt.legend()
    plt.show()