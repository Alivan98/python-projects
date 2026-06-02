import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Φόρτωση δεδομένων
df_Total_Accel = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/TotalAcceleration.csv', index_col=False)
df_Accel = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Accelerometer.csv', index_col=False)
df_Gyro = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Gyroscope.csv', index_col=False)
df_Light = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Light.csv', index_col=False)
df_Magnet = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Magnetometer.csv', index_col=False)

# Ορισμός εξαρτημένης μεταβλητής
y = df_Total_Accel['x'].values

# Λίστα ανεξάρτητων μεταβλητών
independent_vars = {
    'Accelerometer X': df_Accel['x'].values,
    'Gyroscope X': df_Gyro['x'].values,
    'Magnetometer X': df_Magnet['x'].values,
    'Light Lux': df_Light['lux'].values
}

# Δοκιμή πολυωνυμικών μοντέλων
for var_name, X in independent_vars.items():
    X = X.reshape(-1, 1)
    
    for degree in [1, 2, 3]:  # 1ου, 2ου και 3ου βαθμού παλινδρόμηση
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        
        print(f"{var_name} - Πολυωνυμικό Μοντέλο {degree}ου βαθμού: R^2 = {r2:.4f}")
    
    # Οπτικοποίηση
    plt.scatter(X, y, label='Data')
    X_sorted = np.sort(X, axis=0)
    X_poly_sorted = poly.fit_transform(X_sorted)
    y_pred_sorted = model.predict(X_poly_sorted)
    plt.plot(X_sorted, y_pred_sorted, color='red', label=f'{degree}-degree Fit')
    plt.xlabel(var_name)
    plt.ylabel('Total Acceleration X')
    plt.title(f'Παλινδρόμηση Total Acceleration X με {var_name}')
    plt.legend()
    plt.show()
