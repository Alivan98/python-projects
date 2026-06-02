import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Φόρτωση των αρχείων CSV
df_Total_Accel = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/TotalAcceleration.csv', index_col=False)
df_Accel = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Accelerometer.csv', index_col=False)
df_Gyro = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Gyroscope.csv', index_col=False)
df_Light = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Light.csv', index_col=False)
df_Magnet = pd.read_csv('C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400/Magnetometer.csv', index_col=False)

# Ορισμός της εξαρτημένης μεταβλητής (Y) - Total Acceleration (X)
y = df_Total_Accel.x.values.reshape(-1, 1)

# Ορισμός των ανεξάρτητων μεταβλητών (X)
X = np.column_stack([
    df_Accel.x.values,       # Accelerometer X
    df_Gyro.x.values,        # Gyroscope X
    df_Magnet.x.values,      # Magnetometer X
    df_Light.lux.values      # Light Lux
])

# Δημιουργία μοντέλου πολλαπλής γραμμικής παλινδρόμησης
lr = LinearRegression().fit(X, y)

# Εκτύπωση συντελεστών και τέμνουσας
print("Συντελεστές (β):", lr.coef_)
print("Τέμνουσα (Intercept):", lr.intercept_)

# Στατιστική ανάλυση με OLS (Ordinary Least Squares)
X_sm = sm.add_constant(X)  # Προσθήκη σταθεράς (intercept)
est = sm.OLS(y, X_sm).fit()
print(est.summary())

# Πρόβλεψη τιμών
y_pred = lr.predict(X)

# Σχεδίαση γραφήματος
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Πραγματικές τιμές (Total Acceleration X)")
plt.ylabel("Προβλεπόμενες τιμές (Total Acceleration X)")
plt.title("Πολλαπλή Γραμμική Παλινδρόμηση: Total Acceleration(X) ~ (Accel X, Gyro X, Magnet X, Light lux)")
plt.grid()
plt.show()