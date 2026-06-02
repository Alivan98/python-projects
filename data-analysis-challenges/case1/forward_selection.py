import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations


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

def forward_selection(X, y, significance_level=0.05):
    initial_features = []
    remaining_features = list(X.columns)
    best_features = []
    
    while len(remaining_features) > 0:
        best_pval = float('inf')
        best_feature = None
        
        for feature in remaining_features:
            selected_features = initial_features + [feature]
            X_selected = sm.add_constant(X[selected_features])
            model = sm.OLS(y, X_selected).fit()
            pval = model.pvalues[feature]
            
            if pval < best_pval:
                best_pval = pval
                best_feature = feature
        
        if best_pval < significance_level:
            initial_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_features.append(best_feature)
        else:
            break

    return best_features

best_features = forward_selection(data, y)
print("Οι επιλεγμένες μεταβλητές είναι:", best_features)

X_final = sm.add_constant(data[best_features])
final_model = sm.OLS(y, X_final).fit()

print(final_model.summary())
