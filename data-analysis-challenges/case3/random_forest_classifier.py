import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data and preprocessing
data = pd.read_csv("C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/games_data.csv")

le = LabelEncoder()
data['opening_eco_encoded'] = le.fit_transform(data['opening_eco'])

# Define features and target
features = ['white_rating', 'black_rating', 'rating_difference', 'base_time', 'increment', 'opening_eco_encoded']
target = 'winner'
data[target] = data[target].map({'white': 1, 'black': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=6462)

# Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200, max_features=int(len(features) / 2), random_state=6462)
rf_model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy of Random Forest Classifier (n_estimators=200): {accuracy:.4f}")

# Calculate accuracy for different numbers of estimators
n_estimators_range = range(10, 210, 10)
test_accuracies = []

for n in n_estimators_range:
    temp_model = RandomForestClassifier(n_estimators=n, max_features=int(len(features) / 2), random_state=6462)
    temp_model.fit(X_train, y_train)
    y_pred_temp = temp_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_temp)
    test_accuracies.append(acc)

# Plot Test Accuracy vs Number of Trees
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Number of Trees (Estimators)')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Number of Trees in Random Forest')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar plot for feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [features[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances from Random Forest Classifier")
plt.bar(range(len(features)), importances[indices], align="center")
plt.xticks(range(len(features)), feature_names, rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
