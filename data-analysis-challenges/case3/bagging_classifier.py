import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
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

# Train the Bagging Classifier
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=6462),
    n_estimators=200,
    random_state=6462
)
bagging_model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy of Bagging Classifier (n_estimators=200): {accuracy:.4f}")

# Calculate accuracy for different numbers of estimators
n_estimators_range = range(10, 210, 10)
test_accuracies = []

for n in n_estimators_range:
    temp_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=6462),
        n_estimators=n,
        random_state=6462
    )
    temp_model.fit(X_train, y_train)
    y_pred_temp = temp_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_temp)
    test_accuracies.append(acc)

# Plot Test Accuracy vs Number of Trees
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Number of Trees (Estimators)')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Number of Trees in Bagging')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate mean feature importance
feature_importances = np.mean([
    tree.feature_importances_ for tree in bagging_model.estimators_
], axis=0)

# Bar plot for feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(features)), feature_importances, tick_label=features)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importances from Bagging Classifier")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
