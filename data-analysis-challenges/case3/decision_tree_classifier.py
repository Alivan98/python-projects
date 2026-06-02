import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the processed dataset
data = pd.read_csv("C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/processed_games_data.csv")

# Define features and target variable
features = ['white_rating', 'black_rating', 'rating_difference', 'base_time', 'increment']
target = 'winner'

# Convert the categorical target variable to numerical format
data[target] = data[target].map({'white': 1, 'black': 0})

# Split the dataset into training and testing sets (80% training, 20% testing)
# Note: The comment mentions 80-20 split but test_size=0.3 corresponds to 70-30.
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=6462)

# Initialize and train the Decision Tree Classifier with maximum depth of 3
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate outcomes on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the final accuracy metric
print(f"Test Accuracy of Decision Tree (max depth=3): {accuracy:.4f}")

# Visualize the Decision Tree structure
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=features, class_names=['black', 'white'], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

