import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/games_data.csv")

# Encode opening_eco (useful for compatibility with numerical models like Decision Tree)
le = LabelEncoder()
data['opening_eco_encoded'] = le.fit_transform(data['opening_eco'])

# Define features and target
features = ['white_rating', 'black_rating', 'rating_difference', 'base_time', 'increment', 'opening_eco_encoded']
target = 'winner'
data[target] = data[target].map({'white': 1, 'black': 0})

# Split into training and testing set
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=6462)

# Range of depths
depths = range(1, 21)
test_errors = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=6462)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    test_errors.append(1 - acc)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(depths, test_errors, marker='o')
plt.xlabel('Max Depth of Tree')
plt.ylabel('Test Error (1 - Accuracy)')
plt.title('Test Error vs. Tree Depth')
plt.grid(True)
plt.tight_layout()
plt.show()
