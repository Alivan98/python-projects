import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
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

# Split into training and testing set
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=6462)

# Range of estimators
n_estimators_range = range(10, 210, 10)
errors_bagging = []
errors_rf = []
errors_boosting = []

for n in n_estimators_range:
    # Bagging
    bag_model = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=6462), n_estimators=n, random_state=6462)
    bag_model.fit(X_train, y_train)
    acc_bag = accuracy_score(y_test, bag_model.predict(X_test))
    errors_bagging.append(1 - acc_bag)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=n, max_features=int(len(features)/2), random_state=6462)
    rf_model.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf_model.predict(X_test))
    errors_rf.append(1 - acc_rf)
    
    # Boosting
    boost_model = GradientBoostingClassifier(n_estimators=n, learning_rate=0.1, max_depth=1, random_state=6462)
    boost_model.fit(X_train, y_train)
    acc_boost = accuracy_score(y_test, boost_model.predict(X_test))
    errors_boosting.append(1 - acc_boost)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, errors_bagging, label='Bagging')
plt.plot(n_estimators_range, errors_rf, label='Random Forest')
plt.plot(n_estimators_range, errors_boosting, label='Boosting')
plt.xlabel('Number of Estimators')
plt.ylabel('Test Error (1 - Accuracy)')
plt.title('Test Error vs Number of Estimators')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
