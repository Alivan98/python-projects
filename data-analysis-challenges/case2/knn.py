import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/games_data.csv")

# Feature selection
features = df[["white_rating", "black_rating"]].copy()

# Encode 'opening_eco' feature
le_eco = LabelEncoder()
features["opening_eco"] = le_eco.fit_transform(df["opening_eco"])

# Target variable (winner): convert to numerical format
le_winner = LabelEncoder()
y = le_winner.fit_transform(df["winner"])

# Feature scaling / normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Split dataset into training and test sets using random_state = 6462
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=6462)

# Select optimal K using 5-fold Cross-Validation on the training set
k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_range[np.argmax(cv_scores)]
print(f"Optimal K: {best_k}")
print(f"Maximum accuracy on training set via cross-validation: {max(cv_scores):.4f}")

# Train final KNN model with optimal K
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)

# Predict on test set
y_pred = knn_final.predict(X_test)

# Evaluate accuracy on test set
test_accuracy = knn_final.score(X_test, y_test)
print(f"Accuracy on test set: {test_accuracy:.4f}")

# Confusion Matrix configuration and plotting
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_winner.classes_)
disp.plot(cmap=plt.cm.Blues)

plt.title('Confusion Matrix (KNN - Test Set)')
plt.grid(False)
plt.tight_layout()
plt.show()
