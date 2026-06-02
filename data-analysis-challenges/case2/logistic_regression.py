import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/games_data.csv")

# Feature engineering: create new features from increment_code
# Split 'X+Y' into two columns: initial_time and increment
increment_split = df["increment_code"].str.split("+", expand=True)
df["initial_time"] = increment_split[0].astype(int) * 60  # minutes to seconds
df["increment"] = increment_split[1].astype(int)          # in seconds

# Prepare features and target variable
features = df[["white_rating", "black_rating"]].copy()

# Encode 'opening_eco' feature into categorical numerical format
le_eco = LabelEncoder()
features["opening_eco"] = le_eco.fit_transform(df["opening_eco"])

# Add initial_time and increment to the features dataframe
features["initial_time"] = df["initial_time"]
features["increment"] = df["increment"]

# Target variable (winner): convert to numerical format
le_winner = LabelEncoder()
y = le_winner.fit_transform(df["winner"])

# Feature scaling / normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Split dataset into training and test sets using random_state = 6462
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=6462)

# Model initialization and training
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.intercept_
logreg.coef_
logreg.score(X_test, y_test)

# Print evaluation metrics
print("Intercept:", logreg.intercept_)
print("Coefficients:", logreg.coef_)
print("Accuracy:", logreg.score(X_test, y_test))

# Predict on test set
y_pred = logreg.predict(X_test)

# Confusion Matrix configuration and plotting
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_winner.classes_)
disp.plot(cmap=plt.cm.Blues)

plt.title('Confusion Matrix (Logistic Regression - Test Set)')
plt.grid(False)
plt.tight_layout()
plt.show()
