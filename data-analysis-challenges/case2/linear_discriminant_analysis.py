import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Load dataset
df = pd.read_csv("C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/games_data.csv")

# Split increment_code into initial_time and increment
increment_split = df["increment_code"].str.split("+", expand=True)
df["initial_time"] = increment_split[0].astype(int) * 60  # minutes to seconds
df["increment"] = increment_split[1].astype(int)          # in seconds

# Feature selection
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

# Split dataset into training and test sets using random_state = 6462
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=6462)

# Initialize and train LDA model (without normalization/scaling)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Calculate and print accuracy
accuracy = lda.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Predict outcomes on test set
y_pred = lda.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_winner.classes_))

# Confusion Matrix configuration and plotting
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_winner.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (LDA - Test Set)')
plt.grid(False)
plt.tight_layout()
plt.show()
