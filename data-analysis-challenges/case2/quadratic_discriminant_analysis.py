import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Φόρτωση δεδομένων
df = pd.read_csv("C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/games_data.csv")

# Επεξεργασία increment_code
increment_split = df["increment_code"].str.split("+", expand=True)
df["initial_time"] = increment_split[0].astype(int) * 60
df["increment"] = increment_split[1].astype(int)

# Δημιουργία χαρακτηριστικών
features = df[["white_rating", "black_rating"]].copy()

# Κωδικοποίηση opening_eco
le_eco = LabelEncoder()
features["opening_eco"] = le_eco.fit_transform(df["opening_eco"])

# Προσθήκη initial_time και increment
features["initial_time"] = df["initial_time"]
features["increment"] = df["increment"]

# Στόχος (target): μετατροπή winner σε αριθμητική μορφή
le_winner = LabelEncoder()
y = le_winner.fit_transform(df["winner"])

# Διαχωρισμός σε training και test set με random_state = 6462
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=6462)

# Εκπαίδευση QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Ακρίβεια στο test set
test_accuracy = qda.score(X_test, y_test)
print(f"Ακρίβεια στο test set: {test_accuracy:.4f}")

# Πρόβλεψη στο test set
y_pred = qda.predict(X_test)

# Classification Report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred, target_names=le_winner.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_winner.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (QDA - Test Set)')
plt.grid(False)
plt.tight_layout()
plt.show()
