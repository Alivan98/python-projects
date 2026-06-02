import pandas as pd
import os

# Φάκελος με τα αρχεία CSV
data_folder = "C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis"  # Βάλε τη σωστή διαδρομή
output_folder = "C:/Users/Aliva/OneDrive/Υπολογιστής/Data-Analysis_NewDATA/Ordered DATA/200to400"  # Φάκελος για τα νέα αρχεία

# Αν δεν υπάρχει ο φάκελος εξόδου, τον δημιουργούμε
os.makedirs(output_folder, exist_ok=True)

# Λίστα με τα αρχεία που θέλουμε να επεξεργαστούμε
csv_files = [
    "TotalAcceleration.csv",
    "Accelerometer.csv",
    "Gyroscope.csv",
    "Magnetometer.csv",
    "Light.csv"
]

# Επιλογή των 100 πρώτων δειγμάτων από κάθε αρχείο
for file in csv_files:
    input_path = os.path.join(data_folder, file)
    output_path = os.path.join(output_folder, file)
    
    df = pd.read_csv(input_path)
    df_subset = df.iloc[200:400]  # Επιλέγουμε τις πρώτες 100 γραμμές
    
    df_subset.to_csv(output_path, index=False)
    print(f"Αποθηκεύτηκε: {output_path}")

print("✅ Επιλογή των 200 δειγμάτων (201-400) ολοκληρώθηκε!")