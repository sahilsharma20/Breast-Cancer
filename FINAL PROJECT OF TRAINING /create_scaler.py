import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load your dataset (replace with the actual path of your dataset)
data = pd.read_csv('Breast_Cancer_data.csv')  # Update this path as necessary

# Extract features (excluding the target column, e.g., 'diagnosis')
X = data.drop(columns=['diagnosis'])  # Adjust this based on your dataset

# Fit the scaler on the features
scaler = StandardScaler()
scaler.fit(X)

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')

print("Scaler saved successfully.")
