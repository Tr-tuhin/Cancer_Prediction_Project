# models/model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
data_path = os.path.join("data", "cancer_data.csv")
df = pd.read_csv(data_path)

# Features and label
X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
model_path = os.path.join("models", "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as models/model.pkl")
