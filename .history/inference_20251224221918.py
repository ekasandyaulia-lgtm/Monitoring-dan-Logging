import requests
import pandas as pd
import json
import os

# Load test data yang sudah preprocessing
base_path = os.path.dirname(__file__)
X_test = pd.read_csv(os.path.join(base_path, "processed_data", "X_test.csv"))

# MLflow serve endpoint
url = "http://localhost:5000/invocations"

# Format data sesuai JSON yang diterima MLflow
data = {
    "columns": list(X_test.columns),
    "data": X_test.values.tolist()
}

# Kirim request POST ke MLflow model serve
response = requests.post(url, json=data)

if response.status_code == 200:
    print("Predictions:")
    preds = response.json()
    print(preds)
else:
    print(f"Error {response.status_code}: {response.text}")