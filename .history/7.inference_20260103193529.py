import requests
import pandas as pd
import time
import random
import os

PREDICT_URL = "http://127.0.0.1:8000/predict"

# Load X_test
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

MODEL_COLUMNS = ["PassengerId","Pclass","Age","SibSp","Parch","Fare",
                 "Age_scaled","Fare_scaled","SibSp_scaled","Parch_scaled",
                 "Pclass_scaled","Sex_encoded"]
X_test = X_test[MODEL_COLUMNS].astype(float)

def send_requests(n_times=50, delay=0.1):
    for i in range(n_times):
        row = X_test.sample(n=1).iloc[0].to_dict()  # ambil 1 row random
        try:
            resp = requests.post(PREDICT_URL, json=row, timeout=5)
            if resp.status_code == 200:
                print(f"[{i+1}] Prediction:", resp.json()["prediction"])
            else:
                print(f"[{i+1}] Error:", resp.json())
        except Exception as e:
            print(f"[{i+1}] Request failed:", str(e))
        time.sleep(delay)

if __name__ == "__main__":
    send_requests()
    print("Inference load test selesai!")