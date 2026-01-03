import requests
import time
import random

# URL endpoint Flask + Prometheus
PREDICT_URL = "http://127.0.0.1:8000/predict"

X_test = [
    {"PassengerId":147,"Pclass":3,"Age":27,"SibSp":0,"Parch":0,"Fare":7.7958,
     "Age_scaled":-0.1814,"Fare_scaled":-0.4914,"SibSp_scaled":-0.4745,
     "Parch_scaled":-0.4736,"Pclass_scaled":0.8273,"Sex_encoded":1},
    {"PassengerId":417,"Pclass":2,"Age":34,"SibSp":1,"Parch":1,"Fare":32.5,
     "Age_scaled":0.3564,"Fare_scaled":0.0059,"SibSp_scaled":0.4327,
     "Parch_scaled":0.7676,"Pclass_scaled":-0.3693,"Sex_encoded":0},
    {"PassengerId":49,"Pclass":3,"Age":28,"SibSp":2,"Parch":0,"Fare":21.6792,
     "Age_scaled":-0.1046,"Fare_scaled":-0.2119,"SibSp_scaled":1.3401,
     "Parch_scaled":-0.4736,"Pclass_scaled":0.8273,"Sex_encoded":1},
]

def send_requests(n_times=50, delay=0.1):
    """Send multiple requests to /predict to simulate load."""
    for i in range(n_times):
        data = random.choice(X_test)
        try:
            resp = requests.post(PREDICT_URL, json=data, timeout=5)
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