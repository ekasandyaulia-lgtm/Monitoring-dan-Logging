from flask import Flask, request, jsonify
import pandas as pd
import mlflow
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

mlflow.set_tracking_uri(
    "https://dagshub.com/ekasandyaulia-lgtm/SMLS_Eka_Sandy_Aulia_Puspitasari.mlflow"
)
model_uri = "models:/model/1"
model = mlflow.sklearn.load_model(model_uri)

app = Flask(__name__)

# Kolom numeric yang digunakan model
MODEL_COLUMNS = ["PassengerId","Pclass","Age","SibSp","Parch","Fare",
                 "Age_scaled","Fare_scaled","SibSp_scaled","Parch_scaled",
                 "Pclass_scaled","Sex_encoded"]

# Prometheus metrics
REQUEST_COUNT = Counter("request_count_total", "Total requests")
REQUEST_ERRORS = Counter("request_errors_total", "Total failed requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency in seconds")

@app.route("/predict", methods=["POST"])
def predict():
    import time
    start = time.time()
    REQUEST_COUNT.inc()
    try:
        data = request.json
        df = pd.DataFrame([data])
        df = df[MODEL_COLUMNS].astype(float)
        prediction = model.predict(df)
        return jsonify({"prediction": int(prediction[0])})
    except Exception:
        REQUEST_ERRORS.inc()
        return jsonify({"error": "Something went wrong"}), 500
    finally:
        REQUEST_LATENCY.observe(time.time() - start)

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)