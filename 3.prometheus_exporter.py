from flask import Flask, request, jsonify
import pandas as pd
import mlflow
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time

# MLflow model
mlflow.set_tracking_uri("https://dagshub.com/ekasandyaulia-lgtm/SMLS_Eka_Sandy_Aulia_Puspitasari.mlflow")
model_uri = "models:/Titanic_RF_Model/6"  # model versi 6
model = mlflow.sklearn.load_model(model_uri)

app = Flask(__name__)

MODEL_COLUMNS = ["PassengerId","Pclass","Age","SibSp","Parch","Fare",
                 "Age_scaled","Fare_scaled","SibSp_scaled","Parch_scaled",
                 "Pclass_scaled","Sex_encoded"]

# Prometheus metrics (10 metriks)
REQUEST_COUNT = Counter("request_count_total", "Total requests")
REQUEST_ERRORS = Counter("request_errors_total", "Total failed requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency in seconds")
PREDICTION_0 = Counter("prediction_0_total", "Predictions for class 0")
PREDICTION_1 = Counter("prediction_1_total", "Predictions for class 1")
LAST_PREDICTION = Gauge("last_prediction", "Value of last prediction")
INPUT_FEATURES_SUM = Gauge("input_features_sum", "Sum of input features")
INPUT_FEATURES_MEAN = Gauge("input_features_mean", "Mean of input features")
INPUT_FEATURES_MAX = Gauge("input_features_max", "Max of input features")
INPUT_FEATURES_MIN = Gauge("input_features_min", "Min of input features")

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    REQUEST_COUNT.inc()
    try:
        data = request.json
        df = pd.DataFrame([data])
        df = df[MODEL_COLUMNS].astype(float)
        pred = model.predict(df)
        REQUEST_LATENCY.observe(time.time() - start)

        # update additional metrics
        LAST_PREDICTION.set(pred[0])
        INPUT_FEATURES_SUM.set(df.sum().sum())
        INPUT_FEATURES_MEAN.set(df.mean().mean())
        INPUT_FEATURES_MAX.set(df.max().max())
        INPUT_FEATURES_MIN.set(df.min().min())
        if pred[0] == 0:
            PREDICTION_0.inc()
        else:
            PREDICTION_1.inc()

        return jsonify({"prediction": int(pred[0])})
    except Exception:
        REQUEST_ERRORS.inc()
        return jsonify({"error": "Something went wrong"}), 500

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)