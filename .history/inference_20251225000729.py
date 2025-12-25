import time
import psutil
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# =====================
# Start Prometheus exporter
# =====================
start_http_server(8001)  # <<< INI KUNCI

# =====================
# Prometheus metrics
# =====================
REQUEST_COUNT = Counter(
    "request_count", "Total number of prediction requests"
)

PREDICTION_COUNT = Counter(
    "prediction_count", "Total number of predictions made"
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Prediction latency"
)

CPU_USAGE = Gauge(
    "cpu_usage_percent", "CPU usage percentage"
)

MEMORY_USAGE = Gauge(
    "memory_usage_percent", "Memory usage percentage"
)

# =====================
# Load model
# =====================
model = mlflow.pyfunc.load_model("models:/model/1")

# =====================
# FastAPI app
# =====================
app = FastAPI(title="ML Inference API")

class InputData(BaseModel):
    features: list

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    start_time = time.time()

    REQUEST_COUNT.inc()
    PREDICTION_COUNT.inc()

    df = pd.DataFrame([data.features])
    prediction = model.predict(df)

    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)

    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)

    return {
        "prediction": prediction.tolist(),
        "latency": latency
    }