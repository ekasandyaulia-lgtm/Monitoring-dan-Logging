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
start_http_server(8001)   # ⬅️ metrics di sini

# =====================
# Prometheus metrics
# =====================
REQUEST_COUNT = Counter("request_count", "Total prediction requests")
PREDICTION_COUNT = Counter("prediction_count", "Total predictions")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Prediction latency")
CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage")
MEMORY_USAGE = Gauge("memory_usage_percent", "Memory usage")

# =====================
# Load model (RUN ID PUNYAMU)
# =====================
model = mlflow.pyfunc.load_model(
    "runs:/fa62fdf03c9540348861701a11eac172/model"
)

# =====================
# FastAPI app
# =====================
app = FastAPI(title="ML Inference API")

# ⬇️ INPUT HARUS SESUAI TRAINING
class InputData(BaseModel):
    Pclass: int
    Sex_encoded: int
    Age_scaled: float
    Fare_scaled: float
    SibSp_scaled: float
    Parch_scaled: float

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    start_time = time.time()

    REQUEST_COUNT.inc()
    PREDICTION_COUNT.inc()

    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)

    latency = time.time() - start_time
    REQUEST_LATENCY.observe(latency)

    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)

    return {
        "prediction": prediction.tolist(),
        "latency": latency
    }