import pandas as pd
from flask import Flask, request, jsonify
import mlflow
import os

mlflow.set_tracking_uri("https://dagshub.com/ekasandyaulia-lgtm/SMLS_Eka_Sandy_Aulia_Puspitasari.mlflow")
model_uri = "models:/model/1"
model = mlflow.sklearn.load_model(model_uri)

base_path = os.path.dirname(__file__)
X_train = pd.read_csv(os.path.join(base_path, "processed_data", "X_train.csv"))

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    # Reindex input ke kolom X_train
    df = df.reindex(columns=X_train.columns, fill_value=0)

    prediction = model.predict(df)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)