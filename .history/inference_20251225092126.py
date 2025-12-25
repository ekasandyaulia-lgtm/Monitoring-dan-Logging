import mlflow
import pandas as pd
from flask import Flask, request, jsonify

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

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # Pastikan hanya kolom yang ada di MODEL_COLUMNS
    df = df[MODEL_COLUMNS]

    # Pastikan tipe data numeric
    df = df.astype(float)

    prediction = model.predict(df)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)