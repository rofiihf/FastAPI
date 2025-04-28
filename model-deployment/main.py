from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="Power Consumption Prediction API")

# Load model dan scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Skema input untuk konsumsi daya listrik
class PowerConsumptionInput(BaseModel):
    Temperature: float
    Humidity: float
    WindSpeed: float
    GeneralDiffuseFlows: float
    DiffuseFlows: float

# Fungsi preprocessing data input
def preprocess_input(data: PowerConsumptionInput):
    # Buat DataFrame dari input
    df = pd.DataFrame([{
        "Temperature": data.Temperature,
        "Humidity": data.Humidity,
        "WindSpeed": data.WindSpeed,
        "GeneralDiffuseFlows": data.GeneralDiffuseFlows,
        "DiffuseFlows": data.DiffuseFlows
    }])

    # Lakukan normalisasi pada data input
    df_scaled = scaler.transform(df)
    return df_scaled

@app.get("/")
def read_root():
    return {"message": "Power Consumption Prediction API is running"}

# Endpoint prediksi konsumsi daya listrik
@app.post("/predict")
def predict_power_consumption(data: PowerConsumptionInput):
    processed_data = preprocess_input(data)
    prediction = model.predict(processed_data)[0]  # Mengambil hasil prediksi

    return {
        "prediction": prediction,
        "result": f"Predicted Total Power Consumption (Zone1 + Zone2 + Zone3): {prediction:.2f} kWh",
        "note": "Hasil prediksi merupakan akumulasi dari konsumsi daya listrik tiga zona."
    }
