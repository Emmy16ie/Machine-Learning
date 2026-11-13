from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import numpy as np

app = FastAPI(
    
    title="California House Prediction Model",
    description="This model is used to predict the prices of Hosue in California USA.",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent

#Load the trained Model
model_path = BASE_DIR / "models" / "house_price_model.pkl"
model = joblib.load(model_path)

class HouseData(BaseModel):
    MedInc : float
    HouseAge : float
    AveRooms : float
    AveBedrms : float
    Population : float
    AveOccup : float
    Latitude : float
    Longitude : float

@app.get("/")
def home():
    return {
        "message" : "California House Prediction Model is running"
    }

@app.post("/predict")
def predicts(data: HouseData):

    features = np.array([[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]])

    prediction = model.predict(features)[0]

    return {
        "predicted_house_value": float(prediction)
    }