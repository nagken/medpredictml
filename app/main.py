from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# Initialize FastAPI
app = FastAPI()

# Define request model for input validation
class PatientData(BaseModel):
    age: int
    admission_type: str
    diagnosis: str
    medications: str
    length_of_stay: int
    previous_admissions: int

# Load the trained model
MODEL_PATH = "models/xgboost_patient_readmission.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError("Trained model file not found. Train the model first!")

# Define feature processing function
def preprocess_input(data: PatientData):
    """
    Convert patient input data into a format that matches the ML model.
    """
    feature_dict = {
        "age": data.age,
        "length_of_stay": data.length_of_stay,
        "previous_admissions": data.previous_admissions,
        "admission_type_emergency": 1 if data.admission_type.lower() == "emergency" else 0,
        "admission_type_routine": 1 if data.admission_type.lower() == "routine" else 0,
        "admission_type_urgent": 1 if data.admission_type.lower() == "urgent" else 0,
        "diagnosis_diabetes": 1 if data.diagnosis.lower() == "diabetes" else 0,
        "diagnosis_hypertension": 1 if data.diagnosis.lower() == "hypertension" else 0,
        "diagnosis_asthma": 1 if data.diagnosis.lower() == "asthma" else 0,
        "medications_insulin": 1 if data.medications.lower() == "insulin" else 0,
        "medications_metformin": 1 if data.medications.lower() == "metformin" else 0,
        "medications_albuterol": 1 if data.medications.lower() == "albuterol" else 0,
    }
    return pd.DataFrame([feature_dict])

# API Home Route
@app.get("/")
def home():
    return {"message": "🚀 MedPredictML API is running!"}

# API Endpoint for Prediction
@app.post("/predict/")
def predict_readmission(data: PatientData):
    try:
        input_data = preprocess_input(data)
        prediction = model.predict(input_data)[0]
        return {"readmitted": bool(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
