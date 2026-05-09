from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import numpy as np
import pandas as pd
from typing import Optional

print("initiating response...")

app = FastAPI(
    title="Diabetes Readmission Prediction API",
    description="Predicts early hospital readmission (within 30 days) for diabetic patients",
    version="1.0.0"
)

# load models at startup
model = joblib.load("model_artifacts/xgb_readmission_model.pkl")
with open("model_artifacts/feature_names.json") as f:
    feature_names = json.load(f)


class PatientRecord(BaseModel):
    # demographics
    race: int
    gender: int
    age_numeric: float
    weight_numeric: Optional[float] = 87.0
    # admission info
    admission_type: int
    discharge_disposition: int
    admission_source: int
    time_in_hospital: int
    payer_code: int
    medical_specialty: int
    # procedures and labs
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    # lab results
    max_glu_serum: float = -99
    A1Cresult: float = -99
    # drugs (0=No, 1=Steady, 2=Down, 3=Up)
    metformin: int = 0
    repaglinide: int = 0
    nateglinide: int = 0
    chlorpropamide: int = 0
    glimepiride: int = 0
    acetohexamide: int = 0
    glipizide: int = 0
    glyburide: int = 0
    tolbutamide: int = 0
    pioglitazone: int = 0
    rosiglitazone: int = 0
    acarbose: int = 0
    miglitol: int = 0
    troglitazone: int = 0
    tolazamide: int = 0
    examide: int = 0
    citoglipton: int = 0
    insulin: int = 0
    glyburide_metformin: int = 0
    glipizide_metformin: int = 0
    glimepiride_pioglitazone: int = 0
    metformin_rosiglitazone: int = 0
    metformin_pioglitazone: int = 0
    # other
    change: int = 0
    diabetesMed: int = 1
    active_drug_count: int = 1


@app.get("/")
def home():
    return {"message": "Diabetes Readmission API is running"}


@app.post("/predict")
def predict_readmission(patient: PatientRecord):
    try:
        # convert input to dict and rename combo drugs to match training feature names
        data = patient.dict()
        rename_map = {
            "glyburide_metformin": "glyburide-metformin",
            "glipizide_metformin": "glipizide-metformin",
            "glimepiride_pioglitazone": "glimepiride-pioglitazone",
            "metformin_rosiglitazone": "metformin-rosiglitazone",
            "metformin_pioglitazone": "metformin-pioglitazone",
        }
        for old, new in rename_map.items():
            if old in data:
                data[new] = data.pop(old)

        input_df = pd.DataFrame([data])[feature_names]
        prob = model.predict_proba(input_df)[0][1]
        prediction = int(prob >= 0.5)

        risk_level = "High" if prob >= 0.5 else "Moderate" if prob >= 0.3 else "Low"

        return {
            "early_readmission_predicted": bool(prediction),
            "readmission_probability": round(float(prob), 4),
            "risk_level": risk_level,
            "message": "High risk patient - consider follow-up intervention" if prediction else "Lower risk patient"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "XGBoost v1.0"}


print("Executed successfully")