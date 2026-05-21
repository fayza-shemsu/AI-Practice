# app.py
import joblib
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn. Built by Fayza — Week 6.",
    version="1.0.0",
)

# Load model once at startup
model = None
for root, dirs, files in os.walk("./rai_model_v2/"):
    for f in files:
        if f.endswith(".pkl"):
            model = joblib.load(os.path.join(root, f))
            print("Model loaded:", type(model).__name__)
            break

# Input schema — what the API accepts
class CustomerInput(BaseModel):
    age:               float = Field(..., ge=18,  le=100)
    tenure:            float = Field(..., ge=0,   le=200)
    usage_frequency:   float = Field(..., ge=0,   le=100)
    support_calls:     float = Field(..., ge=0,   le=50)
    total_spend:       float = Field(..., ge=0)
    last_interaction:  float = Field(..., ge=0,   le=60)
    gender:            str
    subscription_type: str
    contract_length:   str

    @validator("gender")
    def gender_valid(cls, v):
        if v not in ["Male", "Female"]:
            raise ValueError("must be Male or Female")
        return v

    @validator("subscription_type")
    def sub_valid(cls, v):
        if v not in ["Basic", "Standard", "Premium"]:
            raise ValueError("must be Basic, Standard, or Premium")
        return v

    @validator("contract_length")
    def contract_valid(cls, v):
        if v not in ["Monthly", "Quarterly", "Annual"]:
            raise ValueError("must be Monthly, Quarterly, or Annual")
        return v

# Output schema — what the API returns
class PredictionOutput(BaseModel):
    churn:             bool
    churn_probability: float
    risk_level:        str
    recommendation:    str

# Route 1 — health check
@app.get("/")
def health():
    return {"status": "healthy", "model": "churn-model-fayza v2"}

# Route 2 — model info
@app.get("/model-info")
def info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "type":     type(model).__name__,
        "features": model.feature_names_in_.tolist(),
    }

# Route 3 — prediction
@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = pd.DataFrame([{
        "Age":               customer.age,
        "Tenure":            customer.tenure,
        "Usage Frequency":   customer.usage_frequency,
        "Support Calls":     customer.support_calls,
        "Total Spend":       customer.total_spend,
        "Last Interaction":  customer.last_interaction,
        "Gender":            customer.gender,
        "Subscription Type": customer.subscription_type,
        "Contract Length":   customer.contract_length,
    }])

    encoded = pd.get_dummies(raw, drop_first=True)
    encoded = encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction  = model.predict(encoded)[0]
    probability = model.predict_proba(encoded)[0][1]

    if probability >= 0.8:
        risk_level = "HIGH"
    elif probability >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if not prediction:
        recommendation = "Low risk. No action needed."
    elif customer.support_calls > 5:
        recommendation = "Resolve support tickets immediately."
    elif customer.contract_length == "Monthly":
        recommendation = "Offer annual contract with discount."
    else:
        recommendation = "Schedule retention call within 48 hours."

    return PredictionOutput(
        churn=bool(prediction),
        churn_probability=round(float(probability), 4),
        risk_level=risk_level,
        recommendation=recommendation,
    )