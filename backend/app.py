"""
app.py — FastAPI REST API for Loan Prediction System
-------------------------------------------------------
Endpoints:
  POST /api/predict        — two-stage prediction
  GET  /api/health         — health check
  GET  /api/model-stats    — model evaluation metrics
"""

import os
import sys
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict

app = FastAPI(title="Loan Prediction API")

# allow React frontend on localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ────────────────────────────────────────────────────────────────

class LoanApplication(BaseModel):
    grade: str = "C"
    annual_inc: float = 0
    short_emp: int = 0
    emp_length_num: int = 5
    home_ownership: str = "RENT"
    dti: float = 15
    purpose: str = "other"
    term: str = " 36 months"
    last_delinq_none: int = 1
    revol_util: float = 40
    total_rec_late_fee: float = 0
    od_ratio: float = 0.5

    loan_amount: Optional[float] = None
    term_months: Optional[int] = 36


class ModelMetric(BaseModel):
    name: str
    f1: float
    auc: float
    accuracy: float


class ModelStatsResponse(BaseModel):
    models: list[ModelMetric]
    best_model: str
    dataset_size: int
    features: int
    class_balance: dict


# ─── Health ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Loan Prediction API is running"}


# ─── Main prediction endpoint ────────────────────────────────────────────────

@app.post("/api/predict")
def predict_loan(application: LoanApplication):
    try:
        applicant = {
            "grade":              application.grade,
            "annual_inc":         application.annual_inc,
            "short_emp":          application.short_emp,
            "emp_length_num":     application.emp_length_num,
            "home_ownership":     application.home_ownership,
            "dti":                application.dti,
            "purpose":            application.purpose,
            "term":               application.term,
            "last_delinq_none":   application.last_delinq_none,
            "revol_util":         application.revol_util,
            "total_rec_late_fee": application.total_rec_late_fee,
            "od_ratio":           application.od_ratio,
        }

        result = predict(
            applicant,
            loan_amount=application.loan_amount,
            term_months=application.term_months or 36,
        )
        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Model stats ──────────────────────────────────────────────────────────────

@app.get("/api/model-stats", response_model=ModelStatsResponse)
def model_stats():
    return {
        "models": [
            {"name": "Logistic Regression", "f1": 0.4308, "auc": 0.7134, "accuracy": 0.6577},
            {"name": "Decision Tree",       "f1": 0.3547, "auc": 0.6730, "accuracy": 0.7180},
            {"name": "Random Forest",       "f1": 0.3385, "auc": 0.6894, "accuracy": 0.7635},
        ],
        "best_model": "Random Forest",
        "dataset_size": 20000,
        "features": 14,
        "class_balance": {"non_default": "80%", "default": "20%"},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)