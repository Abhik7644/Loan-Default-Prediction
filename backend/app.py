"""
app.py
------

FastAPI REST API for the Loan Default Prediction System.

Endpoints:

    GET  /api/health
        Health check

    POST /api/predict
        Check eligibility first, then predict default risk

    GET  /api/model-stats
        Return current model information
"""


import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__)),
)

from config import (
    DEFAULT_THRESHOLD,
)

from src.predict import predict_loan


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Loan Default Prediction API",
    description=(
        "API for checking loan eligibility and predicting "
        "default risk."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class LoanApplication(BaseModel):
    """
    Input data submitted by the frontend.
    """

    grade: str = Field(
        default="C",
        description="Loan grade from A to G",
    )

    annual_inc: float = Field(
        default=0,
        ge=0,
        description="Annual income",
    )

    short_emp: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Whether the applicant has short employment history",
    )

    emp_length_num: int = Field(
        default=5,
        ge=0,
        description="Employment length in years",
    )

    home_ownership: str = Field(
        default="RENT",
        description="Home ownership status",
    )

    dti: float = Field(
        default=15,
        ge=0,
        description="Debt-to-income ratio",
    )

    purpose: str = Field(
        default="other",
        description="Loan purpose",
    )

    term: str = Field(
        default="36 months",
        description="Loan term",
    )

    last_delinq_none: int = Field(
        default=1,
        ge=0,
        le=1,
        description="Whether applicant has no previous delinquency",
    )

    revol_util: float = Field(
        default=40,
        ge=0,
        description="Revolving credit utilization",
    )

    od_ratio: float = Field(
        default=0.5,
        description="Overdraft ratio",
    )


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """
    Response returned by /api/predict.
    """

    eligible: bool
    decision: str
    reason: str | None = None
    failed_rules: list[str] = []
    risk_level: str | None = None
    default_probability: float | None = None
    threshold: float | None = None


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    """
    Check whether the API is running.
    """

    return {
        "status": "ok",
        "message": "Loan Default Prediction API is running",
    }


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/predict",
    response_model=PredictionResponse,
)
def predict(application: LoanApplication):
    """
    Complete loan prediction workflow.

    Flow:

        Request
            ↓
        Validate input
            ↓
        Eligibility rules
            ↓
        Not eligible → REJECTED
            ↓
        Eligible
            ↓
        Default ML model
            ↓
        Threshold = 0.55
            ↓
        APPROVED / REJECTED
    """

    try:

        # ---------------------------------------------------------------
        # Validate grade
        # ---------------------------------------------------------------

        grade = application.grade.strip().upper()

        grade_map = {
            "A": 7,
            "B": 6,
            "C": 5,
            "D": 4,
            "E": 3,
            "F": 2,
            "G": 1,
        }

        if grade not in grade_map:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Invalid grade. "
                    "Grade must be one of A, B, C, D, E, F, or G."
                ),
            )

        # ---------------------------------------------------------------
        # Normalize term
        # ---------------------------------------------------------------

        term = (
            application.term
            .strip()
            .lower()
        )

        # ---------------------------------------------------------------
        # Build applicant dictionary
        # ---------------------------------------------------------------

        applicant = {
            "grade": grade_map[grade],
            "annual_inc": application.annual_inc,
            "short_emp": application.short_emp,
            "emp_length_num": application.emp_length_num,
            "home_ownership": application.home_ownership,
            "dti": application.dti,
            "purpose": application.purpose,
            "term": term,
            "last_delinq_none": application.last_delinq_none,
            "revol_util": application.revol_util,
            "od_ratio": application.od_ratio,
        }

        # ---------------------------------------------------------------
        # Run complete prediction workflow
        # ---------------------------------------------------------------

        result = predict_loan(
            applicant
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------------

@app.get("/api/model-stats")
def model_stats():
    """
    Return information about the currently deployed model.
    """

    return {
        "model": "Logistic Regression",
        "threshold": DEFAULT_THRESHOLD,
        "selection_metric": "F1",
        "cross_validation_f1": 0.4206,
        "test_f1": 0.4209,
        "test_precision": 0.3135,
        "test_recall": 0.6400,
        "test_roc_auc": 0.6994,
        "dataset_size": 20000,
        "default_rate": 0.20,
        "features": 11,
    }


# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
    )
    