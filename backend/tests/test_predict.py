"""
tests/test_predict.py
---------------------

Tests for the current Loan Default Prediction system.

Run:

    python -m pytest tests/ -v
"""

import os
import sys

from fastapi.testclient import TestClient

# Allow imports from the backend directory
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from app import app
from src.eligibility import check_eligibility
from src.predict import predict_loan


client = TestClient(app)


# ============================================================
# TEST DATA
# ============================================================

ELIGIBLE_APPLICANT = {
    "grade": "C",
    "annual_inc": 50000,
    "short_emp": 0,
    "emp_length_num": 5,
    "home_ownership": "RENT",
    "dti": 15,
    "purpose": "credit_card",
    "term": "36 months",
    "last_delinq_none": 1,
    "revol_util": 40,
    "od_ratio": 0.5,
}


INELIGIBLE_APPLICANT = {
    "grade": "G",
    "annual_inc": 10000,
    "short_emp": 0,
    "emp_length_num": 2,
    "home_ownership": "RENT",
    "dti": 50,
    "purpose": "other",
    "term": "36 months",
    "last_delinq_none": 1,
    "revol_util": 95,
    "od_ratio": 0.5,
}


# ============================================================
# ELIGIBILITY TESTS
# ============================================================

def test_eligible_applicant_passes_rules():
    applicant = {
        "annual_inc": 50000,
        "dti": 15,
        "grade": 5,
        "revol_util": 40,
    }

    result = check_eligibility(applicant)

    assert result["eligible"] is True
    assert result["failed_rules"] == []


def test_ineligible_applicant_fails_rules():
    applicant = {
        "annual_inc": 10000,
        "dti": 50,
        "grade": 1,
        "revol_util": 95,
    }

    result = check_eligibility(applicant)

    assert result["eligible"] is False

    assert "annual_income_too_low" in result["failed_rules"]
    assert "dti_too_high" in result["failed_rules"]
    assert "grade_too_low" in result["failed_rules"]
    assert "revol_util_too_high" in result["failed_rules"]


# ============================================================
# PREDICTION LOGIC TESTS
# ============================================================

def test_ineligible_applicant_is_rejected():
    result = predict_loan(
        {
            "grade": 1,
            "annual_inc": 10000,
            "short_emp": 0,
            "emp_length_num": 2,
            "home_ownership": "RENT",
            "dti": 50,
            "purpose": "other",
            "term": "36 months",
            "last_delinq_none": 1,
            "revol_util": 95,
            "od_ratio": 0.5,
        }
    )

    assert result["eligible"] is False
    assert result["decision"] == "REJECTED"
    assert "default_probability" not in result
    assert "threshold" not in result


def test_eligible_applicant_gets_prediction():
    result = predict_loan(
        {
            "grade": 5,
            "annual_inc": 50000,
            "short_emp": 0,
            "emp_length_num": 5,
            "home_ownership": "RENT",
            "dti": 15,
            "purpose": "credit_card",
            "term": "36 months",
            "last_delinq_none": 1,
            "revol_util": 40,
            "od_ratio": 0.5,
        }
    )

    assert result["eligible"] is True
    assert result["decision"] in ["APPROVED", "REJECTED"]

    assert result["default_probability"] is not None
    assert 0 <= result["default_probability"] <= 1

    assert result["threshold"] == 0.55


# ============================================================
# API TESTS
# ============================================================

def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_predict_endpoint_with_eligible_applicant():
    response = client.post(
        "/api/predict",
        json=ELIGIBLE_APPLICANT,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["eligible"] is True
    assert data["decision"] in ["APPROVED", "REJECTED"]
    assert data["default_probability"] is not None
    assert 0 <= data["default_probability"] <= 1
    assert data["threshold"] == 0.55


def test_predict_endpoint_with_ineligible_applicant():
    response = client.post(
        "/api/predict",
        json=INELIGIBLE_APPLICANT,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["eligible"] is False
    assert data["decision"] == "REJECTED"

    assert "annual_income_too_low" in data["failed_rules"]
    assert "dti_too_high" in data["failed_rules"]
    assert "revol_util_too_high" in data["failed_rules"]

    assert data["default_probability"] is None
    assert data["threshold"] is None


def test_predict_endpoint_rejects_invalid_grade():
    applicant = ELIGIBLE_APPLICANT.copy()
    applicant["grade"] = "Z"

    response = client.post(
        "/api/predict",
        json=applicant,
    )

    assert response.status_code == 422


def test_predict_endpoint_rejects_negative_income():
    applicant = ELIGIBLE_APPLICANT.copy()
    applicant["annual_inc"] = -5000

    response = client.post(
        "/api/predict",
        json=applicant,
    )

    assert response.status_code == 422


def test_model_stats_endpoint():
    response = client.get("/api/model-stats")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "Logistic Regression"
    assert data["threshold"] == 0.55
    assert data["selection_metric"] == "F1"
    assert data["dataset_size"] == 20000
    assert data["features"] == 11