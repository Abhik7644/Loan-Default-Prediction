"""
predict.py
----------

Loan prediction logic.

Decision flow:

    Applicant
        ↓
    Eligibility Check
        ↓
    Not Eligible ─────→ REJECTED
        │
        │ Eligible
        ▼
    Default ML Model
        ↓
    Default Probability
        ↓
    Threshold = 0.55
        ↓
    Risk Decision
        ↓
    APPROVED / REJECTED
"""


import os
import sys

import joblib
import pandas as pd


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
)

from config import (
    DEFAULT_MODEL_PATH,
    RISK_THRESHOLDS,
)

from src.eligibility import check_eligibility


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """
    Load the trained default prediction pipeline.
    """

    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {DEFAULT_MODEL_PATH}"
        )

    model = joblib.load(DEFAULT_MODEL_PATH)

    return model


# ---------------------------------------------------------------------------
# Default prediction
# ---------------------------------------------------------------------------

def predict_default(model, applicant: dict):
    """
    Predict the probability of loan default.

    Parameters
    ----------
    model : trained sklearn pipeline
        Saved Logistic Regression pipeline.

    applicant : dict
        Applicant features.

    Returns
    -------
    float
        Probability of default.
    """

    input_df = pd.DataFrame([applicant])

    probability = model.predict_proba(
        input_df
    )[0][1]

    return float(probability)


# ---------------------------------------------------------------------------
# Final loan decision
# ---------------------------------------------------------------------------

def predict_loan(applicant: dict):
    """
    Complete loan decision workflow.

    Step 1:
        Check eligibility.

    Step 2:
        If not eligible, reject immediately.

    Step 3:
        If eligible, predict default probability.

    Step 4:
        Apply the selected risk threshold.

    Returns
    -------
    dict
        Final loan decision.
    """

    # ---------------------------------------------------------------
    # Step 1: Eligibility check
    # ---------------------------------------------------------------

    eligibility_result = check_eligibility(
        applicant
    )

    if not eligibility_result["eligible"]:

        return {
            "eligible": False,
            "decision": "REJECTED",
            "reason": "Eligibility criteria not satisfied.",
            "failed_rules": eligibility_result[
                "failed_rules"
            ],
        }

    # ---------------------------------------------------------------
    # Step 2: Load trained model
    # ---------------------------------------------------------------

    model = load_model()

    # ---------------------------------------------------------------
    # Step 3: Predict default probability
    # ---------------------------------------------------------------

    default_probability = predict_default(
        model,
        applicant,
    )

    # ---------------------------------------------------------------
    # Step 4: Apply threshold
    # ---------------------------------------------------------------

    threshold = RISK_THRESHOLDS["medium"]

    if default_probability >= threshold:

        decision = "REJECTED"
        risk_level = "HIGH"

    else:

        decision = "APPROVED"
        risk_level = "LOW"

    # ---------------------------------------------------------------
    # Step 5: Return final result
    # ---------------------------------------------------------------

    return {
        "eligible": True,
        "decision": decision,
        "risk_level": risk_level,
        "default_probability": round(
            default_probability,
            4,
        ),
        "threshold": threshold,
        "failed_rules": [],
    }


# ---------------------------------------------------------------------------
# Manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("LOAN PREDICTION TEST")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Test 1: Ineligible applicant
    # ---------------------------------------------------------------

    ineligible_applicant = {
        "grade": 1,
        "annual_inc": 10000,
        "short_emp": 0,
        "emp_length_num": 5,
        "home_ownership": "RENT",
        "dti": 50,
        "purpose": "credit_card",
        "term": "36 months",
        "last_delinq_none": 1,
        "revol_util": 95,
        "od_ratio": 0.5,
    }

    print("\nTest 1: Ineligible applicant")

    result = predict_loan(
        ineligible_applicant
    )

    print(result)

    # ---------------------------------------------------------------
    # Test 2: Eligible applicant
    # ---------------------------------------------------------------

    eligible_applicant = {
        "grade": 5,
        "annual_inc": 50000,
        "short_emp": 0,
        "emp_length_num": 5,
        "home_ownership": "RENT",
        "dti": 20,
        "purpose": "credit_card",
        "term": "36 months",
        "last_delinq_none": 1,
        "revol_util": 40,
        "od_ratio": 0.5,
    }

    print("\nTest 2: Eligible applicant")

    result = predict_loan(
        eligible_applicant
    )

    print(result)