"""
eligibility.py
--------------

Deterministic loan eligibility rules.

This module checks whether an applicant satisfies the
basic business rules required to proceed to the
default-risk ML model.

It does NOT perform machine-learning prediction.

Flow:

    Applicant
        ↓
    Eligibility Rules
        ↓
    Eligible?
      /    \
    No      Yes
    ↓        ↓
  Reject   Default ML Model
"""


import os
import sys


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
)

from config import APPROVAL_RULES


# ---------------------------------------------------------------------------
# Eligibility check
# ---------------------------------------------------------------------------

def check_eligibility(applicant: dict) -> dict:
    """
    Check whether an applicant satisfies all eligibility rules.

    Parameters
    ----------
    applicant : dict
        Applicant information.

    Returns
    -------
    dict
        Eligibility result containing:

        eligible
        failed_rules
    """

    failed_rules = []

    # ---------------------------------------------------------------
    # Rule 1: Annual income
    # ---------------------------------------------------------------

    annual_inc = applicant.get("annual_inc")

    if annual_inc is None:
        failed_rules.append("annual_inc_missing")

    elif annual_inc < APPROVAL_RULES["min_annual_inc"]:
        failed_rules.append("annual_income_too_low")

    # ---------------------------------------------------------------
    # Rule 2: Debt-to-income ratio
    # ---------------------------------------------------------------

    dti = applicant.get("dti")

    if dti is None:
        failed_rules.append("dti_missing")

    elif dti > APPROVAL_RULES["max_dti"]:
        failed_rules.append("dti_too_high")

    # ---------------------------------------------------------------
    # Rule 3: Grade
    # ---------------------------------------------------------------

    grade = applicant.get("grade")

    if grade is None:
        failed_rules.append("grade_missing")

    elif grade < APPROVAL_RULES["min_grade"]:
        failed_rules.append("grade_too_low")

    # ---------------------------------------------------------------
    # Rule 4: Revolving utilization
    # ---------------------------------------------------------------

    revol_util = applicant.get("revol_util")

    if revol_util is None:
        failed_rules.append("revol_util_missing")

    elif revol_util > APPROVAL_RULES["max_revol_util"]:
        failed_rules.append("revol_util_too_high")

    # ---------------------------------------------------------------
    # Final eligibility decision
    # ---------------------------------------------------------------

    eligible = len(failed_rules) == 0

    return {
        "eligible": eligible,
        "failed_rules": failed_rules,
    }


# ---------------------------------------------------------------------------
# Simple manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    eligible_applicant = {
        "annual_inc": 50000,
        "dti": 20,
        "grade": 5,
        "revol_util": 40,
    }

    ineligible_applicant = {
        "annual_inc": 10000,
        "dti": 50,
        "grade": 1,
        "revol_util": 95,
    }

    print("\nEligible applicant:")
    print(check_eligibility(eligible_applicant))

    print("\nIneligible applicant:")
    print(check_eligibility(ineligible_applicant))