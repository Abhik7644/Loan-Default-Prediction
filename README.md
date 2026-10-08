# LoanSense — Loan Default Prediction System

An end-to-end Machine Learning application that evaluates loan applicants using deterministic eligibility rules and a Logistic Regression model to estimate default risk.

The system separates **basic loan eligibility** from **machine-learning-based default-risk prediction**.

---

## Project Overview

LoanSense is designed to answer two separate questions:

1. **Does the applicant satisfy the basic eligibility criteria?**
2. **If eligible, how likely is the applicant to default?**

The system first applies deterministic business rules. Only applicants who pass these rules are evaluated by the Machine Learning model.

This prevents the ML model from being used as a substitute for basic eligibility checks.

---

## System Architecture

```text
                    React Frontend
                          │
                          ▼
                    Loan Application
                          │
                          ▼
                  FastAPI REST API
                          │
                          ▼
                  Pydantic Validation
                          │
                          ▼
                 Eligibility Rules
                    /           \
                   /             \
              NOT ELIGIBLE      ELIGIBLE
                  │                │
                  ▼                ▼
              REJECTED      Logistic Regression
                                  │
                                  ▼
                         Default Probability
                                  │
                                  ▼
                          Threshold = 0.55
                             /          \
                            /            \
                       >= 0.55          < 0.55
                          │                │
                          ▼                ▼
                      REJECTED          APPROVED