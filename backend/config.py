import os

# ─── Paths ──────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW = os.path.join(
    BASE_DIR, "data", "raw", "dataset.csv"
)

DATA_PROC = os.path.join(
    BASE_DIR, "data", "processed", "processed.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models"
)

DEFAULT_MODEL_PATH = os.path.join(
    MODEL_DIR, "default_pipeline.pkl"
)


# ─── Target ─────────────────────────────────────────────────────────────────

TARGET_COL = "bad_loan"


# ─── Feature definitions ────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "term",
    "home_ownership",
    "purpose",
]

ORDINAL_COLS = [
    "grade",
]

NUMERICAL_COLS = [
    "annual_inc",
    "short_emp",
    "emp_length_num",
    "dti",
    "last_delinq_none",
    "revol_util",
    "od_ratio",
]


# ─── Grade encoding ─────────────────────────────────────────────────────────

GRADE_MAP = {
    "A": 7,
    "B": 6,
    "C": 5,
    "D": 4,
    "E": 3,
    "F": 2,
    "G": 1,
}


# ─── Columns excluded from modelling ────────────────────────────────────────

DROP_COLS = [
    "id",
    "last_major_derog_none",
    "total_rec_late_fee",
]


# ─── Eligibility rules ──────────────────────────────────────────────────────

APPROVAL_RULES = {
    "max_dti": 40.0,
    "min_annual_inc": 15000.0,
    "min_grade": 2,
    "max_revol_util": 90.0,
}


# ─── Risk thresholds ─────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.55

RISK_THRESHOLDS = {
    "low": 0.30,
    "medium": 0.55,
}


# ─── Training configuration ─────────────────────────────────────────────────

RANDOM_STATE = 42

TEST_SIZE = 0.20

SMOTE_STRATEGY = "minority"


# ─── Random Forest hyperparameters ──────────────────────────────────────────

RF_PARAM_GRID = {
    "model__n_estimators": [
        100,
        300,
    ],
    "model__min_samples_split": [
        2,
        5,
    ],
    "model__min_samples_leaf": [
        1,
        2,
    ],
    "model__max_depth": [
        None,
        20,
    ],
}