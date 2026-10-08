"""
threshold_analysis.py
---------------------

Analyze different probability thresholds for the
loan default prediction model.

The goal is to understand the trade-off between:

    Precision
    Recall
    F1 Score

Run:

    python src/threshold_analysis.py
"""

import os
import sys

import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


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
    RANDOM_STATE,
    TEST_SIZE,
)

from src.preprocess import (
    load_raw,
    prepare_training_data,
)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model():
    """
    Load the trained Logistic Regression pipeline.
    """

    model = joblib.load(DEFAULT_MODEL_PATH)

    print(
        f"[threshold] Loaded model from:\n"
        f"            {DEFAULT_MODEL_PATH}"
    )

    return model


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_thresholds():

    print("\n" + "=" * 60)
    print("LOAN DEFAULT THRESHOLD ANALYSIS")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------

    df = load_raw()

    X, y = prepare_training_data(df)

    # ---------------------------------------------------------------
    # 2. Same train/test split used during training
    # ---------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"\n[threshold] Test samples: {len(X_test):,}"
    )

    # ---------------------------------------------------------------
    # 3. Load trained model
    # ---------------------------------------------------------------

    model = load_model()

    # ---------------------------------------------------------------
    # 4. Get default probabilities
    # ---------------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------------------------
    # 5. Test different thresholds
    # ---------------------------------------------------------------

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    print("\n" + "-" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    print("-" * 70)

    best_threshold = None
    best_f1 = -1

    for threshold in thresholds:

        y_pred = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # ---------------------------------------------------------------
    # 6. Best threshold
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print("BEST THRESHOLD BY F1")
    print("=" * 60)

    print(
        f"\nThreshold : {best_threshold:.2f}"
    )

    print(
        f"F1 Score  : {best_f1:.4f}"
    )

    # ---------------------------------------------------------------
    # 7. Confusion matrix for best threshold
    # ---------------------------------------------------------------

    best_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            best_predictions,
        )
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyze_thresholds()