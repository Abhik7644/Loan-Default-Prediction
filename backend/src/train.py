"""
train.py
--------

Train the loan default prediction model.

Pipeline:

    Raw dataset
        ↓
    Basic preprocessing
        ↓
    Train/Test split
        ↓
    Imputation
        ↓
    Scaling / One-Hot Encoding
        ↓
    SMOTE
        ↓
    Logistic Regression
        ↓
    Hyperparameter tuning
        ↓
    Evaluation
        ↓
    Saved model

Run:

    python src/train.py
"""

import os
import sys

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


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
    MODEL_DIR,
    DEFAULT_MODEL_PATH,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    RANDOM_STATE,
    TEST_SIZE,
    SMOTE_STRATEGY,
)

from src.preprocess import (
    load_raw,
    prepare_training_data,
)


# ---------------------------------------------------------------------------
# Logistic Regression hyperparameter search space
# ---------------------------------------------------------------------------

LR_PARAM_GRID = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"],
    "model__class_weight": [None, "balanced"],
}


# ---------------------------------------------------------------------------
# Build preprocessing + model pipeline
# ---------------------------------------------------------------------------

def build_pipeline(
    numerical_cols,
    categorical_cols,
):
    """
    Build the complete ML pipeline.

    Numerical features:
        Missing values → mean imputation
        Scaling → MinMaxScaler

    Categorical features:
        Missing values → most frequent
        Encoding → OneHotEncoder

    Then:
        SMOTE → Logistic Regression
    """

    numerical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="mean"),
            ),
            (
                "scaler",
                MinMaxScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            (
                "numerical",
                numerical_pipeline,
                numerical_cols,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_cols,
            ),
        ]
    )

    pipeline = ImbPipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "smote",
                SMOTE(
                    sampling_strategy=SMOTE_STRATEGY,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return pipeline


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the final model on the untouched test set.
    """

    # Class prediction using default threshold = 0.50
    y_pred = model.predict(X_test)

    # Probability of Default
    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    f1 = f1_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
    )

    recall = recall_score(
        y_test,
        y_pred,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    print(
        f"\nF1 Score      : {f1:.4f}"
    )

    print(
        f"Precision     : {precision:.4f}"
    )

    print(
        f"Recall        : {recall:.4f}"
    )

    print(
        f"ROC-AUC       : {roc_auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Default",
                "Default",
            ],
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model():
    """
    Complete training workflow.
    """

    print("\n" + "=" * 60)
    print("LOAN DEFAULT MODEL TRAINING")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------

    df = load_raw()

    print(
        f"\n[train] Raw dataset shape: {df.shape}"
    )

    # ---------------------------------------------------------------
    # 2. Basic deterministic preprocessing
    # ---------------------------------------------------------------

    X, y = prepare_training_data(df)

    print(
        f"[train] Feature matrix shape: {X.shape}"
    )

    print(
        f"[train] Target distribution:\n{y.value_counts()}"
    )

    # ---------------------------------------------------------------
    # 3. Train/Test split
    # ---------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"\n[train] Training samples: {len(X_train):,}"
    )

    print(
        f"[train] Test samples    : {len(X_test):,}"
    )

    # ---------------------------------------------------------------
    # 4. Determine actual columns present
    # ---------------------------------------------------------------

    numerical_cols = [
        column
        for column in NUMERICAL_COLS
        if column in X_train.columns
    ]

    # Grade is already numeric after prepare_training_data()
    if "grade" in X_train.columns:
        numerical_cols.append("grade")

    categorical_cols = [
        column
        for column in CATEGORICAL_COLS
        if column in X_train.columns
    ]

    print(
        f"\n[train] Numerical columns: {numerical_cols}"
    )

    print(
        f"[train] Categorical columns: {categorical_cols}"
    )

    # ---------------------------------------------------------------
    # 5. Build pipeline
    # ---------------------------------------------------------------

    pipeline = build_pipeline(
        numerical_cols,
        categorical_cols,
    )

    # ---------------------------------------------------------------
    # 6. Hyperparameter tuning
    # ---------------------------------------------------------------

    print(
        "\n[train] Starting Logistic Regression RandomizedSearchCV..."
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=LR_PARAM_GRID,
        n_iter=10,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(
        X_train,
        y_train,
    )

    print(
        "\n[train] Best parameters:"
    )

    print(search.best_params_)

    print(
        f"\n[train] Best CV F1: "
        f"{search.best_score_:.4f}"
    )

    # ---------------------------------------------------------------
    # 7. Evaluate on untouched test set
    # ---------------------------------------------------------------

    metrics = evaluate_model(
        search.best_estimator_,
        X_test,
        y_test,
    )

    # ---------------------------------------------------------------
    # 8. Save final pipeline
    # ---------------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        search.best_estimator_,
        DEFAULT_MODEL_PATH,
    )

    print(
        f"\n[train] Model saved to:"
        f"\n        {DEFAULT_MODEL_PATH}"
    )

    print(
        "\n[train] ✓ Training completed successfully."
    )

    return search.best_estimator_, metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_model()