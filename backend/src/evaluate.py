"""
evaluate.py
-----------
Compare multiple ML models using the same preprocessing,
train/test split, and evaluation strategy.

Models compared:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest

Run:

    python src/evaluate.py
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
    )
)

from config import (
    RANDOM_STATE,
    TEST_SIZE,
    SMOTE_STRATEGY,
    DEFAULT_MODEL_PATH,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
)

from src.preprocess import (
    load_raw,
    prepare_training_data,
)

# ---------------------------------------------------------------------------
# Scikit-learn
# ---------------------------------------------------------------------------

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
)

from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline as ImbPipeline


# ---------------------------------------------------------------------------
# Build preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(
    numerical_cols,
    categorical_cols,
):
    """
    Create the common preprocessing pipeline used by
    every model.
    """

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="mean"),
        ),
        (
            "scaler",
            MinMaxScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
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
    ])

    return ColumnTransformer([
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
    ])


# ---------------------------------------------------------------------------
# Build complete model pipeline
# ---------------------------------------------------------------------------

def build_model_pipeline(
    model,
    numerical_cols,
    categorical_cols,
):
    """
    Create:

        preprocessing
            ↓
        SMOTE
            ↓
        model

    Every model gets the exact same preprocessing.
    """

    preprocessor = build_preprocessor(
        numerical_cols,
        categorical_cols,
    )

    return ImbPipeline([
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
            model,
        ),
    ])


# ---------------------------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------------------------

def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train and evaluate one model.
    """

    print("\n" + "=" * 60)
    print(f"MODEL: {name}")
    print("=" * 60)

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(
        X_test
    )

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

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

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
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
            zero_division=0,
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
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("LOAN DEFAULT MODEL COMPARISON")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load and prepare data
    # -----------------------------------------------------------------------

    df = load_raw()

    X, y = prepare_training_data(
        df
    )

    # -----------------------------------------------------------------------
    # 2. Same train/test split used by train.py
    # -----------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"\n[evaluate] Training samples: "
        f"{len(X_train):,}"
    )

    print(
        f"[evaluate] Test samples: "
        f"{len(X_test):,}"
    )

    # -----------------------------------------------------------------------
    # 3. Determine feature columns
    # -----------------------------------------------------------------------

    numerical_cols = [
        column
        for column in NUMERICAL_COLS
        if column in X_train.columns
    ]

    if "grade" in X_train.columns:
        numerical_cols.append("grade")

    categorical_cols = [
        column
        for column in CATEGORICAL_COLS
        if column in X_train.columns
    ]

    print(
        f"\n[evaluate] Numerical columns:"
        f"\n{numerical_cols}"
    )

    print(
        f"\n[evaluate] Categorical columns:"
        f"\n{categorical_cols}"
    )

    # -----------------------------------------------------------------------
    # 4. Define models
    # -----------------------------------------------------------------------

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # -----------------------------------------------------------------------
    # 5. Train and evaluate models
    # -----------------------------------------------------------------------

    results = []

    for name, model in models.items():

        pipeline = build_model_pipeline(
            model,
            numerical_cols,
            categorical_cols,
        )

        result = evaluate_model(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(
            result
        )

    # -----------------------------------------------------------------------
    # 6. Results comparison
    # -----------------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # -----------------------------------------------------------------------
    # 7. Best model by F1
    # -----------------------------------------------------------------------

    best_model = results_df.loc[
        results_df["F1"].idxmax()
    ]

    print("\n" + "=" * 60)
    print("BEST MODEL BY F1")
    print("=" * 60)

    print(
        f"Model     : {best_model['Model']}"
    )

    print(
        f"F1        : {best_model['F1']:.4f}"
    )

    print(
        f"Precision : {best_model['Precision']:.4f}"
    )

    print(
        f"Recall    : {best_model['Recall']:.4f}"
    )

    print(
        f"ROC-AUC   : {best_model['ROC-AUC']:.4f}"
    )

    # -----------------------------------------------------------------------
    # 8. Load and verify saved final model
    # -----------------------------------------------------------------------

    if os.path.exists(
        DEFAULT_MODEL_PATH
    ):

        print("\n" + "=" * 60)
        print("SAVED FINAL MODEL")
        print("=" * 60)

        saved_model = joblib.load(
            DEFAULT_MODEL_PATH
        )

        saved_pred = saved_model.predict(
            X_test
        )

        saved_probability = saved_model.predict_proba(
            X_test
        )[:, 1]

        saved_f1 = f1_score(
            y_test,
            saved_pred,
            zero_division=0,
        )

        saved_roc_auc = roc_auc_score(
            y_test,
            saved_probability,
        )

        print(
            f"Saved model F1      : {saved_f1:.4f}"
        )

        print(
            f"Saved model ROC-AUC : {saved_roc_auc:.4f}"
        )

        print(
            "\n[evaluate] Saved model loaded successfully."
        )

    else:

        print(
            "\n[evaluate] WARNING: Saved model not found."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()