"""
preprocess.py
-------------
Basic, deterministic preprocessing shared by training and prediction.

Responsibilities:
    1. Load raw data
    2. Remove columns excluded from modelling
    3. Normalize text values
    4. Encode the ordinal grade feature

Model-specific preprocessing such as:
    - missing-value imputation
    - scaling
    - one-hot encoding
    - SMOTE

is handled inside the sklearn training pipeline in train.py.
"""

import os
import sys

import pandas as pd

# Allow imports from the backend directory
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from config import (
    DATA_RAW,
    TARGET_COL,
    DROP_COLS,
    GRADE_MAP,
)


def load_raw(path: str = DATA_RAW) -> pd.DataFrame:
    """
    Load the raw loan dataset.
    """
    df = pd.read_csv(path, low_memory=False)

    print(
        f"[preprocess] Loaded "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that should not be used by the ML model.
    """
    df = df.copy()

    columns_to_drop = [
        column
        for column in DROP_COLS
        if column in df.columns
    ]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

        print(
            f"[preprocess] Dropped columns: "
            f"{columns_to_drop}"
        )

    return df


def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize inconsistent categorical/text values.
    """
    df = df.copy()

    if "term" in df.columns:
        df["term"] = (
            df["term"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    return df


def encode_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert loan grade from categorical letters to
    an ordinal numerical representation.

    A -> 7
    B -> 6
    ...
    G -> 1
    """
    df = df.copy()

    if "grade" in df.columns:
        df["grade"] = df["grade"].map(GRADE_MAP)

    return df


def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply all deterministic preprocessing steps.

    This function is intentionally shared by both
    training and prediction.
    """
    df = df.copy()

    df = drop_unused_columns(df)
    df = normalize_values(df)
    df = encode_grade(df)

    return df


def prepare_training_data(
    df: pd.DataFrame,
):
    """
    Prepare features and target for model training.
    """
    df = prepare_features(df)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' "
            "was not found in the dataset."
        )

    # Remove rows where the target itself is missing.
    df = df.dropna(subset=[TARGET_COL])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    return X, y


if __name__ == "__main__":
    df = load_raw()

    X, y = prepare_training_data(df)

    print("\n[preprocess] Preparation complete")
    print(f"[preprocess] Feature shape : {X.shape}")
    print(f"[preprocess] Target shape  : {y.shape}")
    print("\n[preprocess] Features:")
    print(X.columns.tolist())

    print("\n[preprocess] Target distribution:")
    print(y.value_counts())