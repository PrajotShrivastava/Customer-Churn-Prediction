"""Shared constants and helpers used by notebooks 05-09.

Kept deliberately small: only code that MUST be identical across notebooks
(paths, column lists, feature engineering, preprocessing) lives here.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "churn_cleaned.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TARGET = "Churn"

# ---------------------------------------------------------------- columns
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Plot colours (validated colour-blind-safe categorical pair)
COLOR_RETAINED = "#2a78d6"   # blue   -> Churn = No
COLOR_CHURNED = "#eb6834"    # orange -> Churn = Yes
CHURN_PALETTE = {"No": COLOR_RETAINED, "Yes": COLOR_CHURNED}


def set_plot_style():
    """Consistent, low-clutter matplotlib style for all notebooks."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e5e5e3",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })


def load_split():
    """Return X_train, X_test, y_train, y_test from the saved stratified split (notebook 05)."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------- feature engineering
ENGINEERED_NUMERIC = ["num_addon_services"]
ENGINEERED_CATEGORICAL = ["tenure_group"]


def add_engineered_features(X):
    """Row-wise features (no statistics learned from data -> no leakage risk)."""
    X = X.copy()
    # How many optional add-on services the customer subscribes to (0-6)
    X["num_addon_services"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)
    # Customer lifecycle stage; churn risk is highly non-linear in tenure
    X["tenure_group"] = pd.cut(
        X["tenure"], bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12m", "13-24m", "25-48m", "49m+"],
    ).astype(str)
    return X


# ---------------------------------------------------------------- preprocessing
def build_preprocessor(engineered=False):
    """Scale numeric columns and one-hot encode categorical columns."""
    numeric = NUMERIC_FEATURES + (ENGINEERED_NUMERIC if engineered else [])
    categorical = CATEGORICAL_FEATURES + (ENGINEERED_CATEGORICAL if engineered else [])
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        # drop='if_binary' turns Yes/No columns into a single 0/1 column;
        # handle_unknown='ignore' keeps the pipeline from crashing on unseen categories
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                              sparse_output=False), categorical),
    ])


def build_pipeline(model, engineered=False):
    """Full pipeline: [feature engineering] -> preprocessing -> model."""
    steps = []
    if engineered:
        steps.append(("features", FunctionTransformer(add_engineered_features)))
    steps.append(("preprocess", build_preprocessor(engineered)))
    steps.append(("model", model))
    return Pipeline(steps)
