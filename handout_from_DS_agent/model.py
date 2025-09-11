import numpy as np
import pandas as pd
from typing import Any, List
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb


def _safe_divide(a, b, default=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a, b)
    out = pd.Series(out)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(default)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Gender" not in df.columns and "Sex" in df.columns:
        df["Gender"] = df["Sex"]
        df = df.drop(columns=["Sex"])  # unify name
    if "id" in df.columns:
        df = df.drop(columns=["id"])  # not predictive
    if set(["Height", "Weight"]).issubset(df.columns):
        h_m = df["Height"].astype(float) / 100.0
        with np.errstate(divide="ignore", invalid="ignore"):
            bmi = df["Weight"].astype(float) / np.square(h_m)
        df["BMI"] = np.clip(bmi.replace([np.inf, -np.inf], np.nan).fillna(bmi.median()), 10, 60)
    if set(["Duration", "Heart_Rate"]).issubset(df.columns):
        df["Workload"] = df["Duration"].astype(float) * df["Heart_Rate"].astype(float)
        df["Duration2"] = np.square(df["Duration"].astype(float))
        df["Heart_Rate2"] = np.square(df["Heart_Rate"].astype(float))
        df["log_Duration"] = np.log1p(np.maximum(df["Duration"].astype(float), 0))
        df["log_Heart_Rate"] = np.log1p(np.maximum(df["Heart_Rate"].astype(float), 0))
    if "Body_Temp" in df.columns:
        df["Temp_Delta"] = df["Body_Temp"].astype(float) - 36.8
    if set(["Heart_Rate", "Age"]).issubset(df.columns):
        hr_max = 220.0 - df["Age"].astype(float)
        intensity = _safe_divide(df["Heart_Rate"].astype(float), hr_max, default=np.nan)
        df["Intensity"] = np.clip(intensity.fillna(intensity.median()), 0.2, 2.0)
    if set(["Workload", "Weight"]).issubset(df.columns):
        perkg = _safe_divide(df["Workload"], df["Weight"].astype(float), default=np.nan)
        df["Workload_per_kg"] = np.clip(perkg.fillna(perkg.median()), 0, None)
    if set(["Duration", "BMI"]).issubset(df.columns):
        df["Duration_x_BMI"] = df["Duration"].astype(float) * df["BMI"].astype(float)
    if set(["Heart_Rate", "BMI"]).issubset(df.columns):
        df["HR_x_BMI"] = df["Heart_Rate"].astype(float) * df["BMI"].astype(float)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )
    return pre


@dataclass
class ModelWrapper:
    preprocessor: Any
    booster: xgb.Booster
    feature_names: List[str]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = add_features(df)
        Xt = self.preprocessor.transform(X)
        d = xgb.DMatrix(Xt, feature_names=self.feature_names)
        pred_log = self.booster.predict(d, iteration_range=(0, getattr(self.booster, 'best_iteration', None)))
        return np.expm1(pred_log)

