import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from model import add_features, build_preprocessor, ModelWrapper


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def rmsle_from_logspace(y_log_true, y_log_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_log_true, y_log_pred)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="model.joblib")
    parser.add_argument("--metrics", default="metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_csv = os.path.join(args.data_dir, cfg["data"]["train_csv"])

    df = pd.read_csv(train_csv)
    y = df["Calories"].astype(float)
    X = df.drop(columns=["Calories"])

    Xf = add_features(X)
    pre = build_preprocessor(Xf)

    X_tr, X_val, y_tr, y_val = train_test_split(
        Xf, np.log1p(y),
        test_size=cfg["train"]["valid_size"], random_state=cfg["train"]["random_state"]
    )

    Xt_tr = pre.fit_transform(X_tr)
    Xt_val = pre.transform(X_val)

    dtr = xgb.DMatrix(Xt_tr, label=y_tr, feature_names=pre.get_feature_names_out().tolist())
    dval = xgb.DMatrix(Xt_val, label=y_val, feature_names=pre.get_feature_names_out().tolist())

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": cfg["model"]["eta"],
        "max_depth": cfg["model"]["max_depth"],
        "min_child_weight": cfg["model"]["min_child_weight"],
        "subsample": cfg["model"]["subsample"],
        "colsample_bytree": cfg["model"]["colsample_bytree"],
        "lambda": cfg["model"]["reg_lambda"],
        "alpha": cfg["model"]["reg_alpha"],
        "tree_method": cfg["model"]["tree_method"],
        "seed": cfg["train"]["random_state"],
    }

    booster = xgb.train(
        params,
        dtr,
        num_boost_round=cfg["train"]["num_boost_round"],
        evals=[(dval, "valid")],
        early_stopping_rounds=cfg["train"]["early_stopping_rounds"],
        verbose_eval=False,
    )

    # metrics
    val_pred_log = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    rmsle_val = rmsle_from_logspace(y_val, val_pred_log)

    wrapper = ModelWrapper(pre, booster, feature_names=pre.get_feature_names_out().tolist())
    joblib.dump(wrapper, args.out)

    with open(args.metrics, "w") as f:
        json.dump({"valid_rmsle": rmsle_val, "best_iteration": int(booster.best_iteration)}, f)

    print(f"Saved model to {args.out}; valid RMSLE={rmsle_val:.5f}; best_iter={booster.best_iteration}")


if __name__ == "__main__":
    main()

