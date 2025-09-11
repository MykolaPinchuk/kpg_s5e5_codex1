import os
import argparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from numpy import log1p, expm1
import xgboost as xgb
import lightgbm as lgb


def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))


def load_data(data_dir: str):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def _safe_divide(a, b, default=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a, b)
    out = pd.Series(out)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(default)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize naming: some datasets use 'Sex' vs 'Gender'
    if "Gender" not in df.columns and "Sex" in df.columns:
        df["Gender"] = df["Sex"]
        df = df.drop(columns=["Sex"])  # avoid duplicate
    # Drop id from features if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    # Add BMI if Height/Weight present
    if set(["Height", "Weight"]).issubset(df.columns):
        h_m = df["Height"].astype(float) / 100.0
        with np.errstate(divide="ignore", invalid="ignore"):
            bmi = df["Weight"].astype(float) / np.square(h_m)
        df["BMI"] = np.clip(bmi.replace([np.inf, -np.inf], np.nan).fillna(bmi.median()), 10, 60)
    # Add simple interaction capturing workload
    if set(["Duration", "Heart_Rate"]).issubset(df.columns):
        df["Workload"] = df["Duration"].astype(float) * df["Heart_Rate"].astype(float)
        # Squared and log transforms
        df["Duration2"] = np.square(df["Duration"].astype(float))
        df["Heart_Rate2"] = np.square(df["Heart_Rate"].astype(float))
        df["log_Duration"] = np.log1p(np.maximum(df["Duration"].astype(float), 0))
        df["log_Heart_Rate"] = np.log1p(np.maximum(df["Heart_Rate"].astype(float), 0))
    # Temperature deviation from normal
    if "Body_Temp" in df.columns:
        df["Temp_Delta"] = df["Body_Temp"].astype(float) - 36.8
    # Intensity features relative to age
    if set(["Heart_Rate", "Age"]).issubset(df.columns):
        hr_max = 220.0 - df["Age"].astype(float)
        intensity = _safe_divide(df["Heart_Rate"].astype(float), hr_max, default=np.nan)
        df["Intensity"] = np.clip(intensity.fillna(intensity.median()), 0.2, 2.0)
    # Per-kg normalization
    if set(["Workload", "Weight"]).issubset(df.columns):
        perkg = _safe_divide(df["Workload"], df["Weight"].astype(float), default=np.nan)
        df["Workload_per_kg"] = np.clip(perkg.fillna(perkg.median()), 0, None)
    # Additional cross terms
    if set(["Duration", "BMI"]).issubset(df.columns):
        df["Duration_x_BMI"] = df["Duration"].astype(float) * df["BMI"].astype(float)
    if set(["Heart_Rate", "BMI"]).issubset(df.columns):
        df["HR_x_BMI"] = df["Heart_Rate"].astype(float) * df["BMI"].astype(float)
    return df


def build_rf_pipeline(train_df: pd.DataFrame) -> Pipeline:
    # Features
    target = "Calories"
    feature_cols = [c for c in train_df.columns if c not in (target,)]

    # Identify categorical and numeric columns
    categorical_cols = [c for c in feature_cols if train_df[c].dtype == "object"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "reg",
                TransformedTargetRegressor(
                    regressor=rf, func=log1p, inverse_func=expm1
                ),
            ),
        ]
    )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory with CSV files")
    parser.add_argument(
        "--subsample",
        type=int,
        default=8000,
        help="Rows to use for quick CV (<= len(train))",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--model", choices=["rf", "xgb", "best"], default="xgb")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    train_df, test_df = load_data(args.data_dir)
    # Feature engineering and cleanup
    train_df = add_features(train_df)
    test_df_fe = add_features(test_df)

    # Drop id from features if present; keep it for submission
    id_col = "id" if "id" in test_df.columns else None
    target = "Calories"

    if args.model == "rf":
        # Build RF pipeline
        model = build_rf_pipeline(train_df)

        # Quick CV on a subsample to estimate RMSLE
        n = min(args.subsample, len(train_df))
        if n < len(train_df):
            cv_df = train_df.sample(n=n, random_state=42)
        else:
            cv_df = train_df

        X_cv = cv_df.drop(columns=[target])
        y_cv = cv_df[target].values

        scorer = make_scorer(rmsle, greater_is_better=False)
        cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

        scores = cross_val_score(model, X_cv, y_cv, scoring=scorer, cv=cv, n_jobs=1)
        print(f"CV RMSLE (neg values, higher is better): {scores}")
        print(f"Mean RMSLE: {-scores.mean():.5f} +/- {scores.std():.5f}")

        # Fit on full training data
        X_train = train_df.drop(columns=[target])
        y_train = train_df[target].values
        model.fit(X_train, y_train)

        # Predict test
        X_test = test_df_fe.copy()
        preds = model.predict(X_test)
    elif args.model == "xgb":
        # XGBoost path with log-target (RMSLE) training and 5-fold CV ensembling
        X = train_df.drop(columns=[target])
        y = train_df[target].astype(float)
        y_log = np.log1p(y)

        categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
        numeric_cols = [c for c in X.columns if c not in categorical_cols]

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": 0.03,
            "max_depth": 6,
            "min_child_weight": 2,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "alpha": 0.0,
            "lambda": 1.0,
            "tree_method": "hist",
            "seed": 42,
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        val_scores = []
        test_pred_log_accum = None

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[va_idx]

            pre = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                    ("num", "passthrough", numeric_cols),
                ]
            )
            X_tr_t = pre.fit_transform(X_tr)
            X_val_t = pre.transform(X_val)
            X_test_t = pre.transform(test_df_fe)

            dtr = xgb.DMatrix(X_tr_t, label=y_tr)
            dva = xgb.DMatrix(X_val_t, label=y_val)
            dte = xgb.DMatrix(X_test_t)

            bst = xgb.train(
                params,
                dtr,
                num_boost_round=5000,
                evals=[(dva, "valid")],
                early_stopping_rounds=200,
                verbose_eval=False,
            )

            val_pred_log = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
            from sklearn.metrics import mean_squared_error

            val_rmsle = np.sqrt(mean_squared_error(y_val, val_pred_log))
            val_scores.append(val_rmsle)
            print(f"Fold {fold} RMSLE: {val_rmsle:.5f} | best_iter={bst.best_iteration}")

            te_pred_log = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
            if test_pred_log_accum is None:
                test_pred_log_accum = te_pred_log
            else:
                test_pred_log_accum += te_pred_log

        mean_rmsle = float(np.mean(val_scores))
        std_rmsle = float(np.std(val_scores))
        print(f"CV RMSLE: {mean_rmsle:.5f} +/- {std_rmsle:.5f}")

        preds = np.expm1(test_pred_log_accum / kf.get_n_splits())
    elif args.model == "best":
        # Stronger ensemble: XGBoost + LightGBM in log space with CV and early stopping
        X = train_df.drop(columns=[target])
        y = train_df[target].astype(float)
        y_log = np.log1p(y)

        categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
        numeric_cols = [c for c in X.columns if c not in categorical_cols]

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", "passthrough", numeric_cols),
            ]
        )
        # Fit once on full data to keep stable feature names
        pre.fit(X)
        feat_names = pre.get_feature_names_out().tolist()

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_xgb = np.zeros(len(X))
        oof_lgb = np.zeros(len(X))
        val_scores_xgb = []
        val_scores_lgb = []
        imp_xgb_accum = np.zeros(len(feat_names), dtype=float)
        imp_lgb_accum = np.zeros(len(feat_names), dtype=float)
        te_pred_xgb_log_accum = None
        te_pred_lgb_log_accum = None

        X_all_t = pre.transform(X)
        X_test_t = pre.transform(test_df_fe)
        dtest = xgb.DMatrix(X_test_t, feature_names=feat_names)
        lgb_test = lgb.Dataset(X_test_t, feature_name=feat_names, free_raw_data=False)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
            X_tr_t = X_all_t[tr_idx]
            X_val_t = X_all_t[va_idx]
            y_tr = y_log.iloc[tr_idx]
            y_val = y_log.iloc[va_idx]

            # XGBoost
            dtr = xgb.DMatrix(X_tr_t, label=y_tr, feature_names=feat_names)
            dva = xgb.DMatrix(X_val_t, label=y_val, feature_names=feat_names)
            params_xgb = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "eta": 0.03,
                "max_depth": 7,
                "min_child_weight": 3,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "lambda": 1.0,
                "alpha": 0.0,
                "tree_method": "hist",
                "seed": 42 + fold,
            }
            bst_xgb = xgb.train(
                params_xgb,
                dtr,
                num_boost_round=6000,
                evals=[(dva, "valid")],
                early_stopping_rounds=300,
                verbose_eval=False,
            )

            val_pred_xgb = bst_xgb.predict(dva, iteration_range=(0, bst_xgb.best_iteration + 1))
            from sklearn.metrics import mean_squared_error

            rmsle_xgb = np.sqrt(mean_squared_error(y_val, val_pred_xgb))
            val_scores_xgb.append(rmsle_xgb)
            oof_xgb[va_idx] = val_pred_xgb

            # importance for xgb (align to feat_names)
            gain_map = bst_xgb.get_score(importance_type="gain")
            # ensure we map all features, default 0.0 if unseen
            imp_xgb_accum += np.array([gain_map.get(name, 0.0) for name in feat_names])

            te_pred_xgb = bst_xgb.predict(dtest, iteration_range=(0, bst_xgb.best_iteration + 1))
            te_pred_xgb_log_accum = te_pred_xgb if te_pred_xgb_log_accum is None else te_pred_xgb_log_accum + te_pred_xgb

            # LightGBM
            lgb_tr = lgb.Dataset(X_tr_t, label=y_tr, feature_name=feat_names, free_raw_data=False)
            lgb_va = lgb.Dataset(X_val_t, label=y_val, reference=lgb_tr, feature_name=feat_names, free_raw_data=False)
            params_lgb = {
                "objective": "rmse",
                "metric": "rmse",
                "learning_rate": 0.03,
                "num_leaves": 64,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.85,
                "bagging_freq": 1,
                "min_data_in_leaf": 20,
                "lambda_l2": 1.0,
                "verbosity": -1,
                "num_threads": -1,
                "seed": 42 + fold,
            }
            bst_lgb = lgb.train(
                params_lgb,
                lgb_tr,
                num_boost_round=6000,
                valid_sets=[lgb_va],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
            )

            val_pred_lgb = bst_lgb.predict(X_val_t, num_iteration=bst_lgb.best_iteration)
            rmsle_lgb = np.sqrt(mean_squared_error(y_val, val_pred_lgb))
            val_scores_lgb.append(rmsle_lgb)
            oof_lgb[va_idx] = val_pred_lgb

            imp_lgb_accum += bst_lgb.feature_importance(importance_type="gain")

            te_pred_lgb = bst_lgb.predict(X_test_t, num_iteration=bst_lgb.best_iteration)
            te_pred_lgb_log_accum = te_pred_lgb if te_pred_lgb_log_accum is None else te_pred_lgb_log_accum + te_pred_lgb

            print(f"Fold {fold} RMSLE - XGB: {rmsle_xgb:.5f} | LGB: {rmsle_lgb:.5f} | xgb_best_iter={bst_xgb.best_iteration} lgb_best_iter={bst_lgb.best_iteration}")

        # Choose blend weight on OOF by small grid
        weights = [0.3, 0.4, 0.5, 0.6, 0.7]
        from sklearn.metrics import mean_squared_error
        best_w, best_rmse = None, 1e9
        for w in weights:
            oof_blend = w * oof_xgb + (1 - w) * oof_lgb
            rmse = np.sqrt(mean_squared_error(y_log, oof_blend))
            if rmse < best_rmse:
                best_rmse = rmse
                best_w = w
        print(f"OOF RMSLE XGB: {np.mean(val_scores_xgb):.5f} | LGB: {np.mean(val_scores_lgb):.5f} | Blend(best_w={best_w}): {best_rmse:.5f}")

        # Final blended prediction in log space
        te_pred_xgb_log = te_pred_xgb_log_accum / kf.get_n_splits()
        te_pred_lgb_log = te_pred_lgb_log_accum / kf.get_n_splits()
        preds = np.expm1(best_w * te_pred_xgb_log + (1 - best_w) * te_pred_lgb_log)

        # Save feature importances
        fi_xgb = pd.DataFrame({"feature": feat_names, "gain": imp_xgb_accum / kf.get_n_splits(), "model": "xgb"})
        fi_lgb = pd.DataFrame({"feature": feat_names, "gain": imp_lgb_accum / kf.get_n_splits(), "model": "lgb"})
        fi_all = pd.concat([fi_xgb, fi_lgb], ignore_index=True)

        # Aggregate importance by base feature (collapse OHE)
        def base_feature(name: str) -> str:
            if "__" in name:
                base = name.split("__", 1)[1]
            else:
                base = name
            # For OneHot, keep the part before the category value
            if base.startswith("Gender_") or base.startswith("Sex_"):
                return "Gender/Sex"
            return base.split("_")[0] if base.startswith("cat") else base

        fi_all["base_feature"] = fi_all["feature"].apply(base_feature)
        fi_agg = fi_all.groupby("base_feature", as_index=False)["gain"].sum().sort_values("gain", ascending=False)
        fi_all.sort_values(["model", "gain"], ascending=[True, False]).to_csv("feature_importance_best_detailed.csv", index=False)
        fi_agg.to_csv("feature_importance_best_aggregated.csv", index=False)
        print("Top 15 aggregated feature importances:")
        print(fi_agg.head(15))
    else:
        raise ValueError(f"Unknown model: {args.model}")

    preds = np.maximum(preds, 0)  # ensure non-negative

    # Build submission
    if id_col and id_col in test_df.columns:
        sub = pd.DataFrame({"id": test_df[id_col], "Calories": preds})
    else:
        # Fallback: use row index if id is absent
        sub = pd.DataFrame({"id": np.arange(len(preds)), "Calories": preds})

    sub.to_csv(args.output, index=False)
    print(f"Saved submission to {args.output}")


if __name__ == "__main__":
    main()
