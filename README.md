# PS S5E5 — Fast POC to Near‑SOTA

This repo builds a fast, Kaggle‑ready pipeline for the Playground Series S5E5 (Predict Calorie Expenditure). It includes three model tracks: a quick baseline (RF), a strong single model (XGBoost), and a higher‑accuracy ensemble (XGBoost + LightGBM) with feature importance analysis.

## Files

- `data/`: Downloaded competition CSVs (`train.csv`, `test.csv`, `sample_submission.csv`).
- `train_and_predict.py`: End‑to‑end training and submission script.
- `submission.csv`: Baseline RF submission.
- `submission_xgb.csv`: Improved XGBoost submission.
- `submission_best.csv`: Ensemble (XGB+LGB) submission.
- `feature_importance_best_detailed.csv`: Per‑feature (post‑OHE) gains for XGB & LGB.
- `feature_importance_best_aggregated.csv`: Collapsed feature gains at base feature level.

## Usage

Prereqs: Python 3.10+, `kaggle` CLI (optional after data is downloaded), and installed packages: `xgboost`, `lightgbm`, `scikit-learn`, `pandas`, `numpy`.

1) Download data (already handled once via Kaggle API):
   - `data/train.csv`, `data/test.csv` should exist. If not, run the kaggle CLI download.

2) Run models:

- Baseline RF (quick):
  - `python train_and_predict.py --model rf --output submission.csv`

- Strong single model (XGBoost, default):
  - `python train_and_predict.py --model xgb --output submission_xgb.csv`
  - 5‑fold CV in log‑space; early stopping; OneHot for `Gender`; numeric passthrough.

- Best model (XGB + LightGBM ensemble):
  - `python train_and_predict.py --model best --output submission_best.csv`
  - 5‑fold CV for both models with early stopping; choose blend weight by OOF RMSLE; average fold test predictions; then `expm1`.

Flags:

- `--data-dir data` to point to CSVs directory.
- `--subsample 8000` for quick RF CV speed checks (RF path only).
- `--cv-folds 3` (RF path only) to adjust folds.

## Features

All models share simple, fast, and effective features:

- Drop `id`.
- OneHot encode `Gender` (binary category).
- Numeric passthrough for: `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, `Body_Temp`.
- Engineered:
  - `BMI = Weight / (Height/100)^2` (clipped to [10, 60])
  - `Workload = Duration * Heart_Rate`

## Models

- RandomForestRegressor (baseline):
  - Wrapped in `TransformedTargetRegressor` with `log1p/expm1` to optimize RMSLE.

- XGBoost (strong single model):
  - Hist tree method, log‑target, 5‑fold CV with early stopping, average test predictions.
  - Typical settings: `eta≈0.03`, depth 6, subsample 0.8–0.85, colsample 0.8–0.85.

- Best (Ensemble XGBoost + LightGBM):
  - Both trained in log space with consistent preprocessing and early stopping.
  - 5‑fold CV, compute OOF RMSLE for each model.
  - Grid search a small set of blend weights `[0.3, 0.4, 0.5, 0.6, 0.7]` on OOF to pick the best.
  - Blend test predictions in log space using the best weight and `expm1` at the end.

## Performance (local CV)

- RF baseline: ~0.081 RMSLE (fast sanity check on subsample).
- XGBoost: ~0.0599 RMSLE (5‑fold CV; near SOTA).
- Best ensemble: ~0.0596 OOF RMSLE (selected weight shown in logs). Runtime kept reasonable via hist/early stopping.

Actual Kaggle public score may be slightly different; use `submission_best.csv` for best expected leaderboard placement.

## Feature Importance

- During the `best` model run, the script writes:
  - `feature_importance_best_detailed.csv`: raw per‑feature gains per model after preprocessing (e.g., `cat__Gender_*`).
  - `feature_importance_best_aggregated.csv`: category‑level aggregation (e.g., collapse OHE back to `Gender`).
- The script also prints the top 15 aggregated features. On this dataset, top drivers typically include `Workload` (Duration × Heart_Rate), `Duration`, and `Heart_Rate`, with helpful contributions from `Age`, `Weight`, `Height`, `Body_Temp`, and `BMI`.

## Repro / Notes

- Targets are modeled in log space (`log1p`) to align with RMSLE and transformed back with `expm1`.
- Preprocessing is fit once on full training to keep feature mapping stable (safe here given only one categorical `Gender`).
- Early stopping prevents overfitting and keeps training fast; best_iteration is used for prediction.
- Seeds vary across folds for minor de‑correlation in the ensemble.

## Next Ideas (if you want to squeeze more)

- Slight parameter sweep (depth/leaves, min_child_weight/min_data_in_leaf, feature_fraction/colsample).
- Add a few more lightweight interactions (e.g., `Weight*Duration`, `Duration^2`) and re‑run.
- Try CatBoost in the blend (fast CPU, handles categoricals natively) if installable.

