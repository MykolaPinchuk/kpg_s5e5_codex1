Model Card — PS S5E5 Calorie Prediction (Handout)

Overview
- Task: Predict Calories burned (regression) given demographic and workout features.
- Data: Kaggle PS S5E5 (sampled locally in `data_sample/`).
- Target: `Calories` (continuous). Metric optimized: RMSLE via log-target training.

Intended Use
- Batch scoring over CSVs (predict.py) or simple streaming simulation (stream_predict.py).
- Not intended for clinical or safety-critical decisions.

Training
- Algorithm: XGBoost (hist) trained on log1p(target) with early stopping on 10% holdout.
- Preprocessing: OneHot for Gender/Sex; numeric passthrough.
- Features: BMI; Workload (Duration*Heart_Rate); Intensity (HR/(220-Age)); Workload_per_kg; squared/log transforms; cross terms (Duration×BMI, HR×BMI); Temp_Delta.

Validation
- On sample split (`data_sample/train.csv`): RMSLE ≈ 0.065 ± small.
- On full Kaggle training (external to handout): ~0.060 OOF RMSLE, sub-0.06 with ensemble.

Limitations
- Trained on synthetic derivative; may not generalize to unseen distributions.
- Requires complete inputs; no missing data handling built-in.

Security/Privacy
- No PII used beyond basic demographics.
- No external network calls; entirely offline.

Reproducibility
- Config: `config.yaml` holds training/early stopping and model params.
- Artefact: `model.joblib` contains preprocessor + booster.

