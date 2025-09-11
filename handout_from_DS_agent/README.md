Production Handout: PS S5E5 Calorie Prediction

Contents
- model.py: Feature engineering, preprocessor builder, and model wrapper.
- train.py: Trains an XGBoost model with early stopping and saves artifact.
- predict.py: Loads the artifact and writes a Kaggle‑ready submission.
- config.yaml: Tunable parameters (paths, training, model).
- requirements.txt: Python dependencies pinned to tested versions.

Quick Start
- Train: python train.py --data-dir ../data --out model.joblib
- Predict: python predict.py --data-dir ../data --model model.joblib --out submission.csv

Inputs
- Expects CSVs in --data-dir: train.csv, test.csv with columns:
  - id, Gender (or Sex), Age, Height, Weight, Duration, Heart_Rate, Body_Temp

Outputs
- model.joblib: Serialized preprocessor + XGBoost booster wrapped in a simple class.
- submission.csv: id,Calories

Notes
- Target is optimized in log space (log1p/expm1) to align with RMSLE.
- Preprocessing: OneHotEncoder for Gender/Sex, all other numeric passthrough.
- Features: Adds BMI, Workload, Intensity, per‑kg workload, squared/log transforms, and a few cross terms.
- Early stopping on a 10% validation split; parameters are in config.yaml.
- For reproducibility and stability, preprocessor is fit once and saved inside the artifact.

