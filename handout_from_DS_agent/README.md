Production Handout: PS S5E5 Calorie Prediction

Contents
- model.py: Feature engineering, preprocessor builder, and model wrapper.
- train.py: Trains an XGBoost model with early stopping and saves artifact.
- predict.py: Loads the artifact and writes batch predictions.
- stream_predict.py: Emits JSONL predictions for streaming simulation.
- config.yaml: Tunable parameters (paths, training, model).

Quick Start (Self-contained)
- Install dependencies with your stack (see Dependencies below).
- Train on sample data: `python train.py`
- Batch predict on sample data: `python predict.py`
- Stream simulate (JSONL to stdout): `python stream_predict.py --sleep 0.05 --limit 50`

Inputs
- By default uses local `data_sample/` with `train.csv` and `test.csv` (subsampled from the original competition data) to enable fast training and deployment simulation.
- Schema (see SCHEMA.md):
  - id, Gender (or Sex), Age, Height, Weight, Duration, Heart_Rate, Body_Temp

Outputs
- model.joblib: Serialized preprocessor + XGBoost booster wrapped in a simple class.
- submission.csv: id,Calories (batch predictions)
- stream to stdout: newline-delimited JSON records `{id, Calories}`

Notes
- Target is optimized in log space (log1p/expm1) to align with RMSLE.
- Preprocessing: OneHotEncoder for Gender/Sex, all other numeric passthrough.
- Features: Adds BMI, Workload, Intensity, per‑kg workload, squared/log transforms, and cross terms.
- Early stopping on a 10% validation split; parameters are in config.yaml.
- Preprocessor is fit once and saved inside the artifact (ModelWrapper).
- To point to your own data dir: `python train.py --data-dir /path/to/data` and `python predict.py --data-dir /path/to/data`.

Dependencies
- Choose your own stack; the code expects the following Python packages (versions flexible):
  - pandas, numpy, scikit-learn, xgboost, joblib, pyyaml

Usage Scenarios
- Batch analytics: Provide a CSV of workout records; receive a CSV with calories predictions for analysis dashboards or batch processing.
- Streaming/near‑real‑time: Use `stream_predict.py` as a reference to score one record at a time and emit a JSONL stream. Wrap this into your service of choice.
