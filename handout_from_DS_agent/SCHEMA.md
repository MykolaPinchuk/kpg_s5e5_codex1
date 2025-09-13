Input Schema (Train & Predict)

Required columns:
- id: int64 (unique row identifier; used for submission/outputs)
- Gender or Sex: string categorical (male/female)
- Age: float or int (years)
- Height: float or int (cm)
- Weight: float or int (kg)
- Duration: float or int (minutes)
- Heart_Rate: float or int (bpm)
- Body_Temp: float or int (Celsius)

Training-only column:
- Calories: float (target; only present in train CSV)

Notes:
- If column `Sex` is provided instead of `Gender`, it will be normalized internally.
- Additional columns are ignored.
- No missing handling is included; inputs should be fully populated (as in Kaggle data).

