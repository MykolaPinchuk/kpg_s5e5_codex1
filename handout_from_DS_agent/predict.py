import argparse
import os
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    parser.add_argument("--data-dir", default=os.path.join(here, "data_sample"))
    parser.add_argument("--model", default=os.path.join(here, "model.joblib"))
    parser.add_argument("--out", default=os.path.join(here, "submission.csv"))
    args = parser.parse_args()

    test_csv = os.path.join(args.data_dir, "test.csv")
    df = pd.read_csv(test_csv)
    ids = df["id"] if "id" in df.columns else pd.Series(range(len(df)))

    model = joblib.load(args.model)
    preds = model.predict(df)

    sub = pd.DataFrame({"id": ids, "Calories": preds})
    sub.to_csv(args.out, index=False)
    print(f"Saved submission to {args.out}")


if __name__ == "__main__":
    main()
