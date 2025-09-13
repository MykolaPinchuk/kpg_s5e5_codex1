import argparse
import json
import os
import sys
import time

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    parser.add_argument("--data", default=os.path.join(here, "data_sample", "test.csv"))
    parser.add_argument("--model", default=os.path.join(here, "model.joblib"))
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between records")
    parser.add_argument("--limit", type=int, default=100, help="Max records to emit (0=all)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    model = joblib.load(args.model)

    total = len(df) if args.limit == 0 else min(args.limit, len(df))
    for i in range(total):
        row = df.iloc[[i]].copy()
        pred = float(model.predict(row)[0])
        if "id" in row.columns:
            rid = int(row["id"].iloc[0])
        else:
            rid = i
        rec = {"id": rid, "Calories": pred}
        sys.stdout.write(json.dumps(rec) + "\n")
        sys.stdout.flush()
        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
