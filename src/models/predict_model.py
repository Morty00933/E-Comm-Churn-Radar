from __future__ import annotations
import argparse, json
import pandas as pd
from joblib import load

def predict(model_path: str, input_csv: str, output_csv: str):
    model = load(model_path)
    df = pd.read_csv(input_csv)
    X = df.drop(columns=["user_id","churn"], errors="ignore")
    proba = model.predict_proba(X)[:,1]
    df["churn_proba"] = proba
    df.to_csv(output_csv, index=False)
    print(f"Wrote predictions to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--input", default="data/processed.csv")
    parser.add_argument("--output", default="data/predictions.csv")
    args = parser.parse_args()
    predict(args.model, args.input, args.output)
