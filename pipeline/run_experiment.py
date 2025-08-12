# pipeline/run_experiment.py
import os
import pathlib
import joblib
import numpy as np
import pandas as pd

from features.labels import add_returns, make_label_updown
from features.build import simple_features
from pipeline.split import time_split
from pipeline.scaler import fit_save_scaler, load_apply_scaler
from models.baseline import classify_baseline

# Base directory
ROOT = pathlib.Path(__file__).resolve().parents[1]

# IO locations
RAW = ROOT / "tests" / "sample.csv"     
ARTIFACTS = ROOT / "artifacts"        
SCALER = ARTIFACTS / "scaler.joblib"
MODEL = ARTIFACTS / "model.joblib"

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW)
    df = add_returns(df, price_col="close", horizon=1, log=True).dropna()

    X = simple_features(df).dropna()

    df = df.loc[X.index]
    y = make_label_updown(df, horizon=1).loc[X.index]

    tr, va, te = time_split(X, train_ratio=0.7, val_ratio=0.15)
    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    ytr, yva, yte = y.iloc[tr], y.iloc[va], y.iloc[te]

    sc = fit_save_scaler(Xtr.values, SCALER)
    Xtr_ = sc.transform(Xtr.values)
    Xva_ = load_apply_scaler(Xva.values, SCALER)
    Xte_ = load_apply_scaler(Xte.values, SCALER)

    metrics, model = classify_baseline(Xtr_, ytr, Xte_, yte)
    print("Baseline metrics:", metrics)

    joblib.dump(model, MODEL)
    print(f"Saved scaler -> {SCALER}")
    print(f"Saved model  -> {MODEL}")

if __name__ == "__main__":
    main()
