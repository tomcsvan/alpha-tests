import numpy as np, joblib
from sklearn.preprocessing import StandardScaler

def fit_save_scaler(X_train, path):
    sc = StandardScaler(with_mean=True, with_std=True)
    sc.fit(X_train)
    joblib.dump(sc, path)
    return sc

def load_apply_scaler(X, path):
    sc = joblib.load(path)
    return sc.transform(X)
