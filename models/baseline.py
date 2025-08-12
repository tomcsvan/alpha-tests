import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

def classify_baseline(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]
    yhat = (p>0.5).astype(int)
    return {
        "acc": float(accuracy_score(yte, yhat)),
        "auc": float(roc_auc_score(yte, p))
    }, clf

def regress_baseline(Xtr, ytr, Xte, yte):
    reg = LinearRegression().fit(Xtr, ytr)
    pred = reg.predict(Xte)
    return {"r2": float(r2_score(yte, pred))}, reg
