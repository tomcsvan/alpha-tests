import pandas as pd, numpy as np

def rolling_z(x, win):
    mu = x.rolling(win).mean()
    sd = x.rolling(win).std(ddof=0)
    return (x - mu) / sd

def simple_features(df):
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    out["ret_1"]  = close.pct_change(1)
    out["ret_5"]  = close.pct_change(5)
    out["z_close_20"] = rolling_z(close, 20)
    out["vol_z_20"]   = rolling_z(df["volume"], 20)
    out["mom_10"] = close / close.shift(10) - 1
    return out
