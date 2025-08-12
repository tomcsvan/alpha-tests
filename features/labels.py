import numpy as np
import pandas as pd

def add_returns(df, price_col="close", horizon=1, log=True):
    if log:
        r = np.log(df[price_col]).diff(horizon)
    else:
        r = df[price_col].pct_change(horizon)
    df[f"ret_fwd_{horizon}"] = r.shift(-horizon)
    return df

def make_label_updown(df, horizon=1, thresh=0.0):
    y = (df[f"ret_fwd_{horizon}"] > thresh).astype(int)
    return y
