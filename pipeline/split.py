def time_split(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    i1 = int(n*train_ratio)
    i2 = int(n*(train_ratio+val_ratio))
    return slice(0,i1), slice(i1,i2), slice(i2,n)
