import argparse
import os
import sys
import numpy as np

# ---------------- Core normalization ----------------
def process_data(data: np.ndarray, *, axis: int | None, nan_policy: str = "raise") -> np.ndarray:
    """
    Normalize data to zero mean and unit variance.

    Parameters
    ----------
    data : np.ndarray
        1D or 2D array.
    axis : int | None
        If data is 2D, axis=0 normalizes columns, axis=1 normalizes rows.
        If data is 1D, pass None.
    nan_policy : {'raise','ignore'}
        'raise'  -> error if NaNs exist.
        'ignore' -> compute mean/std with NaNs ignored.

    Returns
    -------
    np.ndarray
        Normalized array.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if data.ndim not in (1, 2):
        raise ValueError("Only 1D or 2D numpy arrays are supported.")

    has_nan = np.isnan(data).any()
    if has_nan and nan_policy == "raise":
        raise ValueError("Input contains NaNs. Use --nan-policy ignore to skip NaNs.")

    mean_fn = np.mean if not has_nan or nan_policy == "raise" else np.nanmean
    std_fn  = np.std  if not has_nan or nan_policy == "raise" else np.nanstd

    if data.ndim == 1:
        mean = mean_fn(data)
        std  = std_fn(data)
        if not np.isfinite(std) or std == 0:
            raise ValueError("Standard deviation is zero or non-finite; cannot normalize.")
        return (data - mean) / std

    # 2D case
    if axis not in (0, 1):
        raise ValueError("For 2D arrays, axis must be 0 (columns) or 1 (rows).")

    mean = mean_fn(data, axis=axis, keepdims=True)
    std  = std_fn(data,  axis=axis, keepdims=True)

    bad = (~np.isfinite(std)) | (std == 0)
    if bad.any():
        if axis == 0:
            bad_idxs = np.where(bad[0])[0].tolist()
            loc = "column(s)"
        else:
            bad_idxs = np.where(bad[:, 0])[0].tolist()
            loc = "row(s)"
        raise ValueError(f"Zero or non-finite std in {loc}: {bad_idxs}; cannot normalize.")

    return (data - mean) / std

# ---------------- Loading helpers ----------------
def _load_with_pandas(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    num_df = df.select_dtypes(include=[np.number])
    nonnum_cols = [c for c in cols if c not in num_df.columns]
    return df, num_df, nonnum_cols

def _gen_numeric_matrix_from_struct(arr: np.ndarray) -> np.ndarray:
    numeric_fields = [
        name for name, (dt, _) in arr.dtype.fields.items()
        if dt.kind in ("f", "i", "u")
    ]
    if not numeric_fields:
        raise ValueError("No numeric columns found in CSV to normalize.")
    return np.vstack([arr[name] for name in numeric_fields]).T, numeric_fields

def load_array_with_metadata(path: str):
    """
    Returns:
        matrix: np.ndarray
        meta: dict with keys:
            - 'pandas': bool
            - 'all_columns': list[str] or None
            - 'numeric_columns': list[str] or None
            - 'non_numeric_columns': list[str] or None
            - 'non_numeric_df': pandas.DataFrame or None
    """
    ext = os.path.splitext(path)[1].lower()
    meta = {
        "pandas": False,
        "all_columns": None,
        "numeric_columns": None,
        "non_numeric_columns": None,
        "non_numeric_df": None,
    }

    if ext == ".npy":
        return np.load(path), meta

    if ext == ".npz":
        npz = np.load(path)
        if len(npz.files) == 0:
            raise ValueError("NPZ file is empty.")
        return npz[npz.files[0]], meta

    if ext in (".csv", ".txt"):
        try:
            import pandas as pd  # noqa
            meta["pandas"] = True
            df, num_df, nonnum_cols = _load_with_pandas(path)
            if num_df.empty:
                df2 = df.drop(columns=df.columns[0], errors="ignore")
                num_df = df2.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
                nonnum_cols = [c for c in df.columns if c not in num_df.columns]

            if num_df.empty:
                raise ValueError("No numeric columns found in CSV to normalize.")

            meta["all_columns"] = df.columns.tolist()
            meta["numeric_columns"] = num_df.columns.tolist()
            meta["non_numeric_columns"] = nonnum_cols
            meta["non_numeric_df"] = df[nonnum_cols] if nonnum_cols else None
            return num_df.to_numpy(), meta

        except ImportError:
            arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
            if arr.size == 0:
                raise ValueError("CSV appears empty or unreadable.")
            matrix, num_cols = _gen_numeric_matrix_from_struct(arr)
            meta["numeric_columns"] = num_cols
            return matrix, meta

    raise ValueError(f"Unsupported file extension: {ext}. Use .npy, .npz, .csv, or .txt")

def infer_output_path(input_path: str, out: str | None, suffix: str, new_ext: str | None = None) -> str:
    if out:
        base, _ = os.path.splitext(out)
        return f"{base}{new_ext or ''}"
    base, ext = os.path.splitext(input_path)
    if new_ext is None:
        return f"{base}_{suffix}{ext}"
    return f"{base}_{suffix}{new_ext}"

# ---------------- Main CLI ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Normalize a 1D/2D array file (zero mean, unit variance). Writes .npy and optional .csv."
    )
    parser.add_argument("input", help="Path to .npy, .npz, .csv, or .txt data file")
    parser.add_argument("-o", "--out", help="Output path for the .npy file. Default: <input>_normalized.npy")
    parser.add_argument(
        "--axis", choices=["columns", "rows", "auto"], default="auto",
        help="For 2D data: normalize by columns or rows. 'auto' -> columns. Ignored for 1D."
    )
    parser.add_argument(
        "--nan-policy", choices=["raise", "ignore"], default="raise",
        help="How to handle NaNs: 'raise' (error) or 'ignore' (skip NaNs in mean/std)."
    )
    parser.add_argument(
        "--emit-csv", action="store_true",
        help="Also write a normalized CSV (preserving non-numeric columns when pandas is available)."
    )
    parser.add_argument(
        "--csv-out", help="Optional explicit path for the CSV output. Default: <input>_normalized.csv"
    )
    args = parser.parse_args()

    # Load
    try:
        data, meta = load_array_with_metadata(args.input)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    if data.ndim not in (1, 2):
        print("Data must be a 1D or 2D numpy array.", file=sys.stderr)
        sys.exit(1)

    axis = None
    if data.ndim == 2:
        axis = 0 if args.axis in ("auto", "columns") else 1

    # Normalize
    try:
        norm = process_data(data, axis=axis, nan_policy=args.nan_policy)
    except Exception as e:
        print(f"Error normalizing data: {e}", file=sys.stderr)
        sys.exit(1)

    # Save NPY
    npy_out = infer_output_path(args.input, args.out, "normalized", new_ext=".npy")
    try:
        np.save(npy_out, norm)
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)
        sys.exit(1)

    # Optionally save CSV
    csv_written = None
    if args.emit_csv:
        ext = os.path.splitext(args.input)[1].lower()
        csv_out = args.csv_out or infer_output_path(args.input, None, "normalized", new_ext=".csv")

        if meta.get("pandas"):
            import pandas as pd
            all_cols = meta["all_columns"]
            num_cols = meta["numeric_columns"] or []
            non_cols = meta["non_numeric_columns"] or []
            non_df  = meta["non_numeric_df"]

            num_df_norm = pd.DataFrame(norm, columns=num_cols)
            if non_df is not None and not non_df.empty:
                parts = {c: (non_df[c] if c in non_cols else num_df_norm[c]) for c in all_cols if c in (non_cols + num_cols)}
                out_df = pd.DataFrame(parts)
            else:
                out_df = num_df_norm

            out_df.to_csv(csv_out, index=False)
            csv_written = csv_out
        else:
            try:
                cols = meta.get("numeric_columns")
                header = ",".join(cols) if cols else None
            except Exception:
                header = None
            np.savetxt(csv_out, norm, delimiter=",", header=header or "", comments="")
            csv_written = csv_out

    # Quick summary
    if norm.ndim == 1:
        mu = float(np.nanmean(norm))
        sd = float(np.nanstd(norm))
        shape = norm.shape
    else:
        mu = np.nanmean(norm, axis=0)
        sd = np.nanstd(norm, axis=0)
        shape = norm.shape

    print("Normalization complete")
    print(f"   Input shape: {data.shape}")
    print(f"   Output shape: {shape}")
    print(f"   Saved NPY: {npy_out}")
    if csv_written:
        print(f"   Saved CSV: {csv_written}")
    print(f"   Mean (post-normalization): {mu}")
    print(f"   Std  (post-normalization): {sd}")

if __name__ == "__main__":
    main()
