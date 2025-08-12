import argparse
import os
import sys
import numpy as np


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
        'ignore' -> compute mean/std with NaNs ignored (np.nanmean/std). Columns/rows
                    with std==0 or all-NaN will raise.

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
    std_fn = np.std if not has_nan or nan_policy == "raise" else np.nanstd

    if data.ndim == 1:
        mean = mean_fn(data)
        std = std_fn(data)
        if not np.isfinite(std) or std == 0:
            raise ValueError("Standard deviation is zero or non-finite; cannot normalize.")
        return (data - mean) / std

    # 2D case
    if axis not in (0, 1):
        raise ValueError("For 2D arrays, axis must be 0 (columns) or 1 (rows).")

    mean = mean_fn(data, axis=axis, keepdims=True)
    std = std_fn(data, axis=axis, keepdims=True)

    # Identify bad std (zero or non-finite)
    bad = (~np.isfinite(std)) | (std == 0)
    if bad.any():
        # Figure out which cols/rows are bad for a clearer error
        if axis == 0:
            bad_idxs = np.where(bad[0])[0].tolist()
            loc = "column(s)"
        else:
            bad_idxs = np.where(bad[:, 0])[0].tolist()
            loc = "row(s)"
        raise ValueError(f"Zero or non-finite std in {loc}: {bad_idxs}; cannot normalize.")

    return (data - mean) / std


def load_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        npz = np.load(path)
        # Pick the first array in the archive
        if len(npz.files) == 0:
            raise ValueError("NPZ file is empty.")
        return npz[npz.files[0]]
    if ext in (".csv", ".txt"):
        # Try comma first; if it fails, fallback to whitespace
        try:
            return np.loadtxt(path, delimiter=",")
        except Exception:
            return np.loadtxt(path)
    raise ValueError(f"Unsupported file extension: {ext}. Use .npy, .npz, .csv, or .txt")


def infer_output_path(input_path: str, out: str | None) -> str:
    if out:
        return out
    base, _ = os.path.splitext(input_path)
    return f"{base}_normalized.npy"


def main():
    parser = argparse.ArgumentParser(
        description="Normalize a 1D/2D numpy array file (zero mean, unit variance)."
    )
    parser.add_argument("input", help="Path to .npy, .npz, .csv, or .txt data file")
    parser.add_argument("-o", "--out", help="Output path (.npy). Default: <input>_normalized.npy")
    parser.add_argument(
        "--axis",
        choices=["columns", "rows", "auto"],
        default="auto",
        help="For 2D data: normalize by columns or rows. 'auto' -> columns. Ignored for 1D."
    )
    parser.add_argument(
        "--nan-policy",
        choices=["raise", "ignore"],
        default="raise",
        help="How to handle NaNs: 'raise' (error) or 'ignore' (skip NaNs in mean/std)."
    )
    args = parser.parse_args()

    try:
        data = load_array(args.input)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    if data.ndim not in (1, 2):
        print("Data must be a 1D or 2D numpy array.", file=sys.stderr)
        sys.exit(1)

    axis = None
    if data.ndim == 2:
        axis = 0 if args.axis in ("auto", "columns") else 1

    try:
        norm = process_data(data, axis=axis, nan_policy=args.nan_policy)
    except Exception as e:
        print(f"Error normalizing data: {e}", file=sys.stderr)
        sys.exit(1)

    out_path = infer_output_path(args.input, args.out)
    try:
        np.save(out_path, norm)
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)
        sys.exit(1)

    # Quick summary
    if norm.ndim == 1:
        mu = float(np.nanmean(norm))
        sd = float(np.nanstd(norm))
        shape = norm.shape
    else:
        mu = np.nanmean(norm, axis=0)
        sd = np.nanstd(norm, axis=0)
        shape = norm.shape

    print("✅ Normalization complete")
    print(f"   Input shape: {data.shape}")
    print(f"   Output shape: {shape}")
    print(f"   Saved: {out_path}")
    print(f"   Mean (post-normalization): {mu}")
    print(f"   Std  (post-normalization): {sd}")


if __name__ == "__main__":
    main()
