import numpy as np


def original_input_tensor(intensities):
    """Original notebook preprocessing: add axes ONLY; log10 is in the model."""
    a = np.asarray(intensities)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 10501 or len(a) == 0:
        raise ValueError("Original LPA-NN input requires (N,10501) intensities.")
    if not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("LPA-NN log10 layer requires strictly positive finite intensities.")
    # Positive float64 data can overflow or underflow when cast for the model.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        tensor = a.astype(np.float32)
    if not np.isfinite(tensor).all() or np.any(tensor <= 0):
        raise ValueError("Intensities must remain finite and strictly positive in float32; no clipping or rescaling is permitted.")
    return tensor[:, :, None, None]


def denormalize_minmax(predictions, minimum, maximum):
    p = np.array(predictions, dtype=float, copy=True)
    lo, hi = np.asarray(minimum), np.asarray(maximum)
    if p.ndim != 2 or lo.shape != (p.shape[1],) or hi.shape != lo.shape:
        raise ValueError("Prediction and target normalization shapes disagree.")
    if not np.isfinite(p).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
        raise ValueError("Invalid prediction or normalization range.")
    return p * (hi-lo) + lo
