import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve


def estimate_baseline(y, smoothness=1e5, asymmetry=0.01, iterations=10):
    """Asymmetric least squares on sample indices, for display/detection only.

    A local linear baseline is jointly fitted for all reported widths.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < 12 or not np.isfinite(y).all():
        raise ValueError("Baseline needs at least 12 finite intensities.")
    if not np.isfinite(smoothness) or smoothness <= 0 or not np.isfinite(asymmetry) or not 0 < asymmetry < 1:
        raise ValueError("Baseline smoothness must be positive and asymmetry must lie between 0 and 1.")
    if not isinstance(iterations, (int, np.integer)) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("Baseline iterations must be a positive integer.")
    d = sparse.diags([1., -2., 1.], [0, 1, 2], shape=(len(y)-2, len(y)), format="csc")
    w = np.ones(len(y))
    penalty = smoothness * (d.T @ d)
    for _ in range(iterations):
        z = spsolve(sparse.diags(w, format="csc") + penalty, w * y)
        w = np.where(y > z, asymmetry, 1-asymmetry)
    return z


def detect_peaks(x, y, prominence_fraction=0.1, *, smoothing=False,
                 smoothing_window=11, min_separation_deg=None, max_peaks=30):
    """Return prominence-ranked candidate positions and the detection baseline.

    Baseline removal and optional Savitzky-Golay smoothing are for detection only;
    supplied intensities and coordinates remain unchanged for scientific fitting.
    Smoothing uses neighboring sample indices, also for nonuniform sampling. An
    explicitly supplied separation is enforced on the actual angular positions,
    with the more prominent maximum retained. No resampling is performed.
    Candidate detection alone does not establish a valid physical reflection.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 12 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Peak detection needs at least 12 finite intensities on an increasing angular axis.")
    if not np.isfinite(prominence_fraction) or not 0 < prominence_fraction <= 1:
        raise ValueError("Detection prominence fraction must be greater than 0 and at most 1.")
    if min_separation_deg is not None and (not np.isfinite(min_separation_deg) or min_separation_deg < 0):
        raise ValueError("Minimum peak separation must be finite and nonnegative in 2theta degrees.")
    if not isinstance(max_peaks, (int, np.integer)) or isinstance(max_peaks, bool) or max_peaks < 1:
        raise ValueError("Maximum candidate count must be a positive integer.")
    base = estimate_baseline(y)
    signal = y - base
    if smoothing:
        if (not isinstance(smoothing_window, (int, np.integer)) or isinstance(smoothing_window, bool)
                or smoothing_window < 3 or smoothing_window % 2 != 1 or smoothing_window > len(signal)):
            raise ValueError("Detection smoothing window must be an odd integer from 3 to the number of samples.")
        signal = savgol_filter(signal, smoothing_window, 2)
    # Keep legacy five-sample spacing when no explicit angular spacing is given.
    idx, props = find_peaks(signal, prominence=max(np.ptp(signal)*prominence_fraction, 1e-12),
                            distance=5 if min_separation_deg is None else 1)
    order = np.argsort(props["prominences"])[::-1]
    selected = []
    for ranked in order:
        candidate = int(idx[ranked])
        if min_separation_deg is None or all(abs(x[candidate] - x[other]) >= min_separation_deg for other in selected):
            selected.append(candidate)
        if len(selected) == max_peaks:
            break
    return x[np.asarray(selected, dtype=int)], base
