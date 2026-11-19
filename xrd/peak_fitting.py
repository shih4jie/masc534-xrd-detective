from dataclasses import dataclass, asdict
import warnings
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import find_peaks, savgol_filter
from physics.simulation import peak_profile


@dataclass
class PeakFit:
    center_deg: float
    fwhm_deg: float
    amplitude: float
    r_squared: float
    normalized_rmse: float
    profile: str
    usable: bool
    warnings: list[str]
    x: np.ndarray
    observed: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray

    def summary(self):
        return {k: v for k, v in asdict(self).items() if k not in {"x", "observed", "fitted", "residual"}}


def fit_peak(x, y, profile="Gaussian"):
    """One peak + local linear baseline, conservative rejection diagnostics."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 12 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Fit needs at least 12 finite points on an increasing axis.")
    if np.ptp(y) <= 0:
        raise ValueError("No peak signal in selected window.")
    span, dx = np.ptp(x), np.median(np.diff(x))
    edge_n = max(2, len(x)//10)
    # Initialize baseline from edges, then fit it jointly with the line shape.
    slope = (np.median(y[-edge_n:]) - np.median(y[:edge_n])) / span
    mid = (x[0] + x[-1])/2
    base = np.median(np.r_[y[:edge_n]-slope*(x[:edge_n]-mid), y[-edge_n:]-slope*(x[-edge_n:]-mid)])
    signal = y - base - slope*(x-mid)
    center = x[np.argmax(signal)]
    amplitude = float(np.max(signal))
    if amplitude <= 0:
        raise ValueError("No positive isolated peak found.")
    above = x[signal > amplitude/2]
    width0 = max(3*dx, np.ptp(above)) if len(above)>1 else 3*dx
    def model(t, amp, cen, width, b, m):
        return b + m*(t-mid) + peak_profile(t, cen, width, amp, profile)
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        try:
            pars, cov = curve_fit(model, x, y, p0=[amplitude, center, min(width0,span/2), base,slope],
                                  bounds=([0,x[0],dx/2,-np.inf,-np.inf], [np.inf,x[-1],span,np.inf,np.inf]), maxfev=10000)
        except (RuntimeError, OptimizeWarning) as exc:
            raise ValueError(f"Peak fit did not converge reliably: {exc}") from exc
    amp, cen, width, _, _ = pars
    fitted = model(x, *pars)
    residual = y-fitted
    rss = np.sum(residual**2)
    r2 = 1 - rss/np.sum((y-y.mean())**2)
    nrmse = np.sqrt(np.mean(residual**2))/amp
    flags=[]
    if r2 < .98 or nrmse > .05:
        flags.append("Poor peak fit: inspect residuals, background and profile assumptions.")
    if width/dx < 5:
        flags.append("Under-resolved peak: fewer than five median sample steps per FWHM.")
    if min(cen-x[0], x[-1]-cen) < 2*width:
        flags.append("Truncated fit window: include at least two FWHM on each side.")
    window = min(21, len(signal) if len(signal)%2 else len(signal)-1)
    smooth = savgol_filter(signal, window, 2)
    peaks, _ = find_peaks(smooth, prominence=.15*amp, distance=max(1,int(width/dx/2)))
    if len(peaks) > 1:
        flags.append("Possible overlapping reflections: a single-peak model is unsupported.")
    # Raw covariance conditioning depends on parameter units. Use correlation;
    # an exact noiseless fit legitimately has zero estimated residual variance.
    variances = np.diag(cov)
    noisy = rss > np.finfo(float).eps * np.sum((y-y.mean())**2)
    bad_covariance = not np.isfinite(cov).all()
    if noisy and not bad_covariance:
        if np.any(variances <= 0):
            bad_covariance = True
        else:
            correlation = cov / np.sqrt(np.outer(variances, variances))
            bad_covariance = np.linalg.cond(correlation) > 1e10
    if bad_covariance:
        flags.append("Ill-conditioned fit parameters: width is not sufficiently identified.")
    return PeakFit(float(cen),float(width),float(amp),float(r2),float(nrmse),profile,
                   not flags,flags,x,y,fitted,residual)
