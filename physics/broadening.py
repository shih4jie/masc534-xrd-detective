import numpy as np
from .units import positive, theta_rad


def _widths(sample, instrument, profile):
    if profile not in ("Gaussian", "Lorentzian"):
        raise ValueError("Only pure Gaussian or Lorentzian FWHM convolution is supported.")
    a = positive(sample, "Sample/observed FWHM")
    b = np.asarray(instrument, dtype=float)
    if not np.all(np.isfinite(b)) or np.any(b < 0):
        raise ValueError("Instrument FWHM must be finite and nonnegative.")
    return a, b


def combine_fwhm(sample, instrument, profile="Gaussian"):
    a, b = _widths(sample, instrument, profile)
    return np.hypot(a, b) if profile == "Gaussian" else a + b


def correct_fwhm(observed, instrument, profile="Gaussian"):
    a, b = _widths(observed, instrument, profile)
    if np.any(a <= b):
        raise ValueError("Observed FWHM must exceed instrumental FWHM; size is unresolved.")
    return np.sqrt(a * a - b * b) if profile == "Gaussian" else a - b


def strain_fwhm_deg(two_theta_deg, microstrain):
    e = np.asarray(microstrain, dtype=float)
    if not np.all(np.isfinite(e)) or np.any(e < 0):
        raise ValueError("Microstrain must be finite and nonnegative.")
    return np.rad2deg(4 * e * np.tan(theta_rad(two_theta_deg)))
