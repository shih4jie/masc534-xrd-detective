"""Defined, isolated reflections; no invented crystal structure or intensities."""
import numpy as np
from .scherrer import size_fwhm_deg
from .broadening import combine_fwhm, strain_fwhm_deg
from .units import positive

PRESETS = {
    "Size dominated": (20.0, 0.0001, 0.03),
    "Strain dominated": (200.0, 0.003, 0.03),
    "Instrument dominated": (200.0, 0.0001, 0.4),
}


def peak_profile(x, center, fwhm, amplitude=1.0, profile="Gaussian"):
    positive(fwhm, "FWHM")
    z = (np.asarray(x) - center) / fwhm
    if profile == "Gaussian":
        return amplitude * np.exp(-4 * np.log(2) * z ** 2)
    if profile == "Lorentzian":
        return amplitude / (1 + 4 * z ** 2)
    raise ValueError("Unsupported profile.")


def simulate_peak(center=40.0, size_nm=40.0, microstrain=0.0,
                  instrument_deg=0.0, wavelength_nm=0.15406,
                  noise=0.0, profile="Gaussian", seed=534, x=None,
                  amplitude=1000.0, background=20.0, shape_factor=0.9):
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("Noise must be finite and nonnegative.")
    positive(amplitude, "Amplitude")
    bs = float(size_fwhm_deg(center, size_nm, wavelength_nm, shape_factor))
    be = float(strain_fwhm_deg(center, microstrain))
    # Gaussian: independent variances add. Lorentzian: Cauchy widths add.
    sample = float(combine_fwhm(bs, be, profile))
    obs = float(combine_fwhm(sample, instrument_deg, profile))
    if x is None:
        half_range = max(1.5, 8 * obs)
        x = np.linspace(max(0.001, center - half_range), min(179.999, center + half_range), 2001)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 12 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Simulation axis must be finite and increasing with at least 12 points.")
    clean = background + peak_profile(x, center, obs, amplitude, profile)
    y = clean + np.random.default_rng(seed).normal(0, noise * amplitude, len(x))
    return {"two_theta_deg": x, "intensity": y, "clean": clean,
            "size_fwhm_deg": bs, "strain_fwhm_deg": be,
            "instrument_fwhm_deg": float(instrument_deg), "observed_fwhm_deg": obs,
            "profile": profile, "seed": seed}
