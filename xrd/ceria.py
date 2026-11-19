"""Bounded comparison for the public ESRF BM16 ceria round-robin pair."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, curve_fit
from scipy.special import voigt_profile

from physics.scherrer import scherrer_size_nm

ASSETS = Path(__file__).resolve().parents[1] / "examples" / "ceria_bm16"
SAMPLE_NAME = "massonbr.xy"
REFERENCE_NAME = "massonsh.xy"
SOURCE = "https://mysite.du.edu/~balzar/"
SAMPLE_SHA256 = "93c62b6670ee3fdf1558e50fb1926f9bf5ef139b9756d040586b1bff6a6bbaa8"
REFERENCE_SHA256 = "9d53c1d4bc45b46f7a422c81560c74afabb0402d62bd80c1bf915b9380d48ac3"
CIF_SHA256 = "85bbb22414b3542647a683c144f3011e217fd49ec1edf0e82f4d1f3f7c515827"
WAVELENGTH_ANGSTROM = 0.39982
GEOMETRY = "Powder capillary scan (analyzer detector)"
FIT_LIMITS_DEG = (6.7, 28.0)
# First four allowed fluorite reflections by increasing angle, isolated in these windows.
REFLECTIONS = (("111", 7.337), ("200", 8.474), ("220", 11.996), ("311", 14.076))


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def is_sample(content):
    return sha256(content) == SAMPLE_SHA256


def read_measured(content, expected_sha256):
    if sha256(content) != expected_sha256:
        raise ValueError("The file differs from the published ESRF BM16 data. Download the original .xy file.")
    # The source provides 2theta in degrees, intensity and one-sigma counting errors.
    return np.loadtxt(io.BytesIO(content), skiprows=2)


def _peak(x, offset, height, sigma, gamma, baseline, slope):
    return (height * voigt_profile(x - offset, sigma, gamma) / voigt_profile(0, sigma, gamma)
            + baseline + slope * x)


def fit_reflection(table, nominal_center, window, reference):
    selected = table[np.abs(table[:, 0] - nominal_center) <= window]
    x, y, y_sigma = selected.T
    delta = x - nominal_center
    height = float(np.ptp(y))
    guesses = (.002, .001) if reference else (.02, .02)
    width_min = np.median(np.diff(x)) / 50
    params, _ = curve_fit(
        _peak, delta, y, p0=(delta[np.argmax(y)], height, *guesses, np.min(y), 0),
        bounds=([-window / 2, 0, width_min, width_min, -np.inf, -np.inf],
                [window / 2, np.inf, window / 2, window / 2, np.inf, np.inf]),
        sigma=y_sigma, absolute_sigma=True, maxfev=2000,
    )
    offset, _, sigma, gamma, _, _ = params
    residual = y - _peak(delta, *params)
    return {"center_deg": float(nominal_center + offset), "sigma_deg": float(sigma),
            "gamma_deg": float(gamma), "weighted_residual_rms": float(np.sqrt(np.mean((residual / y_sigma) ** 2))),
            "points": len(x)}


def local_reference_fits(table):
    return {hkl: fit_reflection(table, center, .035, reference=True) for hkl, center in REFLECTIONS}


def _voigt_fwhm(sigma, gamma):
    half_height = voigt_profile(0, sigma, gamma) / 2
    return 2 * brentq(lambda x: voigt_profile(x, sigma, gamma) - half_height, 0, 5 * (sigma + gamma))


def physics_comparison(content, reference_fits):
    table = read_measured(content, SAMPLE_SHA256)
    results = []
    for hkl, center in REFLECTIONS:
        local = fit_reflection(table, center, .27, reference=False)
        reference = reference_fits[hkl]
        variance = local["sigma_deg"] ** 2 - reference["sigma_deg"] ** 2
        gamma = local["gamma_deg"] - reference["gamma_deg"]
        if variance <= 0 or gamma <= 0:
            raise ValueError(f"The sample broadening of reflection ({hkl}) could not be resolved above the reference.")
        fwhm = _voigt_fwhm(float(np.sqrt(variance)), gamma)
        size_nm = scherrer_size_nm(local["center_deg"], fwhm, WAVELENGTH_ANGSTROM / 10, 1)
        results.append({"reflection": hkl, "observed": local, "reference": reference,
                        "corrected_fwhm_deg": fwhm, "size_nm": float(size_nm)})
    size_nm = float(np.median([row["size_nm"] for row in results]))
    return {"size_nm": size_nm, "reflections": results}


def comparison_difference(physics_nm, gsas_nm):
    """Absolute difference and a symmetric percentage, using full precision."""
    if not all(np.isfinite(n) and n > 0 for n in (physics_nm, gsas_nm)):
        raise ValueError("Two positive, finite coherent-domain sizes are required.")
    difference_nm = abs(physics_nm - gsas_nm)
    return difference_nm, 100 * difference_nm / (physics_nm / 2 + gsas_nm / 2)


def load_evidence():
    evidence = json.loads((ASSETS / "refinement.json").read_text())
    if (evidence["sample_sha256"] != SAMPLE_SHA256 or evidence["reference_sha256"] != REFERENCE_SHA256
            or sha256((ASSETS / "9009008.cif").read_bytes()) != CIF_SHA256
            or sha256((ASSETS / "bm16.instprm").read_bytes()) != evidence["instprm_sha256"]):
        raise ValueError("The comparison assets do not match the recorded calibration. Restore the release files.")
    return evidence
