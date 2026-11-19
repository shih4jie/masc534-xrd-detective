import numpy as np


def positive(value, name):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f"{name} must be finite and positive.")
    return a


def theta_rad(two_theta_deg):
    a = positive(two_theta_deg, "2theta")
    if np.any(a >= 180):
        raise ValueError("2theta must be between 0 and 180 degrees.")
    return np.deg2rad(a / 2)


def wavelength_nm(value, unit="nm"):
    factors = {"nm": 1.0, "angstrom": 0.1, "pm": 0.001}
    if unit not in factors:
        raise ValueError("Use nm, angstrom, or pm for wavelength.")
    return positive(value, "Wavelength") * factors[unit]


def bragg_spacing_nm(two_theta_deg, wavelength, order=1):
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("Diffraction order must be a positive integer.")
    return order * positive(wavelength, "Wavelength (nm)") / (2 * np.sin(theta_rad(two_theta_deg)))
