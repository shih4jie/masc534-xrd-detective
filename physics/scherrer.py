import numpy as np
from .units import positive, theta_rad


def scherrer_size_nm(two_theta_deg, fwhm_deg, wavelength_nm=0.15406, shape_factor=0.9):
    """D = K lambda / (beta cos(theta)); FWHM is measured on the 2theta axis.

    Caller must establish size-only breadth (and appropriate K) before treating
    this as a physical coherent-domain-size measurement.
    """
    beta = np.deg2rad(positive(fwhm_deg, "2theta FWHM (degrees)"))
    return positive(shape_factor, "Shape factor") * positive(wavelength_nm, "Wavelength (nm)") / (beta * np.cos(theta_rad(two_theta_deg)))


def size_fwhm_deg(two_theta_deg, size_nm, wavelength_nm=0.15406, shape_factor=0.9):
    return np.rad2deg(positive(shape_factor, "Shape factor") * positive(wavelength_nm, "Wavelength (nm)") / (positive(size_nm, "Size (nm)") * np.cos(theta_rad(two_theta_deg))))
