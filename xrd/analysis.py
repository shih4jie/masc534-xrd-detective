import numpy as np
from physics.scherrer import scherrer_size_nm
from physics.broadening import correct_fwhm
from physics.units import positive
from .peak_fitting import fit_peak


def analyze_window(pattern, lower, upper, wavelength_nm, profile="Gaussian",
                   instrument_deg=None, shape_factor=.9, size_dominated=False,
                   isolated_confirmed=False, instrument_calibrated=False):
    positive(wavelength_nm, "Wavelength (nm)")
    if not isolated_confirmed:
        raise ValueError("Confirm an isolated reflection with no unresolved spectral doublet before size analysis.")
    mask = (pattern.two_theta_deg >= lower) & (pattern.two_theta_deg <= upper)
    fit = fit_peak(pattern.two_theta_deg[mask],pattern.intensity[mask],profile)
    notes = list(pattern.warnings) + fit.warnings
    result = {"category":"Insufficient information", "size_nm":None,"fit":fit,
              "warnings":notes,"breadth_definition":"FWHM on 2theta axis",
              "instrument_corrected":False,"sample_fwhm_deg":None}
    if not fit.usable:
        result["category"]="Unsupported by current model"
        return result
    beta = fit.fwhm_deg
    if instrument_deg is not None:
        beta = float(correct_fwhm(beta, instrument_deg,profile))
        result["instrument_corrected"] = True
        if not instrument_calibrated:
            notes.append("Instrument width is an assumed constant, not established by a calibration.")
        if beta/fit.fwhm_deg < .3:
            notes.append("Instrument-dominated width: correction is highly sensitive to calibration error.")
    else:
        notes.append("No instrument calibration: only an apparent uncorrected size can be reported.")
    result["sample_fwhm_deg"] = beta
    if not size_dominated:
        notes.append("Size/strain separation is not established; confirm size-dominated broadening to obtain a conditional Scherrer estimate.")
        return result
    result["size_nm"] = float(scherrer_size_nm(fit.center_deg,beta,wavelength_nm,shape_factor))
    result["category"] = "Supported under stated assumptions" if instrument_deg is not None and instrument_calibrated and not notes else "Preliminary estimate"
    notes.append("Conditional coherent-domain size, not grain size; K and pure-profile assumptions require physical justification.")
    return result
