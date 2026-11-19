import numpy as np
from scipy.stats import linregress
from .units import positive, theta_rad


def williamson_hall(two_theta_deg, sample_fwhm_deg, wavelength_nm=0.15406,
                    shape_factor=0.9, profile="Lorentzian"):
    """Uniform-deformation model using additive Lorentzian FWHM only.

    Instrument correction and single-phase/reflection identification are caller
    responsibilities. Gaussian quadrature does NOT obey this linear equation.
    """
    if profile != "Lorentzian":
        raise ValueError("Linear W-H here requires additive Lorentzian breadths.")
    a = np.asarray(two_theta_deg, dtype=float)
    b = positive(sample_fwhm_deg, "Sample FWHM")
    if a.ndim != 1 or a.shape != b.shape or len(a) < 3:
        raise ValueError("At least three usable, distinct reflections are required.")
    t = theta_rad(a)
    x, y = 4 * np.sin(t), np.deg2rad(b) * np.cos(t)
    if len(np.unique(a)) != len(a) or np.ptp(x) < 0.1:
        raise ValueError("Reflections need distinct angles with a wider angular span.")
    fit = linregress(x, y)
    intercept_roundoff = 32 * np.finfo(float).eps * max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    if fit.intercept <= intercept_roundoff or not np.isfinite(fit.intercept):
        raise ValueError("Nonpositive or numerically unresolved W-H intercept: no finite positive size justified.")
    if fit.slope < -1e-12:
        raise ValueError("Negative microstrain slope is inconsistent with this model.")
    warnings = []
    if fit.rvalue ** 2 < 0.95:
        warnings.append("Poor W-H linearity: isotropic uniform-deformation assumptions may fail.")
    fitted = fit.intercept + fit.slope * x
    residuals = y - fitted
    # Regression scatter under independent-reflection assumptions is not a
    # calibrated uncertainty interval for the specimen's size or microstrain.
    residual_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    leverage = 1 / len(x) + (x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2)
    intercept_stderr = float(fit.intercept_stderr)
    slope_stderr = float(fit.stderr)
    if len(x) == 3:
        warnings.append("Only three reflections: one residual degree of freedom gives limited regression diagnostics.")
    if np.max(leverage) > .85:
        warnings.append("W-H fit is sensitive to a high-leverage reflection; inspect angular distribution and residuals.")
    if np.ptp(x) < .5:
        warnings.append("Limited W-H angular leverage: the size intercept is extrapolated from a narrow range.")
    if np.isfinite(intercept_stderr) and 2 * intercept_stderr >= fit.intercept:
        warnings.append("Weakly constrained W-H intercept: regression scatter is large relative to the positive intercept; the size is highly sensitive.")
    diagnostics = {"intercept_stderr": intercept_stderr,
                   "slope_stderr": slope_stderr,
                   "residual_rmse": residual_rmse,
                   "degrees_of_freedom": int(len(x) - 2),
                   "angular_span_deg": float(np.ptp(a)),
                   "x_span": float(np.ptp(x)),
                   "max_leverage": float(np.max(leverage)),
                   "intercept_extrapolation_ratio": float(np.min(x) / np.ptp(x)),
                   "uncertainty_scope": "Regression diagnostics only; not calibrated measurement uncertainty."}
    weak_intercept = np.isfinite(intercept_stderr) and 2 * intercept_stderr >= fit.intercept
    dominant_leverage = diagnostics["max_leverage"] > .98
    distant_extrapolation = diagnostics["intercept_extrapolation_ratio"] > 5
    if dominant_leverage:
        warnings.append("One reflection dominates angular leverage (>0.98); obtain additional reflections distributed across the angle range before reporting W-H size.")
    if distant_extrapolation:
        warnings.append("Reflections occupy a narrow high-angle band: the zero-angle intercept is a distant extrapolation (>5 times the measured x-span); obtain a wider angular distribution.")
    size_justified = not (weak_intercept or dominant_leverage or distant_extrapolation)
    conditional_size = float(positive(shape_factor, "K") * positive(wavelength_nm, "Wavelength") / fit.intercept)
    diagnostics.update(weak_intercept=bool(weak_intercept), dominant_leverage=bool(dominant_leverage),
                       distant_extrapolation=bool(distant_extrapolation))
    return {"size_nm": conditional_size if size_justified else None,
            "conditional_size_nm": conditional_size,
            "size_justified": bool(size_justified),
            "microstrain": float(max(0., fit.slope)), "intercept": float(fit.intercept),
            "r_squared": float(fit.rvalue ** 2), "x": x, "y": y,
            "fitted": fitted, "residuals": residuals,
            "n_reflections": int(len(x)), "diagnostics": diagnostics,
            "warnings": warnings}
