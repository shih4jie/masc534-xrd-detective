"""Conservative automatic local-peak analysis using the existing physics engine.

Detected maxima are candidates, not crystallographic phase identifications.
Numerical fit diagnostics and explicit isolation/phase confirmations are separate.
"""
import numpy as np

from physics.broadening import correct_fwhm
from physics.scherrer import scherrer_size_nm
from physics.units import positive
from physics.williamson_hall import williamson_hall
from .peak_fitting import fit_peak


def validate_instrument(instrument, profile):
    """Validate a local pure-profile calibration, never extrapolating a table."""
    info = dict(instrument or {"mode": "unknown"})
    mode = info.get("mode", "unknown")
    if mode not in {"unknown", "constant", "table"}:
        raise ValueError("Instrument mode must be unknown, constant, or table.")
    if mode == "unknown":
        return {"mode": "unknown", "calibrated": False, "reference": ""}
    if info.get("profile") != profile:
        raise ValueError("Instrument calibration profile must match the selected pure peak profile; Gaussian and Lorentzian widths are not interchangeable.")
    if mode == "constant":
        width = float(info.get("fwhm_deg", np.nan))
        if not np.isfinite(width) or width < 0:
            raise ValueError("Instrument FWHM must be finite and nonnegative, in degrees on 2theta.")
        info["fwhm_deg"] = width
    else:
        angles = np.asarray(info.get("angles_deg", []), dtype=float)
        widths = np.asarray(info.get("widths_deg", []), dtype=float)
        if (angles.ndim != 1 or widths.shape != angles.shape or len(angles) < 2
                or not np.isfinite(angles).all() or not np.isfinite(widths).all()
                or np.any(np.diff(angles) <= 0) or np.any(angles <= 0)
                or np.any(angles >= 180) or np.any(widths < 0)):
            raise ValueError("Calibration needs at least two increasing 2theta angles and matching finite nonnegative FWHM values, both in degrees.")
        info["angles_deg"], info["widths_deg"] = angles, widths
    if not isinstance(info.get("calibrated", False), (bool, np.bool_)):
        raise ValueError("Instrument calibration confirmation must be an explicit boolean.")
    if info.get("calibrated") and not str(info.get("reference") or "").strip():
        raise ValueError("A calibrated instrument correction requires a reference identifier and matching-condition confirmation.")
    info["calibrated"] = bool(info.get("calibrated", False))
    return info


def instrument_width_at(angle_deg, instrument):
    if instrument["mode"] == "unknown":
        return None
    if instrument["mode"] == "constant":
        return instrument["fwhm_deg"]
    angles, widths = instrument["angles_deg"], instrument["widths_deg"]
    if angle_deg < angles[0] or angle_deg > angles[-1]:
        raise ValueError("Reflection lies outside the instrument calibration angle range; extrapolation is not allowed. Supply a local reference covering this reflection.")
    return float(np.interp(angle_deg, angles, widths))


def _window_bounds(x, y, centers, index, half_width):
    center = centers[index]
    left = x[0] if index == 0 else (centers[index - 1] + center) / 2
    right = x[-1] if index == len(centers) - 1 else (center + centers[index + 1]) / 2
    if half_width is None:
        # Estimate a contiguous half-height width only to choose a fitting
        # window. Reported widths come from joint raw-signal/background fits.
        rough = (x >= max(left, center - 3)) & (x <= min(right, center + 3))
        xx, yy = x[rough], y[rough]
        if len(xx) < 3:
            width = 0.
        else:
            nearest = int(np.argmin(np.abs(xx - center)))
            baseline = np.percentile(yy, 10)
            threshold = baseline + (yy[nearest] - baseline) / 2
            lo = hi = nearest
            while lo > 0 and yy[lo - 1] > threshold:
                lo -= 1
            while hi < len(yy) - 1 and yy[hi + 1] > threshold:
                hi += 1
            width = xx[hi] - xx[lo]
        half_width = max(.25, 4 * width, 12 * np.median(np.diff(x)))
    return float(max(left, center - half_width)), float(min(right, center + half_width))


def analyze_peaks(pattern, candidate_centers, wavelength_nm, profile="Gaussian",
                  shape_factor=.9, instrument=None, same_phase_confirmed=False,
                  size_dominated=False, isolated_confirmed=False,
                  window_half_width=None):
    """Fit candidate windows and return apparent Scherrer and conditional W-H.

    Calibration tables contain instrumental FWHM after removal of any intrinsic
    reference width, not the raw standard pattern. Linear width interpolation
    is bounded by the supplied calibration angles and explicitly disclosed.
    Unknown instrumental width is allowed for exploratory Scherrer analysis;
    W-H always requires an explicit correction and same-phase confirmation.
    """
    wavelength_nm = float(positive(wavelength_nm, "Wavelength (nm)"))
    shape_factor = float(positive(shape_factor, "Shape factor K"))
    if profile not in {"Gaussian", "Lorentzian"}:
        raise ValueError("Only pure Gaussian or Lorentzian FWHM profiles are supported; mixed Voigt widths need another method.")
    if window_half_width is not None:
        window_half_width = float(positive(window_half_width, "Fit window half width (degrees)"))
    x, y = pattern.two_theta_deg, pattern.intensity
    centers = np.asarray(candidate_centers, dtype=float)
    if centers.ndim != 1 or not np.isfinite(centers).all():
        raise ValueError("Detected candidate angles must be a finite one-dimensional list.")
    if np.any(centers < x[0]) or np.any(centers > x[-1]):
        raise ValueError("Candidate peak angles must lie inside the uploaded measurement range.")
    centers = np.unique(centers)
    info = validate_instrument(instrument, profile)
    notes = list(pattern.warnings)
    if info["mode"] == "unknown":
        notes.append("Instrumental broadening is unknown: Scherrer results are uncorrected apparent sizes, not calibrated measurements; Williamson–Hall is withheld.")
    elif not info["calibrated"]:
        notes.append("Instrument correction uses a declared assumed profile/width, not an experimentally established calibration.")
    if info["mode"] == "table":
        notes.append("Instrument FWHM is linearly interpolated only within the supplied calibration angle range; interpolation is a stated local approximation.")
    if not isolated_confirmed:
        notes.append("Inspect detected candidates and confirm isolated reflections without unresolved spectral doublets before reporting sizes.")
    if not size_dominated:
        notes.append("Strain/defect broadening has not been separated: each Scherrer value attributes all remaining width to finite size and is an apparent estimate.")
    notes.append("Coherent-domain length is not grain or particle size. Pure-profile, shape-factor and isotropic assumptions require physical justification.")
    rows = []
    for i, center in enumerate(centers):
        bounds = _window_bounds(x, y, centers, i, window_half_width)
        row = {"candidate_deg": float(center), "bounds_deg": bounds,
               "fit": None, "center_deg": None, "observed_fwhm_deg": None,
               "instrument_fwhm_deg": None, "sample_fwhm_deg": None,
               "size_nm": None, "status": "rejected", "warnings": []}
        rows.append(row)
        try:
            mask = (x >= bounds[0]) & (x <= bounds[1])
            fit = fit_peak(x[mask], y[mask], profile)
            row.update(fit=fit, center_deg=fit.center_deg, observed_fwhm_deg=fit.fwhm_deg)
            row["warnings"].extend(fit.warnings)
            if not fit.usable:
                continue
            if abs(fit.center_deg - center) > fit.fwhm_deg:
                row["warnings"].append("Fitted peak moved more than one FWHM from the detected candidate; inspect background and overlap.")
                continue
            width = instrument_width_at(fit.center_deg, info)
            row["instrument_fwhm_deg"] = width
            sample_width = fit.fwhm_deg if width is None else float(correct_fwhm(fit.fwhm_deg, width, profile))
            row["sample_fwhm_deg"] = sample_width
            if width is not None and sample_width / fit.fwhm_deg < .3:
                row["warnings"].append("Instrument-dominated width: corrected size is highly sensitive to calibration error.")
            if not isolated_confirmed:
                row["status"] = "width only"
                row["warnings"].append("Isolation/no-doublet confirmation missing; size withheld.")
                continue
            row["size_nm"] = float(scherrer_size_nm(fit.center_deg, sample_width, wavelength_nm, shape_factor))
            row["status"] = "conditional Scherrer" if size_dominated and info["calibrated"] else "apparent Scherrer"
            if width is None:
                row["warnings"].append("Uncorrected observed width used; instrumental broadening is unknown.")
            if not size_dominated:
                row["warnings"].append("All remaining width attributed to size; unresolved strain/defect contributions can bias the apparent size.")
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            row["warnings"].append(str(exc))
    values = np.asarray([row["size_nm"] for row in rows if row["size_nm"] is not None])
    summary = {"count": int(len(values)), "median_nm": None, "mean_nm": None,
               "std_nm": None, "min_nm": None, "max_nm": None,
               "interpretation": "Descriptive variation across accepted reflections, not a confidence interval or proof of an isotropic crystallite size."}
    if len(values):
        summary.update(median_nm=float(np.median(values)), mean_nm=float(np.mean(values)),
                       std_nm=float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                       min_nm=float(values.min()), max_nm=float(values.max()))
    usable = [row for row in rows if row["size_nm"] is not None and row["instrument_fwhm_deg"] is not None]
    wh = {"status": "withheld", "reason": "", "result": None, "category": "Insufficient information"}
    if profile != "Lorentzian":
        wh["reason"] = "The simplified linear Williamson–Hall equation requires additive Lorentzian FWHM. Gaussian quadrature cannot be inserted into that equation."
    elif info["mode"] == "unknown":
        wh["reason"] = "A local instrumental correction is required before Williamson–Hall. Supply a matching Lorentzian reference or an explicitly declared assumed instrumental profile."
    elif not isolated_confirmed:
        wh["reason"] = "Confirm isolated reflections without unresolved doublets before Williamson–Hall."
    elif not same_phase_confirmed:
        wh["reason"] = "Confirm that selected reflections are independent and belong to the same identified phase. Detected maxima do not establish phase identity."
    elif len(usable) < 3:
        wh["reason"] = "At least three independent, usable, instrument-corrected reflections with a broad angular distribution are required."
    else:
        try:
            result = williamson_hall([r["center_deg"] for r in usable],
                                     [r["sample_fwhm_deg"] for r in usable],
                                     wavelength_nm, shape_factor, profile)
            if not info["calibrated"]:
                result["warnings"].append("Instrument width is an assumed correction: this W-H result is exploratory, not calibrated experimental size/microstrain.")
            result["warnings"].append("Microstrain is dimensionless heterogeneous lattice-spacing broadening in this model, not residual stress.")
            if result["size_justified"]:
                wh.update(status="available", reason="Conditional isotropic uniform-deformation fit under additive Lorentzian FWHM assumptions.",
                          result=result, category="Preliminary estimate" if result["warnings"] or not info["calibrated"] else "Supported under stated assumptions")
            else:
                wh.update(reason="Positive fitted intercept is too weakly constrained or angular leverage is inadequate; primary W-H size is withheld. Inspect the regression diagnostics and obtain more broadly distributed reflections.", result=result)
        except ValueError as exc:
            wh["reason"] = str(exc)
    category = "Insufficient information" if not len(values) else "Preliminary estimate"
    if (len(values) and info["calibrated"] and size_dominated
            and not pattern.warnings and all(not row["warnings"] for row in rows)):
        category = "Supported under stated assumptions"
    if not len(centers):
        notes.append("No detected peak candidates. Lower prominence only if real peaks are visible, or review the background and measurement range.")
    return {"peaks": rows, "summary": summary, "wh": wh, "warnings": notes,
            "category": category, "profile": profile, "wavelength_nm": wavelength_nm,
            "shape_factor": shape_factor, "instrument": info,
            "breadth_definition": "FWHM on 2theta axis; degrees converted to radians for Scherrer and Williamson–Hall",
            "instrument_corrected": any(row["instrument_fwhm_deg"] is not None
                                        and row["sample_fwhm_deg"] is not None for row in rows)}
