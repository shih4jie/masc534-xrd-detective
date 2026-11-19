"""Run existing estimators independently and describe their shared evidence."""
from dataclasses import dataclass, field
from itertools import combinations
import logging
from math import isfinite

from ml.inference import predict_crystallite_size
from physics.broadening import correct_fwhm
from physics.williamson_hall import williamson_hall
from ui.measurement import SUPPORTED_POWDER_GEOMETRIES
from xrd.peak_fitting import fit_peak

LOG = logging.getLogger(__name__)
DISAGREEMENT_THRESHOLD = 25


@dataclass
class MethodResult:
    method: str
    assumptions: str
    status: str = "Unavailable"
    compatibility: str = "Insufficient information"
    size_nm: float | None = None
    units: str = "nm"
    warnings: list[str] = field(default_factory=list)
    peak_fits: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def accept(self, size):
        if size is None:
            return
        if not isfinite(size) or size <= 0:
            raise ValueError("The method did not return a finite positive size.")
        self.size_nm = float(size)
        self.status = "Estimate available"


@dataclass(frozen=True)
class Reflection:
    candidate_deg: float
    lower_deg: float
    upper_deg: float
    instrument_fwhm_deg: float | None


def _run(result, calculate):
    try:
        calculate(result)
    except ValueError as exc:
        result.status = "Failed"
        result.compatibility = "Unsupported by current method"
        result.warnings.append(str(exc))
    except Exception:
        LOG.exception("%s comparison failed", result.method)
        result.status = "Failed"
        result.compatibility = "Execution failed"
        result.warnings.append("This method did not complete. Review its settings and try again.")
    return result


def _run_regression(pattern, metadata, reflections, confirmed, reference, shape_factor,
                    scherrer_local, result):
    result.warnings.extend(pattern.warnings)
    if len(reflections) < 3:
        result.warnings.append("Select at least three distinct, isolated reflections from one identified phase.")
        return
    if not confirmed or not reference.strip():
        result.warnings.append("Supply a matching calibration reference and confirm the Williamson–Hall assumptions.")
        return
    if metadata["wavelength_nm"] is None:
        result.warnings.append("Verify the wavelength in Analyze My XRD.")
        return
    if metadata.get("geometry") not in SUPPORTED_POWDER_GEOMETRIES:
        result.warnings.append("Declare a supported powder geometry in Analyze My XRD.")
        return
    ordered = sorted(reflections, key=lambda reflection: reflection.lower_deg)
    if any(a.upper_deg >= b.lower_deg for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Williamson–Hall windows must be disjoint to represent separate reflections.")

    centers, widths = [], []
    result.diagnostics["reference"] = reference
    result.diagnostics["reflections"] = records = []
    for reflection in ordered:
        lower, upper = reflection.lower_deg, reflection.upper_deg
        label = f"{reflection.candidate_deg:.4f}° 2θ"
        record = {"candidate_deg": reflection.candidate_deg, "window_deg": [lower, upper],
                  "instrument_fwhm_deg": reflection.instrument_fwhm_deg, "status": "Excluded"}
        records.append(record)
        try:
            x = pattern.two_theta_deg
            if not x[0] <= lower < reflection.candidate_deg < upper <= x[-1]:
                raise ValueError("Fit window must bracket the selected reflection within the measured range.")
            mask = (x >= lower) & (x <= upper)
            fit = fit_peak(x[mask], pattern.intensity[mask], "Lorentzian")
            result.peak_fits.append((label, fit))
            record["fit"] = fit.summary()
            if not fit.usable:
                raise ValueError(" ".join(fit.warnings))
            if abs(fit.center_deg - reflection.candidate_deg) > fit.fwhm_deg:
                raise ValueError("Fit moved away from the selected candidate. Inspect the window for overlap.")
            if reflection.instrument_fwhm_deg is None:
                raise ValueError("Missing local instrumental FWHM.")
            if scherrer_local is not None:
                local_center, local_width = scherrer_local
                if lower <= local_center <= upper and abs(local_width - reflection.instrument_fwhm_deg) > .00005:
                    raise ValueError("Instrument FWHM conflicts with the calibrated Lorentzian Scherrer width for this reflection. Use the same reference width.")
            corrected = float(correct_fwhm(fit.fwhm_deg, reflection.instrument_fwhm_deg, "Lorentzian"))
        except ValueError as exc:
            record["reason"] = str(exc)
            result.warnings.append(f"{label} excluded. {exc}")
            continue
        if corrected / fit.fwhm_deg < .3:
            result.warnings.append(f"{label} has instrument-dominated width. Its correction is sensitive to calibration error.")
        centers.append(fit.center_deg)
        widths.append(corrected)
        record.update(status="Used", corrected_fwhm_deg=corrected)
    if len(centers) < 3:
        result.warnings.append(f"Only {len(centers)} usable, calibrated reflections. At least three are required.")
        if result.peak_fits or any("reason" in row for row in records):
            result.status = "Failed"
        return
    regression = williamson_hall(centers, widths, metadata["wavelength_nm"], shape_factor)
    result.diagnostics["regression"] = regression
    result.warnings.extend(regression["warnings"])
    if regression["size_nm"] is None:
        result.compatibility = "Regression does not support a size"
    else:
        result.compatibility = "Conditional on declared profile and calibration"
        result.accept(regression["size_nm"])


def compare_methods(pattern, metadata, scherrer, reflections=(), wh_confirmed=False,
                    wh_reference="", shape_factor=.9, *, batch=None):
    """The Scherrer callback uses the existing single-reflection workflow."""
    def run_scherrer(result):
        if batch is not None:
            result.assumptions = "Median of accepted per-reflection Scherrer sizes. Finite size accounts for the breadth used in each estimate."
            result.compatibility = batch["category"]
            result.warnings.extend(batch["warnings"])
            for row in batch["peaks"]:
                if row["fit"] is not None:
                    result.peak_fits.append((f"{row['candidate_deg']:.4f}° 2θ · {row['status']}", row["fit"]))
                result.warnings.extend(f"{row['candidate_deg']:.4f}° 2θ. {note}" for note in row["warnings"])
            result.accept(batch["summary"]["median_nm"])
            return
        if scherrer is None:
            result.warnings.append("Select and confirm an isolated reflection in Analyze My XRD.")
            return
        analysis = scherrer()
        result.peak_fits.append(("Scherrer reflection", analysis["fit"]))
        result.compatibility = analysis["category"]
        result.warnings.extend(analysis["warnings"])
        result.accept(analysis["size_nm"])
        if not analysis["fit"].usable:
            result.status = "Failed"

    sch = _run(MethodResult("Scherrer", "Isolated reflection, size dominated width, declared pure profile and FWHM shape factor."), run_scherrer)
    calibrated_lorentzian = metadata.get("scherrer_calibrated") and metadata.get("scherrer_profile") == "Lorentzian"
    local_width = metadata.get("scherrer_instrument_fwhm_deg")
    scherrer_local = None
    if calibrated_lorentzian and local_width is not None and sch.peak_fits and sch.peak_fits[0][1].usable:
        scherrer_local = (sch.peak_fits[0][1].center_deg, local_width)

    def run_wh(result):
        if batch is not None:
            wh = batch["wh"]
            result.compatibility = wh["category"]
            result.warnings.append(wh["reason"])
            regression = wh["result"]
            if regression is not None:
                result.diagnostics["regression"] = regression
                result.warnings.extend(regression["warnings"])
            if wh["status"] == "available":
                result.accept(regression["size_nm"])
            return
        _run_regression(pattern, metadata, reflections, wh_confirmed, wh_reference,
                        shape_factor, scherrer_local, result)

    wh = _run(MethodResult("Williamson–Hall", "Distinct same-phase reflections, locally calibrated Lorentzian FWHM and isotropic additive size and strain broadening."), run_wh)

    def run_ml(result):
        prediction = predict_crystallite_size(pattern, metadata)
        result.diagnostics = prediction.to_dict()
        result.compatibility = prediction.input_compatibility_status
        result.warnings.extend(prediction.diagnostic_warnings)
        result.warnings.extend(prediction.applicable_model_limitations)
        if prediction.eligible_for_prediction:
            if prediction.units != "nm":
                raise ValueError("The inference API must return crystallite size in nm.")
            result.accept(prediction.predicted_crystallite_size)

    ml = _run(MethodResult("LPA-NN", "Validated checkpoint for the declared phase system, wavelength, angular grid, geometry and instrument response."), run_ml)
    return [sch, wh, ml]


def pairwise_differences(results):
    """Symmetric relative differences for estimates without independent truth."""
    valid = [r for r in results if r.size_nm is not None and isfinite(r.size_nm) and r.size_nm > 0]
    differences = []
    for a, b in combinations(valid, 2):
        mean = a.size_nm / 2 + b.size_nm / 2
        percent = 100 * (abs(a.size_nm - b.size_nm) / mean)
        differences.append({"Method A": a.method, "Method B": b.method,
                            "Mean denominator (nm)": mean, "Relative difference (%)": percent})
    return differences
