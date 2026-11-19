"""Evidence-based eligibility for the released two-output Spinel checkpoint.

The paper establishes some experimental context, not a transferable calibration.
Passing the measurable input checks alone NEVER approves prediction.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
from .preprocessing import original_input_tensor

MODEL_ID = "LPA-NN/Spinel@87f54292:6efd1ed8a8e2"
WAVELENGTH_NM = 0.071724  # Paper section 5: 0.71724 angstrom.
POINTS = 10501
PHASES = {"mgal2o4", "mgo", "alpha-al2o3"}
VALIDATION_BLOCKERS = (
    "No held-out dataset with known size labels matched to this Spinel checkpoint is publicly supplied; MAE/RMSE are unavailable.",
    "Exact checkpoint-linked instrument U,V,W,X,Y,Z coefficients, integration settings and intensity-scale envelope are unavailable.",
    "The exact 2theta training grid is not certified: the released notebook defines its grid for plotting only.",
    "The published five-output Spinel network is not the released two-output checkpoint; published accuracy cannot certify this asset.",
)


def notebook_grid():
    """Plotting grid from original Spinel notebook; NOT a verified training axis."""
    return np.linspace(5.8438, 27.0799, POINTS, endpoint=False)


def model_requirements():
    return {
        "model_identifier": MODEL_ID,
        "wavelength_nm": WAVELENGTH_NM,
        "wavelength_evidence": "Published paper section 5; exact training configuration not deposited",
        "input_shape": [None, POINTS, 1, 1],
        "output_shape": [None, 2],
        "notebook_plotting_grid": {
            "first_deg": float(notebook_grid()[0]),
            "last_deg": float(notebook_grid()[-1]),
            "step_deg": float((27.0799 - 5.8438) / POINTS),
            "stop_exclusive_deg": 27.0799,
            "certified_training_grid": False,
        },
        "phase_context": ["MgAl2O4", "MgO", "alpha-Al2O3"],
        "experimental_context": "ESRF BM01; sapphire capillary; 1200 C",
        "preprocessing": {
            "dtype": "float32", "positive_intensities_required": True,
            "reshape": "(N,10501) -> (N,10501,1,1)",
            "external_intensity_normalization": "none in original notebook",
            "background_subtraction": False, "interpolation": False,
            "log10": "inside original saved network; apply exactly once",
        },
        "outputs": {
            "size_nm": "inverse MinMax output[0] in micrometres, multiplied by 1000",
            "microstrain_percent": "inverse MinMax output[1], multiplied by 1e-4",
            "size_measurand": "GSAS-II/Rietveld size parameter; paper discusses integral-breadth size tau",
            "warning": "Do not equate model outputs to K=0.9 FWHM Scherrer D or the explorer's effective microstrain without convention analysis.",
        },
        "held_out_validation": {"status": "blocked", "labeled_sample_count": 0,
                                "mae_nm": None, "rmse_nm": None},
        "validated_upload_domain": None,
        "remaining_blockers": list(VALIDATION_BLOCKERS),
    }


@dataclass
class Check:
    name: str
    status: str  # pass, fail, unverified
    detail: str


def _phase_name(value):
    text = str(value).strip().lower().replace("α", "alpha")
    for char in [" ", "_", "₂", "₃", "₄"]:
        text = text.replace(char, {"₂": "2", "₃": "3", "₄": "4"}.get(char, ""))
    return text


def assess_compatibility(pattern, metadata):
    """Return auditable input checks; cannot be overridden by config flags."""
    metadata = metadata if isinstance(metadata, dict) else {}
    checks = []
    values = getattr(pattern, "intensity", pattern)
    try:
        tensor = original_input_tensor(values)
        if len(tensor) != 1:
            raise ValueError("Upload API expects one pattern, not a batch.")
        checks.append(Check("input_tensor", "pass", "10501 positive finite intensities, representable as positive finite float32; no resampling or scaling."))
    except (ValueError, TypeError, OverflowError) as exc:
        checks.append(Check("input_tensor", "fail", str(exc)))

    x = getattr(pattern, "two_theta_deg", None)
    if x is None:
        checks.append(Check("angular_axis", "unverified", "The actual 2theta axis is required; an intensity vector alone does not establish compatibility."))
    else:
        x = np.asarray(x, dtype=float)
        finite = x.ndim == 1 and len(x) == POINTS and np.isfinite(x).all() and np.all(np.diff(x) > 0)
        matches = finite and np.allclose(x, notebook_grid(), rtol=0, atol=1e-7)
        checks.append(Check("angular_axis", "pass" if matches else "fail",
                            "Matches original notebook plotting grid, without interpolation. This does not certify the training grid." if matches else
                            "Expected the notebook's 10501-point plotting grid (5.8438 to 27.0778777 degrees); arbitrary cropping, shifting or interpolation is not approved."))

    try:
        wl = float(metadata["wavelength_nm"])
        if not math.isfinite(wl) or wl <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        checks.append(Check("wavelength", "unverified", "Declare a finite positive measurement wavelength in nm."))
    else:
        match = math.isclose(wl, WAVELENGTH_NM, rel_tol=0, abs_tol=5e-8)
        checks.append(Check("wavelength", "pass" if match else "fail",
                            f"Measurement {wl:g} nm; published Spinel experiment used {WAVELENGTH_NM} nm (0.71724 angstrom). Matching wavelength alone is insufficient."))

    phases = metadata.get("phases")
    if not isinstance(phases, (list, tuple, set)) or not phases:
        checks.append(Check("phase_context", "unverified", "Declare the crystalline phases. This checkpoint targets MgAl2O4 in the MgO / alpha-Al2O3 reaction mixture, not arbitrary materials."))
    else:
        matches = {_phase_name(p) for p in phases} == PHASES
        checks.append(Check("phase_context", "pass" if matches else "fail",
                            "Declared phase mixture matches the paper's context; fractions and structural envelope still require validation." if matches else
                            "Phase mixture differs from MgAl2O4 + MgO + alpha-Al2O3; new materials require an appropriate trained checkpoint."))

    beamline = str(metadata.get("instrument", "")).strip().upper().replace(" ", "").replace("-", "").replace("/", "")
    checks.append(Check("instrument_identity", "pass" if beamline == "ESRFBM01" else "unverified" if not beamline else "fail",
                        "Reported ESRF BM01 identity matches; numerical calibration and acquisition settings are still unverified." if beamline == "ESRFBM01" else
                        "Declare the instrument. A laboratory instrument or another beamline is not interchangeable with the fixed BM01 response."))
    geometry = str(metadata.get("geometry", "")).lower()
    if "θ–2θ" in geometry or geometry == "bragg_brentano":
        checks.append(Check("geometry", "fail", "Reflection laboratory geometry does not match the published capillary experiment."))
    else:
        checks.append(Check("geometry", "unverified", "Published specimen was a sapphire capillary at BM01; exact checkpoint-linked detector integration/configuration has not been deposited."))

    checks.extend(Check(name, "unverified", detail) for name, detail in [
        ("training_grid", VALIDATION_BLOCKERS[2]),
        ("instrument_response_and_intensity_envelope", VALIDATION_BLOCKERS[1]),
        ("held_out_accuracy", VALIDATION_BLOCKERS[0]),
        ("checkpoint_to_publication_identity", VALIDATION_BLOCKERS[3]),
    ])
    failed = any(c.status == "fail" for c in checks)
    return {"eligible_for_prediction": False,
            "status": "Unsupported by current model" if failed else "Insufficient information",
            "checks": [asdict(c) for c in checks],
            "remaining_blockers": list(VALIDATION_BLOCKERS),
            "model_identifier": MODEL_ID}
