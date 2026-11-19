"""Regenerate the ESRF BM16 calibration and GSAS-II refinement.

Run as a module from the repository root with a working GSAS-II Python
environment. The original round-robin .xy files are downloaded separately.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from xrd.ceria import (
    ASSETS, CIF_SHA256, FIT_LIMITS_DEG, REFERENCE_SHA256, SAMPLE_SHA256, WAVELENGTH_ANGSTROM,
    load_evidence, local_reference_fits, physics_comparison, read_measured, sha256,
)

GSAS_REVISION = "46b3d7b541d344f9506848c37b35737b1a87ddac"
INSTRUMENT_PARAMETERS = ("W", "Y", "Z")
INITIAL_PROFILE = (
    "#GSAS-II instrument parameter file; do not add/delete items!\n"
    "Type:PXC;Bank:1;Diff-type:Debye-Scherrer\n"
    f"Lam:{WAVELENGTH_ANGSTROM};Zero:0.;Polariz.:1.;Azimuth:0.\n"
    "U:0.;V:0.;W:0.10;X:0.1;Y:0.;Z:0.;SH/L:0.002\n"
)


def _project(table, cif, profile, work, name, size_um):
    from GSASII import GSASIIfiles, GSASIIscriptable

    GSASIIfiles.G2SetPrintLevel("warn")
    data_file = work / f"{name}.xye"
    np.savetxt(data_file, table, fmt="%.7f")
    project = GSASIIscriptable.G2Project(newgpx=str(work / f"{name}.gpx"))
    phase = project.add_phase(str(cif), phasename="Ceria")
    hist = project.add_powder_histogram(str(data_file), str(profile), phases=[phase], fmthint="xye")
    if hist.data["Sample Parameters"]["Type"] != "Debye-Scherrer":
        raise ValueError("This refinement requires the matched Debye-Scherrer capillary profile.")
    hist.set_refinements({
        "Limits": list(FIT_LIMITS_DEG),
        "Background": {"type": "chebyschev-1", "refine": True, "no. coeffs": 8},
        "Sample Parameters": ["Scale"],
    })
    phase.set_refinements({"Cell": True})
    phase.setSampleProfile(hist, "microstrain", "isotropic", 0., LGmix=1.)
    phase.setSampleProfile(hist, "size", "isotropic", size_um, LGmix=.9)
    project.set_Controls("cycles", 40)
    project.data["Controls"]["data"]["min dM/M"] = 1e-5
    project.refine()
    return project, phase, hist


def _diagnostics(project, phase, hist):
    values = project["Covariance"]["data"]["Rvals"]
    if not values.get("converged") or values["SVD0"] or not np.isfinite(values["Rwp"]):
        raise ValueError("GSAS-II did not return a converged fit with an identifiable parameter set.")
    return {
        "rwp_percent": float(hist.get_wR()),
        "gof": float(values["GOF"]),
        "points": int(values["Nobs"]),
        "varied_parameters": project["Covariance"]["data"]["varyList"],
        "cell_a_angstrom": float(phase.get_cell()["length_a"]),
    }


def _calibrate(reference, cif, profile, work):
    from GSASII import GSASIIfiles

    initial = work / "initial.instprm"
    initial.write_text(INITIAL_PROFILE)
    project, phase, hist = _project(reference, cif, initial, work, "annealed_standard", 10)
    hist.set_refinements({"Instrument Parameters": list(INSTRUMENT_PARAMETERS)})
    project.refine()
    diagnostics = _diagnostics(project, phase, hist)
    inst, additional = hist.data["Instrument Parameters"]
    theta = np.deg2rad(np.asarray(FIT_LIMITS_DEG) / 2)
    # U and V are fixed at zero. Check the physical Lorentzian width
    # throughout the interval, including an internal stationary point if any.
    x, y, z = (float(inst[key][1]) for key in ("X", "Y", "Z"))
    candidate_angles = [*theta]
    if x and abs(y / x) <= 1:
        stationary = np.arcsin(-y / x)
        if theta[0] < stationary < theta[1]:
            candidate_angles.append(stationary)
    angles = np.asarray(candidate_angles)
    if inst["W"][1] <= 0 or np.any(x / np.cos(angles) + y * np.tan(angles) + z <= 0):
        raise ValueError("The refined instrument profile has a nonpositive component in the fitted range.")
    with profile.open("w") as output:
        GSASIIfiles.WriteInstprm(output, inst, additional, Sample=hist.data["Sample Parameters"])
    return diagnostics


def refine_sample(content, cif, profile, work):
    """Run the fixed, validated refinement recipe on the exact measured file."""
    table = read_measured(content, SAMPLE_SHA256)
    project, phase, hist = _project(table, cif, profile, work, "ceria_sample", .029)
    phase.set_HAP_refinements({
        "Size": {"type": "isotropic", "refine": True, "LGmix": {"value": .8, "refine": True}},
    }, histograms=[hist])
    project.refine()
    diagnostics = _diagnostics(project, phase, hist)
    size_um, _, fraction = phase.data["Histograms"][hist.name]["Size"][1]
    if not np.isfinite(size_um) or not 0.001 < size_um < 10 or not 0 < fraction < 1:
        raise ValueError("Refined size or mixing reached an invalid boundary. Review the GSAS-II fit.")
    return {**diagnostics, "size_nm": float(size_um * 1000), "lorentzian_fraction": float(fraction)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Refit the instrument from this annealed standard.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--work-dir", type=Path, help="Optional directory to retain projects and GSAS-II listings.")
    options = parser.parse_args()
    sample = options.sample.read_bytes()
    cif = ASSETS / "9009008.cif"
    if sha256(cif.read_bytes()) != CIF_SHA256:
        raise ValueError("The structure does not match COD 9009008 revision 291735.")
    with TemporaryDirectory(prefix="xrd-ceria-") as scratch:
        work = (options.work_dir or Path(scratch)).resolve()
        work.mkdir(parents=True, exist_ok=True)
        if options.reference:
            profile = options.out.with_name("bm16.instprm").resolve()
            reference = read_measured(options.reference.read_bytes(), REFERENCE_SHA256)
            calibration = _calibrate(reference, cif, profile, work)
            reference_fits = local_reference_fits(reference)
            report = {"calibration": calibration, "local_reference_fits": reference_fits,
                      "physics": physics_comparison(sample, reference_fits),
                      "sample_sha256": SAMPLE_SHA256, "reference_sha256": REFERENCE_SHA256,
                      "instprm_sha256": sha256(profile.read_bytes()),
                      "fit_limits_deg": FIT_LIMITS_DEG, "gsas_revision": GSAS_REVISION}
        else:
            load_evidence()
            profile = ASSETS / "bm16.instprm"
            report = {}
        report["gsas"] = refine_sample(sample, cif, profile, work)
    options.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
