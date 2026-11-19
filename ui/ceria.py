"""Streamlit workflow for the verified ESRF BM16 sample."""
import json
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd
import streamlit as st

from physics.units import wavelength_nm
from xrd import ceria

LOG = logging.getLogger(__name__)
_GEOMETRIES = ["Unknown / not declared", ceria.GEOMETRY, "Other geometry"]


def render_reference_download():
    with st.expander("Try a measured pattern with both methods"):
        st.caption("Measured ceria (CeO₂) from the IUCr round robin, ESRF BM16. "
                   "Download the original .xy and drag it into the upload area.")
        st.link_button("Download measured pattern (.xy)", ceria.SOURCE + ceria.SAMPLE_NAME)
        st.caption("If the link opens as text, save the original file without edits. "
                   "Use degrees 2θ, 0.39982 Å and a powder capillary scan with an analyzer detector.")
        if st.session_state.get("show_research_tools", False):
            st.markdown(f"[Measured annealed reference]({ceria.SOURCE + ceria.REFERENCE_NAME}) · "
                        f"[IUCr dataset]({ceria.SOURCE}s-s_rr.htm)")
            for filename, label in (("9009008.cif", "Phase structure (CIF)"),
                                    ("bm16.instprm", "Derived instrument profile (.instprm)")):
                st.download_button(label, (ceria.ASSETS / filename).read_bytes(), filename)


def render_conditions(data, *, back_to_data):
    st.subheader("2 · Confirm conditions")
    st.button("Back to data", on_click=back_to_data)
    st.caption(f"{data['name']} · original IUCr round-robin measurement matched by file contents.")
    st.info("The published record gives cubic ceria (CeO₂), ESRF BM16, "
            "λ = 0.39982 Å, 2θ in degrees and powder capillary geometry with an analyzer detector.")
    a, b = st.columns([2, 1])
    conditions = data.setdefault("ceria_conditions", {})
    units = ["angstrom", "nm", "pm"]
    value = a.number_input("Wavelength", value=conditions.get("wavelength", ceria.WAVELENGTH_ANGSTROM),
                           min_value=.00001, format="%.6f", key="bm16_wavelength")
    unit = b.selectbox("Wavelength unit", units, index=units.index(conditions.get("unit", units[0])),
                       key="bm16_unit")
    geometry = st.selectbox("Measurement geometry", _GEOMETRIES,
                            index=_GEOMETRIES.index(conditions.get("geometry", _GEOMETRIES[0])),
                            key="bm16_geometry")
    verified = st.checkbox("I verified these conditions and the CeO₂ phase against the published record.",
                           value=conditions.get("verified", False), key="bm16_verified")
    calibration = st.checkbox(
        "I accept the annealed reference as the instrument baseline and a size-only comparison "
        "with strain fixed to zero.", value=conditions.get("calibration", False), key="bm16_calibration")
    conditions.update(wavelength=value, unit=unit, geometry=geometry, verified=verified, calibration=calibration)
    if st.button("Continue to analysis", type="primary"):
        if geometry != ceria.GEOMETRY:
            st.error("Select the recorded powder capillary geometry with an analyzer detector. "
                     "For a different measurement, use the standard Physics workflow.")
        elif not verified or abs(wavelength_nm(value, unit) - ceria.WAVELENGTH_ANGSTROM / 10) > 5e-8:
            st.error("Verify the published wavelength of 0.39982 Å (0.039982 nm) and confirm the record above.")
        elif not calibration:
            st.error("Review and accept the matched annealed reference and size-only assumption before comparing the fits.")
        else:
            st.session_state["workflow_stage"] = 3
            st.rerun()


def _refine(content, evidence):
    executable = os.environ.get("XRD_GSAS_PYTHON")
    if not executable:
        return evidence["gsas"], "Recorded GSAS-II refinement for this exact upload"
    with TemporaryDirectory(prefix="xrd-upload-") as scratch:
        folder = Path(scratch)
        sample = folder / ceria.SAMPLE_NAME
        sample.write_bytes(content)
        output = folder / "result.json"
        run = subprocess.run(
            [executable, "-m", "validation.refine_ceria_bm16", "--sample", str(sample), "--out", str(output)],
            cwd=ceria.ASSETS.parent.parent, capture_output=True, text=True, timeout=90, check=False,
        )
        if run.returncode != 0:
            LOG.error("GSAS-II process exited with status %d: %s", run.returncode, run.stderr[-3000:])
            raise RuntimeError("GSAS-II could not complete the refinement.")
        return json.loads(output.read_text())["gsas"], "Live GSAS-II refinement of this upload"


def render_analysis(data, *, change_data, back_to_conditions):
    st.subheader("3 · Physics and GSAS-II", anchor="ceria-comparison")
    st.button("Back to conditions", on_click=back_to_conditions)
    st.caption("Measured ceria · ESRF BM16 · 0.39982 Å · analyzer capillary scan")
    content = data["ceria_content"]
    run_comparison = st.button("Compare Physics and GSAS-II", type="primary")
    if run_comparison:
        started = perf_counter()
        st.session_state.pop("ceria_result", None)
        with st.status("Analyzing the pattern and checking the matched calibration…", expanded=True) as status:
            try:
                evidence = ceria.load_evidence()
                physics = ceria.physics_comparison(content, evidence["local_reference_fits"])
            except (ValueError, RuntimeError, OSError, KeyError, FloatingPointError):
                LOG.exception("Matched ceria Physics analysis failed")
                st.session_state.pop("ceria_result", None)
                status.update(label="Physics fit needs attention", state="error", expanded=False)
                st.error("The matched calibration or peak fit could not be verified. Download the original measured "
                         "file again or use the standard Physics workflow and review your inputs.")
            else:
                result = {"physics": physics}
                st.session_state["ceria_result"] = result
                try:
                    gsas, label = _refine(content, evidence)
                    diff, percent = ceria.comparison_difference(physics["size_nm"], gsas["size_nm"])
                except (ValueError, OSError, KeyError, RuntimeError, subprocess.TimeoutExpired):
                    LOG.exception("Live GSAS-II refinement failed")
                    status.update(label="Physics complete · GSAS-II needs attention", state="error", expanded=False)
                else:
                    result.update(gsas=gsas, label=label, difference=diff, percent=percent,
                                  elapsed_s=perf_counter() - started)
                    status.update(label="Comparison complete", state="complete", expanded=False)

    result = st.session_state.get("ceria_result")
    if result:
        physics = result["physics"]
        a, b = st.columns(2)
        a.metric("Physics · Scherrer", f"{physics['size_nm']:.2f} nm")
        if "gsas" in result:
            gsas = result["gsas"]
            b.metric("GSAS-II · Rietveld", f"{gsas['size_nm']:.2f} nm")
            a, b = st.columns(2)
            a.metric("Absolute difference", f"{result['difference']:.2f} nm")
            b.metric("Percent difference", f"{result['percent']:.2f}%")
            st.caption(f"{result['label']} · Rwp {gsas['rwp_percent']:.2f}% · "
                       "Percentage uses the mean of the two sizes as denominator.")
        else:
            b.metric("GSAS-II · Rietveld", "Unavailable")
            st.warning("The configured GSAS-II solver could not complete a verified refinement. "
                       "Check its installation or retry. Physics is available on its own.")
        st.warning("These are coherent-domain sizes. The GSAS-II refinement fixed microstrain at zero as a "
                   "fitting assumption. The Scherrer estimate assumes negligible strain. The annealed reference "
                   "can retain some broadening, and residual strain, anisotropy and different width conventions "
                   "can affect agreement. The percentage measures disagreement between the two fits.")
        st.caption("Physics uses the median of four local Voigt fits and Scherrer K = 1. GSAS-II uses an "
                   "isotropic full-pattern size parameter over 6.7–28.0° 2θ.")
        if st.session_state.get("show_research_tools", False):
            with st.expander("Fit diagnostics and reproducibility"):
                rows = [{"Reflection": row["reflection"], "Corrected FWHM (°)": row["corrected_fwhm_deg"],
                         "Physics size (nm)": row["size_nm"],
                         "Sample weighted residual RMS": row["observed"]["weighted_residual_rms"],
                         "Reference weighted residual RMS": row["reference"]["weighted_residual_rms"]}
                        for row in physics["reflections"]]
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                if "gsas" in result:
                    st.caption(f"Server computation took {result['elapsed_s']:.2f} s.")
                    st.json(result["gsas"])
                st.markdown("[IUCr round-robin paper](https://doi.org/10.1107/S0021889804022551) · "
                            "[Phase structure, COD 9009008](https://www.crystallography.net/cod/9009008.html)")
    st.button("Change data", on_click=change_data)
    if run_comparison and result:
        token = str(int(perf_counter() * 1e9))
        # As in the standard analyzer, position a newly rendered result below
        # Streamlit's fixed header. The script contains no upload data.
        st.html("""<script>/* RUN_TOKEN */
        requestAnimationFrame(() => requestAnimationFrame(() => {
          const heading = document.getElementById('ceria-comparison');
          const main = document.querySelector('[data-testid="stMain"]');
          if (heading && main) main.scrollBy({top: heading.getBoundingClientRect().top - 70, behavior: 'instant'});
        }));
        </script>""".replace("RUN_TOKEN", token), unsafe_allow_javascript=True)
