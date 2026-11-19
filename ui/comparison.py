"""Comparison view for the measurement already loaded in Analyze My XRD."""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ui.analyze import analysis_identity, execute_analysis, workflow_metadata
from xrd.comparison import DISAGREEMENT_THRESHOLD, Reflection, compare_methods, pairwise_differences


def _reflection_controls(data, metadata):
    pattern = data["pattern"]
    candidates = sorted(float(value) for value in data["candidates"])
    key = "comparison_" + data["identity"]
    with st.expander("Select and calibrate Williamson–Hall reflections"):
        st.caption("Choose at least three isolated reflections from one identified phase. Review their fit windows and supply a matching local Lorentzian instrument width at each angle.")
        selected = st.multiselect("Williamson–Hall reflections", candidates,
                                  format_func=lambda angle: f"{angle:.4f}° 2θ",
                                  key=key + "_reflections", max_selections=12)
        if not selected:
            return (), False, ""
        reference = st.text_input("Williamson–Hall calibration reference",
                                  value=metadata["scherrer_reference"], key=key + "_reference")
        confirmed = st.checkbox("I verified one phase, isolated peaks without unresolved doublets, matching local calibration, and isotropic additive Lorentzian broadening",
                                key=key + "_confirm")
        reflections = []
        minimum, maximum = map(float, pattern.two_theta_deg[[0, -1]])
        for angle in sorted(selected):
            index = candidates.index(angle)
            # Keep suggested neighboring windows disjoint. The user must still
            # review the raw profile and confirm isolation and adequate tails.
            lower = max(minimum, angle - 1.2)
            upper = min(maximum, angle + 1.2)
            if index:
                lower = max(lower, angle - .45 * (angle - candidates[index - 1]))
            if index + 1 < len(candidates):
                upper = min(upper, angle + .45 * (candidates[index + 1] - angle))
            label = f"{angle:.4f}°"
            field_key = f"{key}_{index}"
            a, b, c = st.columns(3)
            lower = a.number_input(f"Lower bound at {label} (2θ degrees)", minimum, maximum, lower,
                                   format="%.5f", key=field_key + "_lower")
            upper = b.number_input(f"Upper bound at {label} (2θ degrees)", minimum, maximum, upper,
                                   format="%.5f", key=field_key + "_upper")
            width = c.number_input(f"Instrument FWHM at {label} (2θ degrees)", min_value=0., value=None,
                                   format="%.5f", key=field_key + "_width")
            reflections.append(Reflection(angle, lower, upper, width))
        return tuple(reflections), confirmed, reference.strip()


def _plot_sizes(results):
    available = [result for result in results if result.size_nm is not None]
    if not available:
        st.info("A size chart will appear when a method returns an estimate.")
        return
    fig, ax = plt.subplots(figsize=(6.5, 2.5), layout="constrained")
    bars = ax.barh([result.method for result in available], [result.size_nm for result in available], color="#30798a")
    ax.bar_label(bars, fmt="%.4g", padding=4)
    ax.margins(x=.22)
    ax.set_xlabel("Estimated crystallite size (nm)")
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close(fig)


def _plot_regression(regression):
    fig, ax = plt.subplots(figsize=(6.5, 3), layout="constrained")
    ax.scatter(regression["x"], regression["y"], label="Corrected reflections")
    order = np.argsort(regression["x"])
    ax.plot(regression["x"][order], regression["fitted"][order], label="Linear regression")
    ax.set(xlabel="4 sin θ (dimensionless)", ylabel="β cos θ (radians)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def _display(results, source, line_plot):
    table = []
    for result in results:
        if result.method == "LPA-NN" and result.status == "Unavailable":
            limitation = "See input checks and held-out validation evidence."
        else:
            # Put the method-specific reason before general file warnings.
            ordinary = ("Uneven sampling:", "Negative intensities:", "Conditional coherent-domain")
            limitation = next((warning for warning in result.warnings if not warning.startswith(ordinary)),
                              result.assumptions)
        table.append({"Method": result.method,
                      "Crystallite Size (nm)": result.size_nm,
                      "Status": f"{result.status} · {result.compatibility}", "Limitations": limitation})
    st.dataframe(pd.DataFrame(table), hide_index=True,
                 column_config={"Crystallite Size (nm)": st.column_config.NumberColumn(format="%.5g")})
    st.caption("Blank values indicate an unavailable estimate. Data source: " + source)
    _plot_sizes(results)
    differences = pairwise_differences(results)
    st.write("**Pairwise relative differences**")
    st.caption("Denominator: arithmetic mean of each pair. No independent, equivalent size reference is supplied. These percentages describe method disagreement.")
    if differences:
        st.dataframe(pd.DataFrame(differences), hide_index=True)
        for pair in differences:
            if pair["Relative difference (%)"] > DISAGREEMENT_THRESHOLD:
                st.warning(f"Large disagreement between {pair['Method A']} and {pair['Method B']}: {pair['Relative difference (%)']:.1f}%. Review the assumptions and fit diagnostics.")
        st.caption(f"Review threshold: above {DISAGREEMENT_THRESHOLD}% of the pair mean. This is a descriptive flag.")
    else:
        st.info("At least two numerical estimates are required for a pairwise comparison.")

    regression = results[1].diagnostics.get("regression")
    if regression is not None:
        st.write(f"**Williamson–Hall effective microstrain:** {regression['microstrain']:.6g} (dimensionless)")
        st.caption(f"The slope describes heterogeneous lattice strain under the stated model. Residual stress requires a different measurement and interpretation. Regression R² = {regression['r_squared']:.5f}.")
        _plot_regression(regression)
    for result in results:
        with st.expander(f"{result.method} diagnostics"):
            st.write(f"**Input compatibility:** {result.compatibility}")
            st.write(f"**Assumptions:** {result.assumptions}")
            for warning in result.warnings:
                st.warning(warning)
            if result.diagnostics.get("reflections"):
                rows = []
                for record in result.diagnostics["reflections"]:
                    fit = record.get("fit", {})
                    rows.append({"Candidate (° 2θ)": record["candidate_deg"], "Observed FWHM (°)": fit.get("fwhm_deg"),
                                 "Instrument FWHM (°)": record["instrument_fwhm_deg"],
                                 "Corrected FWHM (°)": record.get("corrected_fwhm_deg"),
                                 "Status": record["status"], "Reason": record.get("reason", "")})
                st.caption("Calibration reference: " + result.diagnostics["reference"])
                st.dataframe(pd.DataFrame(rows), hide_index=True)
            if result.diagnostics.get("compatibility_checks"):
                st.write(result.diagnostics["model_identifier"])
                st.dataframe(pd.DataFrame(result.diagnostics["compatibility_checks"]), hide_index=True)
                st.json({key: result.diagnostics[key] for key in ("preprocessing_details", "validation_evidence")})
            if result.peak_fits:
                # Render only one fit per method at a time, including excluded fits.
                labels = tuple(label for label, _ in result.peak_fits)
                index = st.selectbox(f"{result.method} fit to inspect", range(len(result.peak_fits)),
                                     format_func=lambda i, names=labels: names[i])
                label, fit = result.peak_fits[index]
                st.caption(f"{label} · {fit.profile} · FWHM {fit.fwhm_deg:.5f}° 2θ · R² {fit.r_squared:.5f}")
                line_plot(fit.x, {"Observed": fit.observed, "Fit": fit.fitted}, source)
                line_plot(fit.x, {"Observed − fit": fit.residual}, source, ylabel="Residual (a.u.)")
    with st.expander("Interpreting differences"):
        st.write("**Observations.** Use the reported peak widths, fit residuals, calibration records, regression and compatibility checks to see what each estimate uses.")
        st.write("**Possible explanations.** Instrument width errors, microstrain, overlap, anisotropic broadening and mismatched peak shapes can change the fitted breadth or its interpretation. Scherrer uses one reflection under a size-dominance assumption. Williamson–Hall fits an isotropic model across several reflections. LPA-NN depends on its training distribution and size convention. A wavelength or geometry mismatch can shift or alter its input pattern. Pairwise differences alone do not establish which estimate is closer to the specimen’s size.")


def render_comparison(line_plot):
    st.subheader("Compare crystallite-size estimates")
    st.caption("For held-out ZrO₂ simulation accuracy, open Validation and select Synthetic · ZrO₂ GSAS-II CNN.")
    data = st.session_state.get("workflow_data")
    if data is None or st.session_state.get("workflow_stage") != 3:
        st.info("Load a pattern and confirm its measurement conditions in Analyze My XRD, then open this tab. The comparison uses that same pattern.")
        return
    if "ceria_content" in data:
        st.info("Use the matched Physics and GSAS-II comparison for this BM16 ceria sample in Analyze My XRD.")
        return
    conditions = st.session_state["workflow_conditions"]
    metadata = workflow_metadata(conditions)
    peak = st.session_state.get("workflow_peak")
    low, high = map(float, data["pattern"].two_theta_deg[[0, -1]])
    metadata["angular_range_deg"] = [low, high]
    st.caption(f"{data['name']} · {low:.4f}–{high:.4f}° 2θ · {metadata['wavelength_nm']:g} nm · {metadata['geometry']}")
    st.caption(f"Instrument: {metadata['instrument'] or 'Not declared'} · Scherrer calibration: {metadata['scherrer_instrument_mode']}")
    automatic = st.session_state.get("analysis_mode") == "Automatic multi-peak"
    batch = st.session_state.get("multi_result") if automatic else None
    if automatic:
        if batch:
            st.caption("Using the current Automatic multi-peak analysis. The Scherrer row shows the median across accepted reflections.")
        else:
            st.info("Run Automatic multi-peak analysis in Analyze My XRD to add its physics results. You can check LPA-NN compatibility independently.")
        reflections, confirmed, reference = (), False, ""
        peak = None
    elif peak and peak[2]:
        st.caption(f"Scherrer will use the confirmed window {peak[0]:.4f}–{peak[1]:.4f}° 2θ and the conditions from Analyze My XRD.")
    else:
        st.info("For Scherrer, select an isolated peak in Analyze My XRD and confirm its fit window. The other methods can be checked independently.")
    if not automatic:
        reflections, confirmed, reference = _reflection_controls(data, metadata)
    metadata["williamson_hall_reference"] = metadata["scherrer_reference"] if batch else reference
    metadata["williamson_hall_local_widths"] = [
        {"candidate_deg": reflection.candidate_deg, "fwhm_deg": reflection.instrument_fwhm_deg}
        for reflection in reflections
    ]
    identity = (analysis_identity(data, conditions, peak), reflections, confirmed, reference,
                automatic, batch["identity"] if batch else None)
    saved = st.session_state.get("comparison_result")
    if saved and saved[0] != identity:
        st.session_state.pop("comparison_result")
        saved = None
        st.info("Comparison cleared because its inputs changed. Run it again to use the new settings.")
    if st.button("Compare available methods", type="primary"):
        scherrer = partial(execute_analysis, data, conditions, *peak[:3], metadata=metadata) if peak and peak[2] else None
        with st.spinner("Running the available methods…"):
            results = compare_methods(data["pattern"], metadata, scherrer, reflections, confirmed, reference,
                                      conditions["k"], batch=batch["result"] if batch else None)
        saved = identity, results
        st.session_state["comparison_result"] = saved
    if saved:
        _display(saved[1], data["name"], line_plot)
