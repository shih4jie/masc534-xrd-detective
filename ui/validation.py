"""Validation views over saved evidence, with no model inference at render time."""
import json

import pandas as pd
import streamlit as st


def _render_zirconia_benchmark(results):
    report_path = results / "zro2_gsas_size_benchmark.json"
    trials_path = results / "zro2_gsas_size_predictions.csv"
    plot_path = results / "zro2_gsas_size_benchmark.png"
    if not all(path.is_file() for path in (report_path, trials_path, plot_path)):
        st.warning("The saved ZrO₂ benchmark files are missing. See the README for reproduction steps.")
        return

    report = json.loads(report_path.read_text())
    st.markdown("#### Monoclinic ZrO₂ · held-out simulations")
    st.caption("The LPA-NN Size architecture was trained on new COD + GSAS-II simulations. Reference sizes are simulator inputs.")
    st.write(
        f"**{report['sample_count']} held-out patterns** · "
        f"**MAE {report['mae']:.2f} nm** · **RMSE {report['rmse']:.2f} nm** · "
        f"**Mean absolute relative error {report['mean_absolute_relative_error_percent']:.2f}%**"
    )
    low, high = report["truth_range_nm"]
    st.caption(f"Reference sizes {low:.2f}–{high:.2f} nm. Mean signed error (prediction − reference) {report['bias_nm']:+.2f} nm.")
    st.info("These errors apply to simulations with zero microstrain and the specified Cu Kα₁ instrument. Experimental accuracy and evaluation on the authors’ SVQART test split remain pending.")
    st.image(str(plot_path), caption="CNN predictions and residuals against simulated reference sizes.")
    with st.expander("Simulation conditions and accuracy by size range"):
        dataset = report["dataset"]
        grid = dataset["angular_grid"]
        counts = dataset["split_counts"]
        st.write(
            f"Cu Kα₁ at {dataset['wavelength_angstrom']} Å, {dataset['geometry']} geometry, "
            f"{grid['min_deg']:g}–{grid['max_deg']:g}° 2θ at {grid['step_deg']:g}° intervals. "
            f"Training used {counts['train']['patterns']:,} patterns. "
            f"Checkpoint selection used {counts['validation']['patterns']:,} validation patterns."
        )
        rows = [
            {
                "Size range (nm)": (
                    f"{row['lower_nm']:g}–<{row['upper_nm']:g}"
                    if row["upper_nm"] is not None else f"≥{row['lower_nm']:g}"
                ),
                "Test patterns": row["sample_count"],
                "MAE (nm)": round(row["mae"], 3),
                "RMSE (nm)": round(row["rmse"], 3),
                "Mean absolute relative error (%)": round(100 * row["mean_absolute_relative_error"], 2),
            }
            for row in report["size_bins"] if row["sample_count"]
        ]
        st.dataframe(rows, hide_index=True)
    st.download_button("Download ZrO₂ test predictions", trials_path.read_bytes(),
                       file_name=trials_path.name, mime="text/csv")
    sensitivity_path = results / "zro2_robustness.json"
    sensitivity_plot = results / "zro2_robustness.png"
    sensitivity_trials = results / "zro2_robustness_trials.csv"
    if all(path.is_file() for path in (sensitivity_path, sensitivity_plot, sensitivity_trials)):
        sensitivity = json.loads(sensitivity_path.read_text())
        if sensitivity["checkpoint_sha256"] == report["checkpoint_sha256"]:
            with st.expander("CNN sensitivity to noise and angular sampling"):
                st.caption("Paired synthetic stress tests use the frozen checkpoint on the same held-out patterns. Coarser synthetic traces were reconstructed on the model grid solely for this test.")
                st.image(str(sensitivity_plot))
                st.dataframe(sensitivity["summary"], hide_index=True)
            st.download_button("Download ZrO₂ sensitivity trials", sensitivity_trials.read_bytes(),
                               file_name=sensitivity_trials.name, mime="text/csv")


def render_validation(root, line_plot):
    results = root / "validation/results"
    st.subheader("Computed evidence and open gaps")
    dataset = st.selectbox("Validation dataset", [
        "Synthetic · ZrO₂ GSAS-II CNN", "Synthetic · Lorentzian comparison",
        "Synthetic · Gaussian studies",
        "Experimental · NIST ZnO", "Experimental · original Spinel",
        "Synthetic · published LPA-NN splits"], key="validation_dataset")
    if dataset == "Synthetic · Lorentzian comparison":
        if not (results / "method_comparison.json").exists():
            st.warning("Generate results with python -m validation.method_comparison.")
            return
        report = json.loads((results / "method_comparison.json").read_text())
        trials = pd.read_csv(results / "method_trials.csv")
        st.caption("Synthetic reference sizes are generator inputs. LPA-NN requires a matching full-pattern dataset.")
        st.image(str(results / "method_accuracy.png"))
        factor = st.selectbox("Study factor", ["analytical", *report["config"]["levels"]])
        summary = pd.DataFrame(report["summary"])
        st.dataframe(summary[summary.factor == factor], hide_index=True)
        st.caption("MAE, RMSE and bias use accepted predictions. Failure and rejection rates use all attempts. Rates are fractions.")
        with st.expander("Matched cases accepted by both physics methods"):
            paired = pd.DataFrame(report.get("paired_summary", []))
            selected = paired[paired.factor == factor] if not paired.empty else paired
            if selected.empty:
                st.info("There are no matched successful predictions for this factor.")
            else:
                st.caption("Error metrics for shared accepted trials. Use the main table for failure and rejection rates.")
                st.dataframe(selected.drop(columns=["failure_rate", "input_rejection_rate", "attempts"]),
                             hide_index=True)
        st.image(str(results / "method_robustness.png"))
        example = st.selectbox("Inspect a synthetic example", list(report["examples"]))
        record = report["examples"][example]
        traces = {"Synthetic observations": record["observed"]}
        if record["fit"] is not None:
            traces["Fitted profile"] = record["fit"]
        line_plot(record["x"], traces, "Synthetic Lorentzian benchmark")
        if record["reason"]:
            st.warning(record["reason"])
        with st.expander("Trial details"):
            st.dataframe(trials[trials.factor == factor], hide_index=True)
        st.download_button("Download synthetic trials", (results / "method_trials.csv").read_bytes(),
                           file_name="method_trials.csv", mime="text/csv")
    elif dataset == "Synthetic · ZrO₂ GSAS-II CNN":
        _render_zirconia_benchmark(results)
        return
    elif dataset == "Synthetic · Gaussian studies":
        for stem in ("analytical", "sensitivity"):
            if (results / f"{stem}.json").exists():
                st.json(json.loads((results / f"{stem}.json").read_text()))
                st.image(str(results / f"{stem}.png"), caption=f"Synthetic physics study · {stem}")
        archived_trials = results / "sensitivity_trials.csv"
        if archived_trials.exists():
            st.download_button("Download Gaussian sensitivity trials", archived_trials.read_bytes(),
                               file_name=archived_trials.name, mime="text/csv")
    elif dataset == "Experimental · NIST ZnO":
        if (results / "nist_experimental.json").exists():
            record = json.loads((results / "nist_experimental.json").read_text())
            st.warning(record["primary_conclusion"])
            st.image(str(results / "nist_experimental_fit.png"), caption="Experimental · NIST SRM1979 ZnO")
            with st.expander("Measurement and fit diagnostics"):
                st.json(record)
        st.caption("Scherrer and Williamson–Hall need supported broadening assumptions and calibration. Spinel LPA-NN targets a different phase and instrument.")
    elif dataset == "Synthetic · published LPA-NN splits":
        st.warning("LPA-NN accuracy unavailable. The released Spinel checkpoint lacks a matching labeled held-out split. The published labeled zirconia splits lack matching trained weights.")
        st.caption("Published synthetic zirconia deposit V1: 7,997 test patterns per variant. Compatible labeled Spinel samples: 0. MAE, RMSE and bias are unavailable.")
    else:
        st.warning("The original Spinel traces have no independent size labels. Runtime reproduction does not measure prediction accuracy.")
        original = root / "third_party/LPA-NN/Spinel/data.npz"
        if original.exists():
            import numpy as np

            index = st.slider("Original experimental measurement index", 0, 125, 0)
            with np.load(original, allow_pickle=False) as archive:
                signal = archive["exp_data"][index]
            line_plot(np.linspace(5.8438, 27.0799, 10501, endpoint=False),
                      {"Measured intensity": signal}, "Experimental Spinel · notebook plotting grid")
        with st.expander("Model and dataset evidence"):
            for stem in ("spinel_benchmark_status", "spinel_reproduction", "spinel_upload_reproduction"):
                if (results / f"{stem}.json").exists():
                    st.json(json.loads((results / f"{stem}.json").read_text()))
    st.caption("Reproduce the comparison with python -m validation.method_comparison.")
