"""Automatic uploaded-pattern analysis around the existing scientific modules."""
import hashlib
import json
import logging
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ui.measurement import require_physics_metadata
from ui.pattern_chart import render_pattern_chart
from xrd.batch_analysis import analyze_peaks
from xrd.peak_detection import detect_peaks

LOG=logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def _detect(x,y,prominence,smoothing,window,separation,maximum):
    return detect_peaks(x,y,prominence,smoothing=smoothing,smoothing_window=window,
                        min_separation_deg=separation,max_peaks=maximum)


def instrument_definition(conditions):
    mode=conditions["instrument_mode"]
    if mode=="Unknown":
        return {"mode":"unknown"}
    calibrated=mode in {"Calibrated local reference width","Calibrated width vs angle"}
    if calibrated and not (conditions["reference"].strip() and conditions["calibration_confirmed"]):
        raise ValueError("Complete the calibration reference and matching-condition confirmation before correcting widths.")
    info={"mode":"table" if mode=="Calibrated width vs angle" else "constant",
          "profile":conditions["profile"],"calibrated":calibrated,"reference":conditions["reference"]}
    if info["mode"]=="table":
        if not conditions.get("calibration_table"):
            raise ValueError("Provide an instrumental-width calibration table covering your reflections.")
        info.update(conditions["calibration_table"])
    else:
        info["fwhm_deg"]=conditions["instrument"]
    return info


def peak_table(result):
    rows=[]
    for row in result["peaks"]:
        fit=row["fit"]
        rows.append({"Candidate 2theta (deg)":row["candidate_deg"],
            "Fitted 2theta (deg)":row["center_deg"],"Observed FWHM (deg)":row["observed_fwhm_deg"],
            "Instrument FWHM (deg)":row["instrument_fwhm_deg"],"Sample FWHM (deg)":row["sample_fwhm_deg"],
            "Scherrer size (nm)":row["size_nm"],"Fit R²":None if fit is None else fit.r_squared,
            "Residual RMSE / amplitude":None if fit is None else fit.normalized_rmse,
            "Status":row["status"],"Warnings":"; ".join(row["warnings"])})
    return pd.DataFrame(rows)


def _pattern_plot(data,candidates,chosen,baseline):
    pattern=data["pattern"];x,y=pattern.two_theta_deg,pattern.intensity
    fig,ax=plt.subplots(figsize=(9,3.1))
    ax.plot(x,y,lw=.8,label="Uploaded intensity")
    ax.plot(x,baseline,lw=.8,alpha=.8,label="Detection background")
    if len(candidates):
        positions=np.array([np.argmin(abs(x-c)) for c in candidates])
        ax.scatter(x[positions],y[positions],marker="x",s=30,color="gray",label="Detected candidates")
    if len(chosen):
        positions=np.array([np.argmin(abs(x-c)) for c in chosen])
        ax.scatter(x[positions],y[positions],marker="o",s=32,facecolors="none",edgecolors="red",label="Selected for fitting")
    ax.set(xlabel="2θ (degrees)",ylabel="Intensity (a.u.)")
    ax.legend(fontsize=8);fig.tight_layout();st.pyplot(fig);plt.close(fig)
    st.caption("Data source: "+data["name"]+" · background/smoothing are for detection only; widths are fitted to raw supplied intensities.")
    st.caption(f"{len(x):,} supplied points · {x[0]:.4f}–{x[-1]:.4f}° · median sampling interval {np.median(np.diff(x)):.6g}° · {len(candidates)} detected candidates")


def _details(result,data,conditions,metadata,line_plot):
    with st.expander("Individual peak estimates and complete diagnostics"):
        table=peak_table(result)
        st.dataframe(table,hide_index=True,width="stretch")
        st.download_button("Download peak results",table.to_csv(index=False),file_name="xrd_peak_results.csv",mime="text/csv")
        if result["summary"]["count"]:
            s=result["summary"]
            variation = f"sample standard deviation {s['std_nm']:.3f} nm" if s['count'] > 1 else "across-peak variation unavailable with one reflection"
            st.write(f"Across {s['count']} accepted reflections: mean {s['mean_nm']:.3f} nm; {variation}; range {s['min_nm']:.3f}–{s['max_nm']:.3f} nm.")
            st.caption(s["interpretation"])
        fitted=[row for row in result["peaks"] if row["fit"] is not None]
        if fitted:
            index=st.selectbox("Inspect fitted reflection",range(len(fitted)),format_func=lambda i:f"{fitted[i]['center_deg']:.4f}° · {fitted[i]['status']}")
            row=fitted[index];fit=row["fit"]
            line_plot(fit.x,{"Observed":fit.observed,"Peak + local linear background":fit.fitted},data["name"])
            line_plot(fit.x,{"Observed − fitted":fit.residual},data["name"],ylabel="Residual (a.u.)")
            st.write("Fit interval (2theta degrees):",row["bounds_deg"])
            for warning in row["warnings"]:st.warning(warning)
        for note in result["warnings"]:st.write("• "+note)
        st.json({"measurement":metadata,"profile":conditions["profile"],"K":conditions["k"],
                 "breadth_definition":result["breadth_definition"],"instrument_corrected":result["instrument_corrected"]})
    wh=result["wh"]
    if wh["result"] is not None:
        with st.expander("Williamson–Hall plot and regression diagnostics",expanded=wh["status"]=="available"):
            r=wh["result"]
            fig,axes=plt.subplots(2,1,figsize=(8,4.7),layout="constrained")
            axes[0].scatter(r["x"],r["y"],label="Corrected fitted reflections")
            order=np.argsort(r["x"])
            axes[0].plot(np.asarray(r["x"])[order],np.asarray(r["fitted"])[order],label="Linear regression")
            axes[0].set(xlabel="4 sin θ (dimensionless)",ylabel="βsample cos θ (radians)")
            axes[0].legend(fontsize=8)
            axes[1].scatter(r["x"],r["residuals"]);axes[1].axhline(0,color="black",lw=.7)
            axes[1].set(xlabel="4 sin θ (dimensionless)",ylabel="Regression residual (radians)")
            st.pyplot(fig);plt.close(fig)
            st.caption(f"Data source: {data['name']} · {r['n_reflections']} accepted reflections · R² {r['r_squared']:.5f}")
            st.dataframe(pd.DataFrame({"4 sin theta":r["x"],"beta cos theta (rad)":r["y"],"fitted (rad)":r["fitted"],"residual (rad)":r["residuals"]}),hide_index=True,width="stretch")
            for note in r["warnings"]:st.warning(note)
            st.json(r["diagnostics"])


def render_multi_peak(data,conditions,line_plot):
    # Lazy import preserves the detailed analyzer and its integration metadata.
    from .analyze import (clear_result, back_to_conditions, change_data, workflow_metadata,
                          _ml_details, render_gsas_status)
    metadata=workflow_metadata(conditions)
    pattern=data["pattern"];x,y=pattern.two_theta_deg,pattern.intensity
    research = st.session_state.get("show_research_tools", False)
    st.subheader("3 · Detect, fit and analyze")
    previous_step = st.empty()
    example = data.get("example")
    if example:
        centers = data["candidates"]
        chosen = centers.tolist()
        halfwidth = example["fit_window_half_width_deg"]
        isolated = True
        same_phase = example["profile"] == "Lorentzian"
        settings = {"centers": chosen, "halfwidth": halfwidth, "isolated": isolated, "same_phase": same_phase}
        st.caption(f"{len(chosen)} simulated reflections · {example['profile']} profile · "
                   f"{example['instrument_fwhm_deg']:g}° simulated instrument FWHM")
        if same_phase:
            st.caption("Open Research tools for the Williamson–Hall size and microstrain result.")
        with st.expander("Inspect the pattern and selected reflections"):
            _pattern_plot(data, centers, chosen, data["baseline"])
    else:
        st.caption(f"{len(x):,} uploaded points · {x[0]:.4f}–{x[-1]:.4f}° · median sampling {np.median(np.diff(x)):.6g}°")
        with st.expander("Advanced: peak detection, selection and fit windows"):
            prominence=st.slider("Peak prominence (% of corrected intensity range)",.5,50.,10.,step=.5)/100
            separation=st.number_input("Minimum peak separation (2theta degrees)",min_value=.001,max_value=30.,value=.2,step=.05,format="%.3f")
            smoothing=st.checkbox("Smooth only the detection signal (Savitzky–Golay)")
            window=int(st.number_input("Detection smoothing window (odd sample count)",min_value=5,max_value=101,value=11,step=2)) if smoothing else 11
            maximum=int(st.number_input("Maximum candidate peaks",min_value=1,max_value=100,value=30))
            manual=st.checkbox("Override automatic fit-window half width")
            halfwidth=st.number_input("Fit-window half width (degrees)",min_value=.01,max_value=20.,value=1.2,step=.05) if manual else None
            st.caption("Detection removes an estimated background. A local linear background is fitted jointly with each raw peak. Numerical checks can reject overlap, poor sampling and truncated windows; a local maximum is not phase identification.")
        try:
            centers,baseline=_detect(x,y,prominence,smoothing,window,separation,maximum)
        except (ValueError,RuntimeError) as exc:
            st.error(f"Peak detection needs your attention: {exc}. Review the smoothing window and detection settings.")
            return None
        with st.expander("Select detected reflections"):
            chosen=st.multiselect("Candidate peaks to include",[float(c) for c in centers],default=[float(c) for c in centers],
                format_func=lambda c:f"{c:.4f}°",key="multi_selection_"+hashlib.sha256(np.asarray(centers).tobytes()+data["identity"].encode()).hexdigest())
            st.caption("Exclude unassigned phases, overlap, doublets and artifacts after inspecting the pattern and the fit residuals. Same-phase membership must be established independently for W-H.")
        with st.expander("Inspect the pattern and selected reflections"):
            _pattern_plot(data,centers,chosen,baseline)
        confirmation_id=hashlib.sha256((data["identity"]+repr(chosen)+repr(halfwidth)+conditions["profile"]).encode()).hexdigest()
        isolated=st.checkbox("I inspected the selected candidates as isolated reflections without unresolved spectral doublets",key="multi_isolated_"+confirmation_id)
        same_phase = False
        if research:
            same_phase=st.checkbox("Selected reflections are independent, from one identified phase, and suitable for isotropic additive Lorentzian size/strain analysis",key="multi_phase_"+confirmation_id)
            st.caption("Phase confirmation enables W-H only with a Lorentzian profile, explicit instrument correction, and at least three usable well-distributed reflections. Unknown strain gives apparent Scherrer estimates.")
        settings={"centers":chosen,"prominence":prominence,"separation":separation,"smoothing":smoothing,
                  "smoothing_window":window,"maximum":maximum,"halfwidth":halfwidth,"isolated":isolated,"same_phase":same_phase}
    identity=hashlib.sha256((data["identity"]+json.dumps(conditions,sort_keys=True)+json.dumps(settings,sort_keys=True)).encode()).hexdigest()
    saved=st.session_state.get("multi_result")
    if saved and saved["identity"]!=identity:
        clear_result();saved=None
        st.info("Previous results cleared because the input, conditions, selection or detection settings changed.")
    if st.button("Analyze",type="primary",disabled=not len(chosen)):
        clear_result();started=perf_counter()
        with st.status("Analyzing uploaded peaks — fitting and correcting breadths…",expanded=True) as status:
            feedback_ms=(perf_counter()-started)*1000
            try:
                require_physics_metadata(metadata)
                if not conditions.get("instrument_reviewed"):
                    raise ValueError("Confirm the instrument information in Confirm conditions first.")
                result=analyze_peaks(pattern,chosen,metadata["wavelength_nm"],profile=conditions["profile"],
                    shape_factor=conditions["k"],instrument=instrument_definition(conditions),
                    same_phase_confirmed=same_phase,size_dominated=conditions["size_dominated"],
                    isolated_confirmed=isolated,window_half_width=halfwidth)
                saved={"identity":identity,"result":result,"elapsed_s":perf_counter()-started,"status_requested_ms":feedback_ms,
                       "render_token":int(perf_counter()*1_000_000_000)}
                st.session_state["multi_result"]=saved
                st.session_state["multi_scroll_result"]=True
                status.update(label="Peak analysis complete",state="complete",expanded=False)
            except (ValueError,TypeError,RuntimeError,FloatingPointError) as exc:
                status.update(label="Analysis needs your attention",state="error",expanded=False)
                st.error(f"A defensible analysis could not be completed: {exc}. Review the conditions and selected reflections.")
                saved=None
            except Exception:
                LOG.exception("Automatic peak analysis stopped")
                status.update(label="Analysis needs your attention",state="error",expanded=False)
                st.error("The numerical analysis stopped unexpectedly. Review individual windows in the detailed single-reflection mode; no size is reported for this run.")
                saved=None
    if not len(chosen):
        st.warning("No selected peak candidates. Lower detection prominence only if real reflections are visible, review the background, or choose detected reflections to include.")
    result=None
    if saved:
        result=saved["result"];s=result["summary"];wh=result["wh"];r=wh["result"]
        st.subheader("Crystallite size results")
        st.button("Back to conditions", on_click=back_to_conditions)
        size_card, chart = st.columns([1, 1.9], vertical_alignment="center")
        if s["count"]:
            size_card.metric("Scherrer crystallite size (median)", f"{s['median_nm']:.2f} nm")
        else:
            size_card.warning("No justified size estimate.")
        with chart:
            render_pattern_chart(pattern, [row["center_deg"] for row in result["peaks"]
                                           if row["size_nm"] is not None])
        st.markdown("**Coherent-domain length, not grain or particle size.**")
        st.caption(f"Method: Scherrer · {conditions['profile']} FWHM · K = {conditions['k']:g} · λ = {metadata['wavelength_nm']:.6g} nm")
        if s["count"]:
            variation = f"sample standard deviation {s['std_nm']:.2f} nm" if s['count'] > 1 else "across-peak variation unavailable with one reflection"
            st.caption(f"{s['count']} of {len(result['peaks'])} selected candidates accepted · range {s['min_nm']:.2f}–{s['max_nm']:.2f} nm · {variation}")
            st.caption("Across-peak variation is descriptive; it is not a confidence interval or evidence that the domains are isotropic.")
        else:
            st.caption("Peak widths and individual rejection reasons remain available in the diagnostics.")
            if not isolated:
                st.error("Isolation/no-doublet confirmation is missing. Inspect the selected candidates and confirm only those that are physically justified; fitting a local maximum alone does not validate a reflection.")
            else:
                reasons=[note for row in result["peaks"] for note in row["warnings"]]
                if reasons:st.error("Peak analysis needs review: "+reasons[0]+" Next: inspect the individual fit, calibration and selected window, or use the detailed single-reflection mode.")
        st.subheader(result["category"])
        if result["instrument"]["mode"]=="unknown":
            st.warning("Instrument width is unknown. Scherrer values are uncorrected apparent sizes, not experimentally calibrated sizes.")
        elif not result["instrument"]["calibrated"]:
            st.warning("Instrument correction uses an explicitly assumed width, not a measured calibration. Any Scherrer or W-H result is exploratory.")
        if not conditions["size_dominated"]:
            st.warning("Apparent Scherrer sizes attribute all remaining breadth to finite size. Strain/defects have not been separated and can bias these estimates.")
        if s["count"]>1 and not same_phase:
            st.warning("The Scherrer median pools selected reflections without a confirmed common phase. It is a descriptive summary, not an established single-phase material size.")
        if research and wh["status"]=="available" and r is not None and r["size_nm"] is not None:
            st.metric("Williamson–Hall crystallite size",f"{r['size_nm']:.2f} nm")
            st.metric("Microstrain (dimensionless)",f"{r['microstrain']:.6g}")
            st.caption(f"ε = {r['microstrain']*100:.5g}% · {r['n_reflections']} corrected reflections · regression R² {r['r_squared']:.5f}. Microstrain is not residual stress.")
            for note in r["warnings"]:st.warning(note)
        elif research:
            st.write("**Williamson–Hall crystallite size: withheld · Microstrain: undetermined**")
            st.info(wh["reason"]+" Next: review the matching instrument correction, reflection count, phase identity, profile and angular distribution.")
        if s["count"]<len(result["peaks"]):
            st.warning(f"{len(result['peaks'])-s['count']} selected candidates did not produce a size; their fit/correction reasons are retained below.")
        st.caption(f"Data source: {data['name']} · {saved['elapsed_s']:.3f} s analysis computation")
        render_gsas_status()
    else:
        previous_step.button("Back to conditions", on_click=back_to_conditions)
    st.button("Change data",on_click=change_data)
    if saved:
        _details(result,data,conditions,metadata,line_plot)
    _ml_details(pattern,metadata)
    if st.session_state.pop("multi_scroll_result",False):
        st.html("""<script>/* multi result TOKEN */
        requestAnimationFrame(() => requestAnimationFrame(() => {
          const h = document.getElementById('crystallite-size-results');
          const m = document.querySelector('[data-testid="stMain"]');
          if (h && m) m.scrollBy({top: h.getBoundingClientRect().top - 80, behavior:'instant'});
        }));</script>""".replace("TOKEN",str(saved["render_token"])),unsafe_allow_javascript=True)
    if result is not None:
        return {**result,"size_nm":result["summary"]["median_nm"]}
    return None
