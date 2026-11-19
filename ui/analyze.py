"""Three-step workflow for uploaded and example XRD patterns."""
from contextlib import nullcontext
import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import streamlit as st
from xrd.io import read_pattern
from xrd.peak_detection import detect_peaks
from xrd.analysis import analyze_window
from ml.inference import predict_crystallite_size
from .measurement import measurement_metadata, require_physics_metadata
from .feedback import show_problem, error_kind
from xrd.instrument_io import read_width_calibration
from xrd import ceria
from . import ceria as ceria_ui

LOG = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def prepare_pattern(content, angle_unit, columns, source):
    pattern = read_pattern(content, angle_unit, columns, source=source)
    candidates, baseline = detect_peaks(pattern.two_theta_deg, pattern.intensity)
    if not len(candidates):
        candidates = np.array([pattern.two_theta_deg[np.argmax(pattern.intensity)]])
    return pattern, candidates, baseline


def _stage(number):
    st.session_state["workflow_stage"] = number
    st.session_state.pop("workflow_problem", None)


def clear_result():
    st.session_state.pop("workflow_result", None)
    st.session_state.pop("multi_result", None)
    st.session_state.pop("multi_scroll_result", None)
    st.session_state.pop("comparison_result", None)
    st.session_state.pop("workflow_scroll_result", None)
    st.session_state.pop("analysis", None)  # Remove pre-wizard cached results too.
    st.session_state.pop("ceria_result", None)


def change_data():
    clear_result()
    st.session_state.pop("workflow_data", None)
    st.session_state.pop("workflow_conditions", None)
    st.session_state.pop("workflow_peak", None)
    _stage(1)


def back_to_conditions():
    clear_result()
    _stage(2)


def _defaults(example=None):
    conditions = {"material":"", "wl_value":1.5406,
            "wl_unit":"angstrom", "verified":False, "geometry":"Unknown / not declared",
            "profile":"Gaussian", "instrument_mode":"Unknown", "instrument":.1,
            "reference":"", "calibration_confirmed":False, "k":.9,
            "size_dominated":False,"instrument_reviewed":False,"calibration_table":None}
    if example is not None:
        conditions.update(wl_value=example["wavelength_nm"], wl_unit="nm",
                          geometry="Synthetic powder-like isolated peak", profile=example["profile"],
                          instrument_mode="Declared simplified constant width",
                          instrument=example["instrument_fwhm_deg"], k=example["shape_factor"],
                          size_dominated=example["microstrain"] == 0, instrument_reviewed=True)
    return conditions


def _load(content, name, angle_unit, columns, *, example=None):
    clear_result()
    teaching = example is not None
    try:
        matched_ceria = not teaching and ceria.is_sample(content)
        if matched_ceria:
            if angle_unit != "degrees" or columns not in (None, (0, 1)):
                raise ValueError("The published BM16 .xy uses degrees 2theta in column 0 and counts in column 1. "
                                 "Select degrees and use automatic column selection for this file.")
            columns = (0, 1)
        with st.status("Loading pattern and finding candidate reflections…", expanded=True) as status:
            pattern, candidates, baseline = prepare_pattern(content, angle_unit, columns, name)
            if teaching:
                candidates = np.array([example["two_theta_center_deg"]])
            status.update(label="Data loaded", state="complete", expanded=False)
        fingerprint = hashlib.sha256(content + repr((name, angle_unit, columns, teaching)).encode()).hexdigest()
        st.session_state["workflow_data"] = {"pattern":pattern, "candidates":candidates,
            "baseline":baseline, "identity":fingerprint, "teaching":teaching,
            "angle_unit":angle_unit, "name":name}
        if matched_ceria:
            st.session_state["workflow_data"]["ceria_content"] = content
        st.session_state["workflow_conditions"] = _defaults(example)
        st.session_state.pop("workflow_peak", None)
        st.session_state.pop("analysis_mode", None)
        _stage(2)
        st.rerun()
    except (ValueError, UnicodeError, TypeError, OverflowError) as exc:
        st.session_state["workflow_problem"] = ("file", str(exc))
    except Exception:
        LOG.exception("Pattern loading failed")
        st.session_state["workflow_problem"] = ("file", "The plain-text reader could not interpret this file. Try a two-column text export.")


def _load_step(root):
    st.subheader("1 · Load data")
    if loaded := st.session_state.get("workflow_data"):
        st.caption(f"Loaded data · {loaded['name']} · 2θ in {loaded['angle_unit']}")
        st.button("Continue with loaded data", on_click=_stage, args=(2,))
    st.write("Drag a CSV, TXT, XY or DAT file into the upload area, or try the synthetic example.")
    mode = st.radio("Data source", ["Synthetic teaching example", "Upload measurement"], index=1, horizontal=True)
    if st.button("Load synthetic example", type="primary" if mode=="Synthetic teaching example" else "secondary"):
        sample = root / "examples" / "synthetic_40nm_gaussian.csv"
        example = json.loads(sample.with_suffix(".json").read_text())
        _load(sample.read_bytes(), "Synthetic 40 nm teaching example · seed 534",
              "degrees", None, example=example)
    st.caption("Synthetic example: a known 40 nm isolated peak, not experimental data or an identified material.")
    file = st.file_uploader("2theta and intensity", type=["csv","txt","xy","dat"], disabled=mode!="Upload measurement")
    angle_unit = st.selectbox("Uploaded 2theta unit", ["degrees","radians"])
    st.caption("Headers identify angle/intensity columns when unambiguous; otherwise choose columns. Increasing angles and no missing values are required. Confirm units and geometry in step 2.")
    with st.expander("Column selection (only for files with extra columns)"):
        explicit = st.checkbox("Choose columns explicitly (files with more than two columns)")
        columns = None
        if explicit:
            a, b = st.columns(2)
            columns = (int(a.number_input("Angle column (zero based)",0,20,0)),
                       int(b.number_input("Intensity column (zero based)",0,20,1)))
    if mode == "Upload measurement":
        if st.button("Load uploaded data", type="primary"):
            if file is None:
                st.session_state["workflow_problem"] = ("missing_file", "")
            elif Path(file.name).suffix.lower() not in {".csv",".txt",".xy",".dat"}:
                st.session_state["workflow_problem"] = ("file", "Use CSV, TXT, XY or DAT, not a binary workbook or image.")
            else:
                _load(file.getvalue(), file.name, angle_unit, columns)
    if st.session_state.get("workflow_problem"):
        show_problem(*st.session_state["workflow_problem"])


def _measurement_fields(conditions, teaching, verified):
    material = st.text_input("Material / sample description", value=conditions["material"],
                            placeholder="Optional; leave blank if unknown", key="sample_material")
    a, b = st.columns([2,1])
    prefix = "demo" if teaching else "upload"
    wl_value = a.number_input("Wavelength", min_value=.0001, value=conditions["wl_value"], format="%.6f", key=prefix+"_wavelength")
    units = ["nm","angstrom","pm"]
    wl_unit = b.selectbox("Wavelength unit", units, index=units.index(conditions["wl_unit"]), key=prefix+"_wavelength_unit")
    if teaching:
        geometry = "Synthetic powder-like isolated peak"
        st.caption("Geometry · Synthetic powder-like isolated reflection")
    else:
        geometries = ["Unknown / not declared","Powder θ–2θ scan","Integrated powder Debye–Scherrer rings","Powder capillary scan (analyzer detector)","Other (requires method review)"]
        geometry = st.selectbox("Measurement geometry", geometries, index=geometries.index(conditions["geometry"]), key="upload_geometry")
        st.caption("Choose how the instrument collected the scan, such as a powder θ–2θ scan or a "
                   "capillary scan. Use the instrument record or the conditions JSON supplied with a trial file.")
        st.caption("The wavelength starts with an editable Cu Kα example. Verify the actual wavelength "
                   "and how the source doublet was handled.")
        verified = st.checkbox("I verified the wavelength and angular units for this measurement",
                               value=conditions["verified"], key="upload_wavelength_verified")
    profile = st.selectbox("Pure peak profile", ["Gaussian","Lorentzian"], index=["Gaussian","Lorentzian"].index(conditions["profile"]))
    st.caption("Use a profile supported by your reflection and reference. Mixed Voigt profiles require another method.")
    modes = ["Unknown", "Declared simplified constant width"]
    if not teaching:
        modes.extend(["Calibrated local reference width", "Calibrated width vs angle"])
    prior_mode = conditions["instrument_mode"]
    instrument_mode = st.selectbox("Instrument broadening information", modes,
                                   index=modes.index(prior_mode) if prior_mode in modes else 0)
    instrument, reference, confirmed = conditions["instrument"], conditions["reference"], conditions["calibration_confirmed"]
    calibration_table = conditions.get("calibration_table")
    calibration_error = None
    if instrument_mode in {"Declared simplified constant width","Calibrated local reference width"}:
        instrument = st.number_input("Instrument FWHM at this reflection (2theta degrees)",min_value=0.,value=instrument,step=.01,format="%.4f")
        st.caption("For multiple peaks, a constant width must be justified across their angular range. Use a width-versus-angle table if the instrument response varies with angle.")
    elif instrument_mode == "Calibrated width vs angle":
        def clear_removed_calibration():
            if st.session_state.get("width_calibration_file") is None:
                conditions["calibration_table"] = None

        calibration = st.file_uploader("Instrument width calibration: 2theta degrees, FWHM degrees",
                                       type=["csv","txt","xy","dat"],key="width_calibration_file",
                                       on_change=clear_removed_calibration)
        st.caption("Upload fitted instrumental widths after removing intrinsic reference broadening, not a raw intensity pattern. Width interpolation is bounded; there is no extrapolation.")
        if calibration is not None:
            try:
                calibration_table = read_width_calibration(calibration.getvalue())
                st.caption(f"{len(calibration_table['angles_deg'])} calibration points, covering {calibration_table['angles_deg'][0]:g}–{calibration_table['angles_deg'][-1]:g}°")
            except (ValueError,UnicodeError) as exc:
                calibration_error = str(exc)
                calibration_table = None
                st.error(calibration_error)
        elif calibration_table is not None:
            st.caption("Using the calibration table previously loaded for this pattern.")
            if st.button("Remove retained calibration"):
                calibration_table = None
    else:
        st.caption("Without calibration, a size can only be an apparent preliminary estimate.")
    if instrument_mode in {"Calibrated local reference width","Calibrated width vs angle"}:
        reference = st.text_input("Reference identifier / calibration measurement",value=reference)
        confirmed = st.checkbox("Calibration uses matching geometry, wavelength, selected angular range and the same pure profile; reference intrinsic width is negligible or removed",value=confirmed)
    reviewed = st.checkbox("I reviewed the instrument information and profile choice above, including any unknown width",value=conditions.get("instrument_reviewed",False),key="instrument_reviewed")
    size_dominated = st.checkbox("Assume corrected sample broadening is size dominated (negligible strain/defects)",value=conditions["size_dominated"])
    st.caption("Automatic mode can report apparent Scherrer sizes when strain is unknown. These attribute all remaining breadth to size; the detailed single-reflection mode retains the stricter size-withholding option.")
    with st.expander("Advanced: Scherrer factor and physical assumptions"):
        k = st.number_input("FWHM Scherrer shape factor K",min_value=.1,max_value=2.,value=conditions["k"],step=.01)
        st.write("K = 0.9 is a stated FWHM convention, not a universal shape constant. Size is coherent-domain length, not grain size. Instrument correction requires matching pure profiles.")
    with st.expander("Instrument identity and phase information"):
        instrument_name = st.text_input("Instrument / beamline", value=conditions.get("instrument_name", ""),
                                        placeholder="Optional", key="measurement_instrument")
        phases = st.text_input("Crystalline phases (comma separated)", value=conditions.get("phases", ""),
                               placeholder="Only if independently identified", key="measurement_phases")
    conditions.update(material=material,wl_value=wl_value,wl_unit=wl_unit,verified=verified,
                      geometry=geometry,profile=profile,instrument_mode=instrument_mode,
                      instrument=instrument,reference=reference,calibration_confirmed=confirmed,
                      k=k,size_dominated=size_dominated,instrument_name=instrument_name,phases=phases,
                      instrument_reviewed=reviewed,calibration_table=calibration_table)
    return calibration_error


def _conditions_step(data):
    st.subheader("2 · Confirm conditions")
    st.button("Back to data", on_click=_stage, args=(1,))
    st.caption(f"Source: {data['name']} · 2θ in {data['angle_unit']} · no interpolation")
    conditions = st.session_state["workflow_conditions"]
    teaching = data["teaching"]
    verified = None
    if teaching:
        st.info("The built-in sample has one Gaussian reflection near 40° 2θ, wavelength 0.15406 nm, "
                "simulated instrument FWHM 0.10°, zero simulated microstrain, and a Scherrer factor of 0.9.")
        verified = st.checkbox("I reviewed the synthetic example settings",
                               value=conditions["verified"], key="demo_wavelength_verified")
    with st.expander("Optional · change sample settings") if teaching else nullcontext():
        calibration_error = _measurement_fields(conditions, teaching, verified)
    metadata = workflow_metadata(conditions)
    if st.button("Continue to analysis",type="primary"):
        clear_result()
        try:
            require_physics_metadata(metadata)
            mode = conditions["instrument_mode"]
            if not conditions["instrument_reviewed"]:
                st.error("Review the instrument width and pure-profile choice, then tick the instrument confirmation. Unknown width is allowed for explicitly uncorrected exploratory analysis.")
            elif mode in {"Calibrated local reference width","Calibrated width vs angle"} and not (
                conditions["reference"].strip() and conditions["calibration_confirmed"]
            ):
                show_problem("calibration")
            elif mode=="Calibrated width vs angle" and (not conditions["calibration_table"] or calibration_error):
                st.error("Upload a valid two-column instrumental-width table covering the reflections, or select Unknown for uncorrected Scherrer exploration.")
            else:
                _stage(3)
                st.rerun()
        except ValueError as exc:
            show_problem(error_kind(exc),str(exc))
    return metadata


def workflow_metadata(conditions):
    metadata = measurement_metadata(material=conditions["material"],wavelength_value=conditions["wl_value"],
        wavelength_unit=conditions["wl_unit"],geometry=conditions["geometry"],wavelength_verified=conditions["verified"])
    mode = conditions["instrument_mode"]
    return {
        **metadata,
        "instrument": conditions.get("instrument_name", "").strip(),
        "phases": [phase.strip() for phase in conditions.get("phases", "").split(",") if phase.strip()],
        "scherrer_profile": conditions["profile"],
        "scherrer_instrument_mode": mode,
        "scherrer_instrument_fwhm_deg": conditions["instrument"] if mode in {
            "Declared simplified constant width", "Calibrated local reference width"
        } else None,
        "scherrer_reference": conditions["reference"] if mode.startswith("Calibrated") else "",
        "scherrer_calibration_table": conditions.get("calibration_table") if mode == "Calibrated width vs angle" else None,
        "scherrer_calibrated": mode.startswith("Calibrated") and conditions["calibration_confirmed"],
    }


def analysis_identity(data, conditions, peak):
    return hashlib.sha256((data["identity"]+json.dumps(conditions,sort_keys=True)+repr(peak)).encode()).hexdigest()


def execute_analysis(data, conditions, lower, upper, isolated, *, metadata=None):
    """Thin, timed wrapper around the unchanged analysis engine."""
    if metadata is None:
        metadata = workflow_metadata(conditions)
    require_physics_metadata(metadata)
    if conditions["instrument_mode"]=="Calibrated width vs angle":
        raise ValueError("Use Automatic multi-peak mode to apply a width-versus-angle table. Detailed single-reflection analysis accepts a verified local constant width.")
    calibrated = conditions["instrument_mode"]=="Calibrated local reference width"
    if calibrated and not (conditions["reference"].strip() and conditions["calibration_confirmed"]):
        raise ValueError("Instrument calibration reference and matching-condition confirmation are required.")
    return analyze_window(data["pattern"], lower, upper, metadata["wavelength_nm"],
        conditions["profile"], None if conditions["instrument_mode"]=="Unknown" else conditions["instrument"],
        conditions["k"], conditions["size_dominated"], isolated, calibrated)


def _ml_details(pattern, metadata):
    if not st.session_state.get("show_research_tools", False):
        return
    result = predict_crystallite_size(pattern, metadata)
    st.caption("LPA-NN is unavailable for this input: validation/calibration evidence is incomplete, and different wavelengths or phases are unsupported. Use physics analysis only under its stated assumptions.")
    with st.expander("ML applicability and complete diagnostics"):
        st.caption("Edit the instrument identity and identified phases in Confirm conditions.")
        st.write(result.input_compatibility_status)
        st.dataframe(pd.DataFrame(result.compatibility_checks),hide_index=True,width="stretch")
        st.button("Predict with original LPA-NN",disabled=True)
        with st.expander("Structured ML result"):
            st.json(result.to_dict())


def render_gsas_status():
    st.info("GSAS-II refinement is unavailable for this upload. A full-pattern comparison "
            "requires a confirmed phase structure (CIF), a matching calibrated instrument "
            "profile (.instprm), and a verified GSAS-II recipe. Try the published BM16 "
            "ceria pattern to see both methods on one page.")


def _result(data, conditions, saved, line_plot):
    result = saved["result"]
    st.subheader("3 · Result")
    st.button("Back to conditions", on_click=back_to_conditions)
    st.success("Analysis complete")
    st.markdown("**Coherent-domain size** — the length over which the crystal diffracts coherently; this is not grain size.")
    if result["size_nm"] is not None:
        st.metric("Scherrer size",f"{result['size_nm']:.2f} nm")
    else:
        st.warning("No justified size estimate. The available information or peak fit does not support one.")
    method = "Scherrer" if result["size_nm"] is not None else "Peak fit; size withheld"
    st.write(f"**Method:** {method} · {conditions['profile']} FWHM")
    st.subheader(result["category"])
    st.caption("Data source: "+data["name"])
    render_gsas_status()
    # Critical limitations remain beside the size, never only in details.
    if result["size_nm"] is not None:
        if conditions["instrument_mode"]=="Unknown":
            st.warning("Important limit: instrumental broadening is unknown. This is an apparent, uncorrected size.")
        elif conditions["instrument_mode"]=="Declared simplified constant width":
            st.warning("Important limit: instrument width is an assumed constant, not a measured calibration.")
        st.caption("Size assumes negligible strain/defect broadening, an isolated reflection, and the selected profile/K convention.")
    else:
        show_problem("fit" if not result["fit"].usable else "size_evidence", "; ".join(result["warnings"]))
    fit = result["fit"]
    for note in result["warnings"]:
        if "Instrument-dominated" in note or "Negative intensities" in note or "Uneven sampling" in note:
            st.warning(note)
    st.caption(f"Fit R² {fit.r_squared:.5f} · FWHM {fit.fwhm_deg:.5f}° on 2θ · {saved['elapsed_s']:.3f} s computation")
    if st.button("Analyze again",type="primary"):
        clear_result()
        st.rerun()
    st.button("Change data",on_click=change_data)
    with st.expander("Peak fit, residuals and numerical details"):
        line_plot(fit.x,{"Observed":fit.observed,"Joint peak + linear baseline fit":fit.fitted},data["name"])
        line_plot(fit.x,{"Observed − fitted":fit.residual},data["name"],ylabel="Residual (a.u.)")
        st.write("All scientific warnings:")
        for warning in result["warnings"]: st.write("• "+warning)
        st.json({"fit":fit.summary(),"measurement_metadata":workflow_metadata(conditions),
                 "instrument_corrected":result["instrument_corrected"],
                 "sample_fwhm_deg":result["sample_fwhm_deg"],"K":conditions["k"]})
    _ml_details(data["pattern"],workflow_metadata(conditions))
    if st.session_state.pop("workflow_scroll_result",False):
        # Native reruns preserve the old plot/button scroll position. This
        # constant-only script positions the new result below the fixed header.
        # Never interpolate uploaded text or measurement values into this HTML.
        st.html("""<script>/* result RUN_TOKEN */
        requestAnimationFrame(() => requestAnimationFrame(() => {
          const heading = document.getElementById('3-result');
          const main = document.querySelector('[data-testid="stMain"]');
          if (heading && main) main.scrollBy({top: heading.getBoundingClientRect().top - 80, behavior: 'instant'});
        }));
        </script>""".replace("RUN_TOKEN",str(saved["render_token"])),unsafe_allow_javascript=True)
    return result


def _reflection_controls(data, peak, line_plot):
    x, y = data["pattern"].two_theta_deg, data["pattern"].intensity
    teaching = data["teaching"]
    candidates = data["candidates"]
    labels = [f"{position:.4f}°" for position in candidates]
    index = st.selectbox("Reflection to analyze",range(len(labels)),format_func=lambda i:labels[i],index=peak[3] if peak else 0)
    center = float(candidates[index])
    nominal_window = (float(max(x[0],center-1.2)),float(min(x[-1],center+1.2)))
    defaults = nominal_window
    if peak and peak[3]==index:
        defaults = peak[:2]
    lower, upper = st.slider("Fit window (2theta degrees)",float(x[0]),float(x[-1]),defaults,step=float(max(.0001,np.median(np.diff(x)))))
    with st.expander("Advanced: exact fit window"):
        exact = st.checkbox("Enter exact fit bounds",key=f"exact_window_{data['identity']}_{index}")
        if exact:
            st.caption("These numeric bounds override the slider. Keep one isolated reflection and enough background on both sides.")
            a, b = st.columns(2)
            lower = a.number_input("Lower fit bound (2theta degrees)",min_value=float(x[0]),max_value=float(x[-1]),
                value=float(lower),format="%.4f",key=f"exact_lower_{data['identity']}_{index}")
            upper = b.number_input("Upper fit bound (2theta degrees)",min_value=float(x[0]),max_value=float(x[-1]),
                value=float(upper),format="%.4f",key=f"exact_upper_{data['identity']}_{index}")
            if lower >= upper:
                st.error("Lower fit bound must be below the upper bound. Enter an increasing interval around one reflection.")
    line_plot(x,{"Supplied intensity":y},data["name"])
    example_window = teaching and np.allclose((lower, upper), nominal_window, atol=1e-8, rtol=0)
    same_window = peak and peak[3] == index and np.allclose((lower, upper), peak[:2], atol=1e-8, rtol=0)
    isolation_key = f"isolated_{data['identity']}_{index}"
    if teaching:
        # A changed teaching window needs its own confirmation. Uploaded patterns
        # use an explicit, stable checkbox while entering and confirming bounds.
        isolation_key += f"_{lower:.10g}_{upper:.10g}"
    isolated = st.checkbox("Selected window contains one isolated reflection and no unresolved spectral doublet",
                           value=bool(peak[2] if same_window else example_window),
                           key=isolation_key)
    return lower, upper, isolated, index


def _analysis_step(data, line_plot):
    conditions = st.session_state["workflow_conditions"]
    saved = st.session_state.get("workflow_result")
    peak = st.session_state.get("workflow_peak")
    if saved and saved["identity"]==analysis_identity(data,conditions,peak):
        return _result(data,conditions,saved,line_plot)
    if saved:
        clear_result()
        st.info("Previous results cleared because data or analysis conditions changed.")
    teaching = data["teaching"]
    st.subheader("3 · Analyze the synthetic example" if teaching else "3 · Select a reflection and analyze")
    st.button("Back to conditions", on_click=back_to_conditions)
    if teaching:
        st.caption("The known reflection and its fit window are preselected. The size is calculated from the fitted pattern.")
    else:
        st.write("Choose one isolated peak. Inspect the profile before confirming the assumptions.")
    if not len(data["candidates"]):
        st.warning("No candidate reflection detected. Use Automatic multi-peak mode to adjust detection settings, or load a pattern with resolved peaks.")
        st.button("Change data",on_click=change_data)
        return None
    with st.expander("Optional · review the selected reflection and fit window") if teaching else nullcontext():
        peak = _reflection_controls(data, peak, line_plot)
    lower, upper, isolated, _ = peak
    st.caption(f"{conditions['profile']} profile · K={conditions['k']:g} · size dominance {'assumed' if conditions['size_dominated'] else 'not established'}")
    st.session_state["workflow_peak"] = peak
    if st.button("Analyze",type="primary",disabled=lower>=upper):
        clear_result()
        started = perf_counter()
        with st.status("Analyzing peak — fitting and checking physical assumptions…",expanded=True) as status:
            feedback_ms = (perf_counter()-started)*1000
            try:
                result = execute_analysis(data,conditions,lower,upper,isolated)
                elapsed = perf_counter()-started
                st.session_state["workflow_result"] = {"identity":analysis_identity(data,conditions,peak),
                    "result":result,"elapsed_s":elapsed,"status_requested_ms":feedback_ms,
                    "render_token":int(perf_counter()*1_000_000_000)}
                st.session_state["workflow_scroll_result"] = True
                status.update(label="Analysis complete",state="complete",expanded=False)
                st.rerun()
            except (ValueError, TypeError, RuntimeError, FloatingPointError) as exc:
                status.update(label="Analysis needs your attention",state="error",expanded=False)
                show_problem(error_kind(exc),str(exc))
            except Exception:
                LOG.exception("Peak analysis stopped unexpectedly")
                status.update(label="Analysis needs your attention",state="error",expanded=False)
                show_problem("fit", "The numerical solver stopped before a defensible result was available.")
    st.button("Change data",on_click=change_data)
    pattern = data["pattern"]
    x, y = pattern.two_theta_deg, pattern.intensity
    with st.expander("Input details and detected baseline"):
        st.write(f"{len(x):,} points; {x[0]:.4f}–{x[-1]:.4f}°; median step {np.median(np.diff(x)):.5f}°; no interpolation")
        for warning in pattern.warnings: st.warning(warning)
        line_plot(x,{"Supplied intensity":y,"Detection baseline":data["baseline"]},data["name"])
    _ml_details(pattern,workflow_metadata(conditions))
    return None


def render_analysis(root, line_plot):
    stage = st.session_state.setdefault("workflow_stage",1)
    st.caption("1 Load data  →  2 Confirm conditions  →  3 Analyze")
    with st.expander("What does the size mean?"):
        st.write("Peak broadening can estimate coherent-domain size under explicit assumptions. Scherrer does not automatically separate strain and instrument effects, and coherent domains are not necessarily physical grains.")
    if stage == 1 or "workflow_data" not in st.session_state:
        ceria_ui.render_reference_download()
        _load_step(root)
        return None
    data = st.session_state["workflow_data"]
    if "ceria_content" in data:
        if stage == 2:
            ceria_ui.render_conditions(data, back_to_data=lambda: _stage(1))
            if st.button("Use standard Physics workflow"):
                data.pop("ceria_content")
                st.rerun()
            return None
        return ceria_ui.render_analysis(data, change_data=change_data, back_to_conditions=back_to_conditions)
    if stage == 2:
        _conditions_step(data)
        return None
    if data["teaching"]:
        return _analysis_step(data, line_plot)
    modes = ["Automatic multi-peak", "Single reflection (detailed)"]
    mode = st.radio("Analysis mode", modes, index=modes.index(data.get("analysis_mode", modes[0])),
                    horizontal=True, key="analysis_mode")
    data["analysis_mode"] = mode
    if mode == "Automatic multi-peak":
        from .multi_peak import render_multi_peak
        return render_multi_peak(data,st.session_state["workflow_conditions"],line_plot)
    return _analysis_step(data,line_plot)
