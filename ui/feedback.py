"""Plain-language problems and next actions, without exposing tracebacks."""
import streamlit as st


def show_problem(kind, detail=""):
    messages = {
        "missing_file": ("No file selected", "There is no measurement to load.",
                         "Choose a CSV, TXT, XY or DAT file, then select Load uploaded data. Or select Load synthetic example."),
        "file": ("This file could not be read", "The reader needs two numeric columns: 2theta and intensity.",
                 "Export a plain-text pattern with increasing angles and no missing values. For extra columns, open Column selection and choose the angle and intensity columns."),
        "wavelength": ("Confirm the wavelength", "A source name or approximate value alone does not establish the wavelength used for size analysis.",
                       "Enter the measurement wavelength, select its unit, verify the angular unit, and tick the wavelength confirmation. Check unresolved spectral doublets before fitting."),
        "geometry": ("Declare the measurement geometry", "The geometry is unknown or outside this powder-analysis workflow. Cu radiation does not identify geometry.",
                     "Check the instrument record and select the matching powder geometry. If it is another geometry, request a method review before estimating size."),
        "fit": ("This peak does not support a size estimate", "A reliable isolated-peak width could not be established.",
                "Select a different isolated reflection or widen the fit window. Inspect the residuals, sampling and chosen profile. Do not fit overlapping peaks or an unresolved spectral doublet as one peak."),
        "instrument": ("Instrument correction is not valid", "The observed peak must be wider than the matching instrumental reference; otherwise sample broadening is unresolved.",
                       "Check the reference width, degree units and profile type. Use a broader resolved reflection or report that size is beyond instrument resolution. Do not force a corrected width to zero."),
        "assumptions": ("Confirm the reflection assumptions", "The selected window has not been confirmed as an isolated reflection.",
                        "Inspect the reflection and tick the isolation/no-doublet confirmation only if justified. Size output also needs the stated size-dominance assumption."),
        "size_evidence": ("Peak width is available; size is not established", "The contributions from finite size, strain/defects and the instrument have not been separated.",
                          "Review calibration and strain/defect evidence. Declare size dominance only when physically justified; otherwise retain the peak width and report size as undetermined."),
        "calibration": ("Complete the calibration record", "A calibrated correction needs a reference and matching measurement conditions.",
                        "Enter the reference identifier and confirm matching wavelength, geometry and profile. Otherwise select Declared simplified constant width for a preliminary estimate."),
    }
    title, reason, next_step = messages.get(kind, messages["fit"])
    st.error(f"**{title}.** {reason}\n\n**Next:** {next_step}")
    if detail:
        with st.expander("Technical reason"):
            # Scientific exceptions are text, not executable/rendered traceback.
            st.text(str(detail)[:1200])


def error_kind(exc):
    text = str(exc).lower()
    if "wavelength" in text or "angular units" in text:
        return "wavelength"
    if "geometry" in text:
        return "geometry"
    if "instrument" in text or "observed fwhm" in text:
        return "instrument"
    if "isolated reflection" in text:
        return "assumptions"
    return "fit"
