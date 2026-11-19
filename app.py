"""USC MASC 534: conservative interactive XRD line-broadening demonstrator."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from physics.simulation import simulate_peak, peak_profile, PRESETS
from physics.units import bragg_spacing_nm
from ui.analyze import render_analysis
from ui.comparison import render_comparison
from ui.validation import render_validation
from ui.ml_explorer import render_ml_explorer

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="XRD Size–Strain Detective",page_icon="🔬",layout="wide",
                   initial_sidebar_state="collapsed")
st.title("XRD Size–Strain Detective")
st.caption("USC MASC 534 · Fall 2026")
research = st.sidebar.toggle("Research tools", key="show_research_tools",
                             help="Show the simulator, ML compatibility, and validation evidence.")
if research:
    analyze, explorer, compare, validation = st.tabs(
        ["Analyze My XRD", "ML Explorer", "ML vs. Physics", "Validation"])
else:
    (analyze,) = st.tabs(["Analyze My XRD"])


def line_plot(x, series, source, xlabel="2θ (degrees)",ylabel="Intensity (a.u.)"):
    fig,ax=plt.subplots(figsize=(9,3.3))
    for name,y in series.items():ax.plot(x,y,label=name,lw=1.3)
    ax.set(xlabel=xlabel,ylabel=ylabel)
    ax.legend(loc="best");fig.tight_layout();st.pyplot(fig);plt.close(fig)
    st.caption("Data source: "+source)


def render_sample_downloads():
    with st.expander("Download sample XRD data"):
        st.caption("All files use degrees 2θ. Upload a CSV, then enter the conditions in its matching JSON.")
        for stem, label in (
            ("synthetic_40nm_gaussian", "Synthetic Gaussian single peak · 40 nm"),
            ("multi_peak_gaussian_synthetic", "Synthetic Gaussian six peaks · 80 nm"),
            ("multi_peak_lorentzian_synthetic", "Synthetic Lorentzian six peaks · 80 nm"),
            ("nist_srm1979_5866_experimental", "Measured NIST ZnO · APS 11-BM"),
        ):
            csv_path = ROOT / "examples" / f"{stem}.csv"
            metadata_path = csv_path.with_suffix(".json")
            if csv_path.is_file() and metadata_path.is_file():
                data_col, conditions_col = st.columns([3, 1])
                data_col.download_button(label, csv_path.read_bytes(), csv_path.name, "text/csv",
                                         key=f"sample_csv_{stem}")
                conditions_col.download_button("Conditions (JSON)", metadata_path.read_bytes(),
                                               metadata_path.name, "application/json", key=f"sample_json_{stem}")


with analyze:
    if st.session_state.get("workflow_stage", 1) == 1:
        render_sample_downloads()
    render_analysis(ROOT, line_plot)

if research:
    with explorer:
        render_ml_explorer(ROOT, line_plot)
        with st.expander("Physics simulator and synthetic parameter exploration",expanded=False):
            st.subheader("Explore one defined reflection")
            st.caption("Teaching model of independent broadening contributions; no structure factors or full crystal structure are implied.")
            preset=st.selectbox("Physical regime",list(PRESETS))
            d,e,b=PRESETS[preset]
            c1,c2,c3=st.columns(3)
            size=c1.slider("Coherent domain size (nm)",5.,500.,d,key="size_"+preset)
            strain=c2.slider("Microstrain (dimensionless)",0.,.01,e,step=.0001,format="%.4f",key="strain_"+preset)
            instr=c3.slider("Instrument FWHM (2theta degrees)",0.,1.,b,step=.01,key="instrument_"+preset)
            c1,c2,c3=st.columns(3)
            wl=c1.number_input("Explorer wavelength (nm)",min_value=.01,max_value=1.,value=.15406,format="%.5f")
            center=c2.slider("Reflection position (2theta degrees)",10.,140.,40.)
            noise=c3.slider("Noise sigma / amplitude",0.,.1,0.,step=.005)
            explorer_profile=st.selectbox("Convolution profile",["Gaussian","Lorentzian"])
            sim=simulate_peak(center,size,strain,instr,wl,noise,explorer_profile)
            line_plot(sim["two_theta_deg"],{"Noisy observation":sim["intensity"],"Noiseless profile":sim["clean"]},"Synthetic defined peak; seed 534")
            widths={"Size":sim["size_fwhm_deg"],"Strain":sim["strain_fwhm_deg"],"Instrument":instr,"Observed":sim["observed_fwhm_deg"]}
            st.dataframe(pd.DataFrame({"Contribution":widths.keys(),"FWHM on 2theta (degrees)":widths.values()}),hide_index=True)
            components={name:peak_profile(sim["two_theta_deg"],center,width,1,explorer_profile) for name,width in widths.items() if width>0}
            line_plot(sim["two_theta_deg"],components,"Synthetic unit-height component profiles; convolved widths, not summed intensities",ylabel="Normalized intensity")
            st.write(f"Bragg d spacing (first order): {float(bragg_spacing_nm(center,wl)):.5f} nm")
            st.latex(r"2d\sin\theta=n\lambda,\quad \beta_D=\frac{K\lambda}{D\cos\theta},\quad \beta_\epsilon=4\epsilon\tan\theta")
            st.latex(r"\text{Gaussian:}\quad \beta_{obs}^2=\beta_D^2+\beta_\epsilon^2+\beta_{inst}^2")
            st.latex(r"\text{Lorentzian:}\quad \beta_{obs}=\beta_D+\beta_\epsilon+\beta_{inst}")
            st.caption("β is FWHM on the 2θ axis in radians in the equations; plotted breadths are converted to degrees. K=0.9 is a stated teaching convention, not a universal constant. Gaussian microstrain is an effective width parameter here, not automatically an RMS strain.")
            explanations={"Size dominated":"Smaller coherent domains broaden reflections approximately as 1/(D cos θ).",
                          "Strain dominated":"The effective strain width grows as tan θ. Microstrain describes a distribution of spacings, not macroscopic residual stress.",
                          "Instrument dominated":"The observed width is dominated by the instrument. Small errors in its calibration can greatly change the corrected size."}
            st.write(explanations[preset])
            st.caption("Analytical reference: D=40 nm, λ=0.15406 nm, K=0.9, 2θ=40° gives size FWHM=0.2113533946°. See Validation for independently computed fit recovery.")

    with compare:
        render_comparison(line_plot)

    with validation:
        render_validation(ROOT, line_plot)
