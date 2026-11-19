"""Compact result preview of the supplied scan and accepted size reflections."""
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import streamlit as st


def pattern_figure(pattern, centers, labels=()):
    figure = Figure(figsize=(5.7, 1.68), layout="constrained", facecolor="white")
    axes = figure.subplots()
    axes.plot(pattern.two_theta_deg, pattern.intensity, color="#204771", lw=1)
    axes.margins(x=0)
    axes.set_xlabel("2θ (degrees)", fontsize=11, labelpad=2)
    axes.set_ylabel("Intensity", fontsize=11, labelpad=3)
    axes.xaxis.set_major_locator(MaxNLocator(nbins=6))
    axes.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axes.tick_params(labelsize=10, length=2, pad=2)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for center in centers:
        axes.axvline(center, color="#b45309", lw=1.3, ls="--")
    for index, (center, label) in enumerate(zip(centers, labels)):
        axes.annotate(label, (center, 1), xycoords=axes.get_xaxis_transform(),
                      xytext=(0, -4 - 12 * (index % 2)), textcoords="offset points",
                      ha="center", va="top", fontsize=10, color="#9a3412",
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    return figure


def render_pattern_chart(pattern, centers=(), *, labels=()):
    if centers:
        positions = ", ".join(f"{center:.4f}°" for center in centers)
        reflections = f" Reflections {', '.join(labels)}." if labels else ""
        alt = f"Original XRD scan with Scherrer size peaks marked at {positions} 2θ.{reflections}"
        caption = "Original XRD · Orange lines mark peaks used for the Scherrer size."
    else:
        alt = "Original XRD scan. No peaks contributed to a size estimate."
        caption = "Original XRD · No peaks contributed to a size estimate."
    st.pyplot(pattern_figure(pattern, centers, labels), alt=alt, width="stretch")
    st.caption(caption)
