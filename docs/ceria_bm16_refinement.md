# Measured ceria · ESRF BM16

The IUCr Commission on Powder Diffraction round robin includes a broadened CeO₂ specimen and an annealed CeO₂ reference measured at the same ESRF BM16 analyzer beamline in capillary geometry. The monochromatic wavelength is 0.39982 Å. The measured files contain 2θ in degrees, intensity and a counting standard deviation. The broadened pattern has 6,551 points from 3.0 to 29.2° at 0.004° nominal spacing. The reference has 65,501 points over the same range at 0.0004° spacing. [IUCr dataset](https://mysite.du.edu/~balzar/s-s_rr.htm) · [Balzar et al. (2004)](https://doi.org/10.1107/S0021889804022551)

| Input | File | Provenance |
| --- | --- | --- |
| Broadened sample | [massonbr.xy](https://mysite.du.edu/~balzar/massonbr.xy) | Original Masson/Fitch measured pattern |
| Annealed reference | [massonsh.xy](https://mysite.du.edu/~balzar/massonsh.xy) | Original matched measurement used as the instrument baseline |
| Phase | [9009008.cif](../examples/ceria_bm16/9009008.cif) | Cubic fluorite CeO₂, Wyckoff, *Crystal Structures*, 2nd ed., vol. 1, pp. 239–444 (1963), via [COD 9009008](https://www.crystallography.net/cod/9009008.html), revision 291735 |
| Instrument profile | [bm16.instprm](../examples/ceria_bm16/bm16.instprm) | Produced here by refinement of the measured annealed reference. The archive supplies the reference measurement. |

## From upload to comparison

1. In **Analyze My XRD**, expand **Try a measured pattern with both methods** for the download link. If the browser opens the file as text, download the original `.xy` without formatting changes. The app matches a SHA-256 checksum of the original bytes.
2. Drop `massonbr.xy` into the upload surface, keep degrees selected and select **Load uploaded data**. The recognized file uses columns 0 and 1 for the angles and intensities, with column 2 retained for the fit weights.
3. Verify **0.39982 angstrom**, choose **Powder capillary scan (analyzer detector)**, and confirm the phase, reference and zero-strain modeling assumption. Select **Continue to analysis** and then **Compare Physics and GSAS-II**.

**Research tools** exposes the CIF and instrument profile downloads, the method diagnostics and the existing research tabs. Other scans use the standard Physics workflow. GSAS-II comparison is available for the exact measured pattern and confirmed conditions described here. The application does not apply this beamline calibration to arbitrary Cu-source scans.

## Methods and results

**Physics** fits intensity-weighted-by-counting-variance local Voigt profiles with linear backgrounds at the first four isolated fluorite reflections (111, 200, 220, 311). Windows are ±0.27° for the broadened scan and ±0.035° for the reference. Independent Gaussian variances subtract in quadrature, while Lorentzian half widths subtract linearly. The corrected Voigt FWHM goes into Scherrer with a declared factor of 1, and the result is the median of the four estimates. This route uses the measured reference's local fitted widths directly.

**GSAS-II** refines the annealed reference over 6.7–28.0° 2θ using its measured counting uncertainties, the cubic CIF and a constant-wavelength Debye–Scherrer instrument model. The reference sample size is fixed at 10 µm to approximate negligible reference broadening. Zero, wavelength, axial asymmetry, U, V and X are fixed, with W, Y, Z refined after scale, background and cell. The polarization correction is fixed to unity. The fitted Lorentzian width stays positive throughout the fitted angle range. The resulting instrument coefficients are fixed for the broadened specimen. After scale, an eight-term Chebyshev background and cubic cell are refined, the solver fits isotropic size and size profile mixing. Microstrain is fixed to zero for both fits. The phase occupancies and coordinates are fixed. GSAS-II size is converted from µm to nm. [GSAS-II phase parameters](https://advancedphotonsource.github.io/GSAS-II-tutorials/help/phasedata.html)

| Measured-sample comparison | Result |
| --- | ---: |
| Physics · Scherrer median | 30.25 nm |
| GSAS-II · Rietveld isotropic size | 24.63 nm |
| Absolute difference | 5.62 nm |
| Symmetric difference | 20.48% |
| GSAS-II sample Rwp (5,326 points) | 8.91% |
| GSAS-II annealed reference Rwp (53,252 points) | 12.75% |

Percent difference is 100 times the absolute difference divided by the arithmetic mean of the two estimates. The per-reflection Scherrer sizes span 29.96–30.51 nm. The GSAS-II size uses a mixed pseudo-Voigt profile parameter, with a Lorentzian mixing coefficient of 0.498, so it is defined differently from a deconvolved Voigt FWHM. A size-only model can absorb contributions from strain and anisotropy. The annealed reference may retain finite broadening. The reference profile has correlated angular-width coefficients, and sample goodness of fit is 4.44, indicating residual disagreement beyond the assigned counting errors. The paper discusses small strain and method-dependent size definitions. This comparison measures agreement between model-dependent coherent-domain lengths. It does not establish independent accuracy, a confidence interval or grain size. [Balzar et al. (2004)](https://doi.org/10.1107/S0021889804022551)

## Reproduce and run live

The recorded calibration and fit used GSAS-II source revision
`46b3d7b541d344f9506848c37b35737b1a87ddac` with the Apple Silicon
`mac_arm_p3.12_n2.2` extension package, Python 3.12 and PyCifRW 5.0.1.
Use a working GSAS-II environment with NumPy, SciPy and PyCifRW whose Python
can import `GSASII.GSASIIscriptable`. [Official GSAS-II installation](https://advancedphotonsource.github.io/GSAS-II-tutorials/install.html)

From the repository root, download both original input files and run the following with the GSAS-II-enabled Python.

```sh
mkdir -p tmp/ceria_sources
curl -fL https://mysite.du.edu/~balzar/massonbr.xy -o tmp/ceria_sources/massonbr.xy
curl -fL https://mysite.du.edu/~balzar/massonsh.xy -o tmp/ceria_sources/massonsh.xy
python -m validation.refine_ceria_bm16 \
  --sample tmp/ceria_sources/massonbr.xy --reference tmp/ceria_sources/massonsh.xy \
  --out tmp/ceria_sources/refinement.json --work-dir tmp/ceria_sources/gsas_projects
```

The command writes the instrument profile beside the JSON. The optional work directory retains GSAS-II projects and refinement listings. The app ships [the recorded results](../examples/ceria_bm16/refinement.json), including input digests, residual metrics and fitted widths. Keep the raw downloaded files for an independent rerun.

To run the solver when a visitor clicks **Compare Physics and GSAS-II**, specify the full path to the GSAS-II-enabled Python when starting Streamlit.

```sh
XRD_GSAS_PYTHON=/absolute/path/to/gsas-enabled/python python -m streamlit run app.py
```

If the GSAS-II source is outside that environment, set its source root in `PYTHONPATH`. With only the standard app dependencies, Streamlit computes the Physics estimate from the upload and labels the GSAS-II result as a recorded refinement for that exact file. A configured live failure leaves Physics available and withholds both the GSAS-II size and the difference.

## Browser verification

The public .xy was selected with the macOS file picker in Chrome. After confirming the published conditions, the live backend displayed both sizes, the difference and GSAS-II Rwp. The 1280×800 and 390×844 Chrome viewports showed no document-level horizontal overflow. The phone view scrolls to the comparison and keeps the limitations below the four measurements.

Five warm runs on the same Apple Silicon Mac (macOS 27.0.1, Python 3.12.14) took 3.029, 2.560, 6.268, 2.458 and 2.344 s from button interaction to the visible completion status. Median time was 2.560 s. Chrome automation observed the processing message in 0.406, 0.279, 4.100, 0.281 and 0.285 s. Four of five observations met the one-second target. These timings include automation and browser interaction overhead. The five-person usability target remains unverified.
