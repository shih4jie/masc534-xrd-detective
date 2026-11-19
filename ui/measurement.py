"""Measurement declarations shared by the UI and workflow verification.

Material is a description, not phase identification. Geometry is never inferred
from material, filename or X-ray wavelength.
"""
from physics.units import wavelength_nm

UNKNOWN_GEOMETRY = "Unknown / not declared"
SUPPORTED_POWDER_GEOMETRIES = {
    "Powder θ–2θ scan",
    "Integrated powder Debye–Scherrer rings",
    "Powder capillary scan (analyzer detector)",
    "Synthetic powder-like isolated peak",
}


def measurement_metadata(*, material, wavelength_value, wavelength_unit,
                         geometry, wavelength_verified):
    value = float(wavelength_nm(wavelength_value, wavelength_unit))
    return {
        "material": material.strip() or None,
        "geometry": geometry,
        "wavelength_nm": value if wavelength_verified else None,
        "entered_wavelength": float(wavelength_value),
        "entered_wavelength_unit": wavelength_unit,
        "wavelength_verified": bool(wavelength_verified),
    }


def require_physics_metadata(metadata):
    if not metadata.get("wavelength_verified") or metadata.get("wavelength_nm") is None:
        raise ValueError("Verify the wavelength and angular units first.")
    if metadata.get("geometry") not in SUPPORTED_POWDER_GEOMETRIES:
        raise ValueError("Declare a supported powder measurement geometry before size analysis. Geometry is not inferred from the Cu source or material.")
