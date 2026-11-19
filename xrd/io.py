from dataclasses import dataclass
import csv
import re
import numpy as np


@dataclass
class Pattern:
    two_theta_deg: np.ndarray
    intensity: np.ndarray
    warnings: list[str]
    source: str = "User upload"

    @property
    def sampling_interval_deg(self):
        """Median supplied 2theta step, not a resampled-grid specification."""
        return float(np.median(np.diff(self.two_theta_deg)))

    @property
    def sampling_uniform(self):
        steps = np.diff(self.two_theta_deg)
        return bool(np.allclose(steps, np.median(steps), rtol=.02, atol=1e-8))


def validate_pattern(x, y, source="User upload"):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 12:
        raise ValueError("Expected at least 12 rows of two numerical columns.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Missing or nonfinite angles/intensities are not supported.")
    if np.any(x <= 0) or np.any(x >= 180):
        raise ValueError("2theta must lie strictly between 0 and 180 degrees.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Angles must strictly increase; duplicate/reversed values are rejected.")
    if np.ptp(y) <= 0 or np.max(y) <= 0:
        raise ValueError("Intensity must contain a positive, nonconstant signal.")
    warnings = []
    if np.any(y < 0):
        warnings.append("Negative intensities: possible background subtraction; ML log input is invalid.")
    dx = np.diff(x)
    if not np.allclose(dx, np.median(dx), rtol=0.02, atol=1e-8):
        warnings.append("Uneven sampling: local fitting uses actual coordinates; no resampling applied.")
    return Pattern(x, y, warnings, source)


def _fields(line):
    """Recognize common text delimiters; quoted delimited headers stay intact."""
    for delimiter in (",", ";", "\t"):
        if delimiter in line:
            return [field.strip().strip('"') for field in next(csv.reader([line], delimiter=delimiter))]
    return re.split(r"\s+", line.strip())


def _column_role(field):
    label = field.casefold().replace("θ", "theta")
    label = re.sub(r"\([^)]*\)|\[[^]]*\]", "", label)
    label = re.sub(r"[^a-z0-9]", "", label)
    angle_names = {"2theta", "twotheta", "twothetadeg", "twothetadegrees", "2thetadeg",
                   "2thetadegrees", "2thetarad", "2thetaradians", "angle", "angledeg", "angledegrees"}
    intensity_names = {"intensity", "i", "counts", "count", "cps", "intensityau",
                       "intensitycounts", "intensitycps", "countspersecond", "yobs", "iobs"}
    if label in angle_names:
        return "angle"
    if label in intensity_names:
        return "intensity"
    return None


def _header_columns(line):
    # Unit text in a whitespace-separated header does not represent extra data
    # columns. Delimited labels may contain spaces and are handled by csv.reader.
    candidate = line
    if not any(delimiter in line for delimiter in (",", ";", "\t")):
        candidate = re.sub(r"\([^)]*\)|\[[^]]*\]", "", candidate)
        candidate = re.sub(r"\b(?:2|two)\s+(?:theta|θ)\b", "2theta", candidate, flags=re.I)
    fields = _fields(candidate)
    roles = [_column_role(field) for field in fields]
    angles = [i for i, role in enumerate(roles) if role == "angle"]
    intensities = [i for i, role in enumerate(roles) if role == "intensity"]
    if len(angles) == 1 and len(intensities) == 1:
        return (angles[0], intensities[0]), len(fields)
    if angles and intensities:
        raise ValueError("Ambiguous angle/intensity headers; explicitly select the required columns.")
    return None, None


def read_pattern(content, angle_unit="degrees", columns=None, source="User upload"):
    """Read measured CSV/TXT/XY/DAT without dropping malformed measurement rows.

    Clear header names identify angle/intensity columns, including reversed and
    extra-column exports. Unlabeled two-column data use the documented 2theta,
    intensity order. Ambiguous extra columns require explicit zero-based indices.
    Header labels do not establish units, geometry, wavelength or crystalline
    phase; angle_unit remains an explicit caller declaration. Decimal commas are
    not inferred.
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    if len(content) > 15_000_000:
        raise ValueError("Use a text file smaller than 15 MB.")
    if columns is not None and (len(columns) != 2 or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in columns)
                                or min(columns) < 0 or columns[0] == columns[1]):
        raise ValueError("Choose two distinct nonnegative column indices.")
    rows, started, headers = [], False, 0
    inferred, header_width, measurement_width = None, None, None
    for line_no, line in enumerate(content.splitlines(), 1):
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        if line.startswith(("#", "!", ";")):
            if not started and columns is None:
                selected_header, width = _header_columns(line.lstrip("#!;").strip())
                if selected_header is not None:
                    inferred, header_width = selected_header, width
            continue
        line = line.split("#", 1)[0].strip()
        if not started and columns is None:
            selected_header, width = _header_columns(line)
            if selected_header is not None:
                inferred, header_width = selected_header, width
                headers += 1
                if headers > 100:
                    raise ValueError("Too many header lines; use a two-column text export.")
                continue
        fields = _fields(line)
        selected = columns if columns is not None else inferred or (0, 1)
        try:
            pair = [float(fields[i]) for i in selected]
        except (ValueError, IndexError) as exc:
            # A recognized column header starts the measurement section. Later
            # malformed rows must never be reinterpreted as extra metadata.
            numeric_start = inferred is not None
            for index in selected:
                try:
                    float(fields[index])
                    if index == selected[0] or selected[0] >= len(fields) or not fields[selected[0]].strip():
                        numeric_start = True
                except (ValueError, IndexError):
                    pass
            if started or numeric_start:
                raise ValueError(f"Malformed or missing numeric data at line {line_no}.") from exc
            headers += 1
            if headers > 100:
                raise ValueError("Too many header lines; use a two-column text export.")
            continue
        if columns is None and inferred is None and len(fields) != 2:
            raise ValueError("Extra numerical columns found; explicitly select angle/intensity columns.")
        if columns is None and header_width is not None and len(fields) != header_width:
            raise ValueError(f"Measurement columns do not match the detected header at line {line_no}.")
        if measurement_width is not None and len(fields) != measurement_width:
            raise ValueError(f"Inconsistent measurement column count at line {line_no}.")
        measurement_width = len(fields)
        started = True
        rows.append(pair)
    if not rows:
        raise ValueError("No numerical data found.")
    data = np.asarray(rows)
    if angle_unit == "radians":
        data[:, 0] = np.rad2deg(data[:, 0])
    elif angle_unit != "degrees":
        raise ValueError("Declare 2theta units as degrees or radians.")
    return validate_pattern(data[:, 0], data[:, 1], source)
