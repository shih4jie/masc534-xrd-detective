"""Explicit instrumental-width calibration, distinct from a standard pattern."""
import csv
import re
import numpy as np


def read_width_calibration(content):
    """Read 2theta degrees and intrinsic-width-corrected instrumental FWHM degrees."""
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    if len(content)>1_000_000:
        raise ValueError("Use an instrument-width table smaller than 1 MB.")
    rows, started = [], False
    for index, line in enumerate(content.splitlines(),1):
        line=line.strip()
        if not line or line.startswith(("#","!",";")):
            continue
        delimiter=next((d for d in (",",";","\t") if d in line),None)
        fields=next(csv.reader([line],delimiter=delimiter)) if delimiter else re.split(r"\s+",line)
        if len(fields)!=2:
            raise ValueError("Instrument calibration needs exactly two columns: 2theta degrees and instrumental FWHM degrees.")
        try:
            values=[float(v) for v in fields]
        except ValueError as exc:
            if started or index>30 or any(re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",v.strip()) for v in fields):
                raise ValueError(f"Invalid calibration row {index}; no rows were silently dropped.") from exc
            continue
        rows.append(values);started=True
    values=np.asarray(rows,dtype=float)
    if (values.ndim!=2 or values.shape[1]!=2 or len(values)<2 or not np.isfinite(values).all()
            or np.any(np.diff(values[:,0])<=0) or np.any(values[:,0]<=0)
            or np.any(values[:,0]>=180) or np.any(values[:,1]<0)):
        raise ValueError("Calibration needs at least two increasing angles between 0 and 180 degrees and finite nonnegative FWHM values.")
    return {"angles_deg":values[:,0].tolist(),"widths_deg":values[:,1].tolist()}
