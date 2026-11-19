import numpy as np


def error_metrics(truth, predicted):
    a,b = np.asarray(truth,dtype=float),np.asarray(predicted,dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a)==0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Metrics require matching, nonempty finite vectors in the same units.")
    residual = b-a
    return {"sample_count":len(a),"mae":float(np.mean(np.abs(residual))),
            "rmse":float(np.sqrt(np.mean(residual**2)))}
