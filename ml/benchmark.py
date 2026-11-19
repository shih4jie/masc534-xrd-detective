"""Metrics for independently documented matched held-out predictions, in nm."""
import numpy as np
from .evaluation import error_metrics


def benchmark_predictions(truth_nm, predicted_nm, size_edges=(0,20,50,100,200,500,float('inf'))):
    truth=np.asarray(truth_nm,dtype=float); pred=np.asarray(predicted_nm,dtype=float)
    metrics=error_metrics(truth,pred)
    if np.any(truth<=0):
        raise ValueError('Ground-truth crystallite sizes must be positive.')
    edges=np.asarray(size_edges,dtype=float)
    if edges.ndim!=1 or len(edges)<2 or np.any(np.isnan(edges)) or np.any(np.diff(edges)<=0) or edges[0]>truth.min() or edges[-1]<=truth.max():
        raise ValueError('Size bins must strictly increase and cover every true size (upper bounds exclusive).')
    relative=np.abs(pred-truth)/truth
    groups=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(truth>=lo)&(truth<hi)
        groups.append({'lower_nm':float(lo),'upper_nm':float(hi) if np.isfinite(hi) else None,
            **(error_metrics(truth[mask],pred[mask]) if mask.any() else {'sample_count':0,'mae':None,'rmse':None}),
            'mean_absolute_relative_error':float(relative[mask].mean()) if mask.any() else None})
    return {**metrics,'units':'nm','bias_nm':float((pred-truth).mean()),
        'truth_range_nm':[float(truth.min()),float(truth.max())],'mean_absolute_relative_error':float(relative.mean()),
        'mean_absolute_relative_error_percent':float(100*relative.mean()),
        'nonpositive_prediction_count':int(np.count_nonzero(pred<=0)), 'size_bins':groups,
        'prediction_pairs':[{'truth_nm':float(a),'prediction_nm':float(b),'residual_nm':float(b-a),
                            'absolute_relative_error':float(abs(b-a)/a)} for a,b in zip(truth,pred)]}
