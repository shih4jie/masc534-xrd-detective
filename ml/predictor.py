"""Public upload validation and explicit original-data reproduction.

Reproduction results are never evidence that a new uploaded pattern is supported.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from .inference import predict_crystallite_size
from .compatibility import MODEL_ID, notebook_grid
from .lpa_nn_adapter import (verify_spinel_assets, load_original_spinel,
                            load_spinel_normalization, predict_original_spinel)

ROOT = Path(__file__).resolve().parents[1]
SPINEL = ROOT / 'third_party/LPA-NN/Spinel'


def predict_lpa_nn(pattern, metadata):
    """Validate experimental uploads; fail closed until accuracy/domain certified."""
    result = predict_crystallite_size(pattern, metadata).to_dict()
    return {**result, 'predicted_size_nm':result['predicted_crystallite_size'],
            'predicted_microstrain_percent':None, 'accuracy_validated':False}


@lru_cache(maxsize=1)
def _original_resources():
    return load_original_spinel(SPINEL), load_spinel_normalization(SPINEL)


def reproduce_original_measurement(index):
    """Real trained inference ONLY on a hash-verified original archive row.

    Does not accept arbitrary uploads, paths, resampled/simulated inputs or flags.
    Each invocation verifies all assets, including when the model is cached.
    """
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < 126:
        raise ValueError('Select an original measurement index from 0 to 125.')
    verify_spinel_assets(SPINEL)
    with np.load(SPINEL/'data.npz', allow_pickle=False) as archive:
        row = archive['exp_data'][int(index)]
    model, (lo, hi) = _original_resources()
    prediction = predict_original_spinel(model, row, lo, hi)
    size = float(prediction['size_nm'][0]); strain = float(prediction['microstrain_percent'][0])
    if not np.isfinite(size) or not np.isfinite(strain) or size <= 0:
        raise ValueError('Checkpoint produced a nonphysical or nonfinite output; reproduction withheld.')
    return {'predicted_size_nm':size, 'predicted_microstrain_percent':strain,
            'model_identifier':MODEL_ID, 'measurement_index':int(index),
            'input_compatibility_status':'Original-data reproduction only',
            'accuracy_validated':False, 'independent_ground_truth':False,
            'diagnostic_warnings':['Unvalidated research output; not an experimentally established size.',
                'No matched labeled held-out set; MAE/RMSE unavailable.',
                'Notebook angular coordinates are for plotting only; do not reuse them as certified upload metadata.'],
            'plotting_axis_deg':notebook_grid().tolist()}
