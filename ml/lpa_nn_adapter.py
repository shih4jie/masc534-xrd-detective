"""Original checkpoint loader, used only in the explicit reproduction command."""
from pathlib import Path
import hashlib
import pickle
import numpy as np
from .preprocessing import denormalize_minmax, original_input_tensor

SPINEL_SHA256 = {
    "CNN.hdf5":"6efd1ed8a8e28861195c0b59de1c3986db60f808055049762682b6ddcad93edc",
    "normL.class":"2295185620e3ccbb977636af23c2fdaad8f2769f3139e7b4f8bf5c7a7bb7bb62",
    "data.npz":"80684b441b253fd80b7c115c8213e7084eb6c7575d3d41a7c32094e715d15d26",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_spinel_assets(folder, include_examples=True):
    assets = SPINEL_SHA256 if include_examples else {k:v for k,v in SPINEL_SHA256.items() if k != "data.npz"}
    for name, expected in assets.items():
        path = Path(folder)/name
        if not path.is_file():
            raise ValueError(f"Missing original asset: {path}")
        if file_sha256(path) != expected:
            raise ValueError(f"Original checkpoint/example identity mismatch: {name}")


class _NormalizationState:
    pass


class _RestrictedNormalizationUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("utilities", "NormalizeL"):
            return _NormalizationState
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "_reconstruct":
            return np._core.multiarray._reconstruct
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        raise ValueError(f"Unexpected normalization pickle global: {module}.{name}")


def load_spinel_normalization(folder):
    # Only the hash-pinned original pickle, never an uploaded pickle.
    verify_spinel_assets(folder, include_examples=False)
    with open(Path(folder)/"normL.class", "rb") as f:
        state = _RestrictedNormalizationUnpickler(f).load()
    if state.norm != "MinMax" or state.ymin.shape != (2,) or state.ymax.shape != (2,):
        raise ValueError("Unexpected original normalization scheme.")
    return np.array(state.ymin, copy=True), np.array(state.ymax, copy=True)


def load_original_spinel(folder):
    verify_spinel_assets(folder, include_examples=False)
    import tensorflow as tf
    class Log10Layer(tf.keras.layers.Layer):
        def call(self, inputs):
            return tf.math.log(inputs)/tf.math.log(tf.constant(10.0,dtype=inputs.dtype))
    # No optimizer restoration is needed for inference. Network and weights are
    # read directly from the original full-model HDF5; no substitute architecture.
    model = tf.keras.models.load_model(Path(folder)/"CNN.hdf5",
                                      custom_objects={"Log10Layer":Log10Layer},compile=False)
    if tuple(model.input_shape) != (None,10501,1,1) or tuple(model.output_shape) != (None,2):
        raise ValueError("Original checkpoint tensor shape mismatch.")
    return model


def predict_original_spinel(model, intensities, minimum, maximum, batch_size=16):
    tensor = original_input_tensor(intensities)
    raw = np.concatenate([model([tensor[i:i+batch_size]], training=False).numpy()
                          for i in range(0,len(tensor),batch_size)])
    if raw.shape != (len(tensor),2):
        raise ValueError("Unexpected inference output shape.")
    physical = denormalize_minmax(raw,minimum,maximum)
    return {"size_nm":physical[:,0]*1000,"microstrain_percent":physical[:,1]*1e-4,
            "raw_normalized":raw}
