from dataclasses import dataclass, field, asdict
from .compatibility import assess_compatibility, model_requirements, MODEL_ID


@dataclass
class PredictionResult:
    predicted_crystallite_size: float | None = None
    units: str = "nm"
    model_identifier: str = MODEL_ID
    preprocessing_details: dict = field(default_factory=dict)
    input_compatibility_status: str = "Unsupported by current model"
    eligible_for_prediction: bool = False
    compatibility_checks: list = field(default_factory=list)
    validation_evidence: dict = field(default_factory=dict)
    diagnostic_warnings: list = field(default_factory=list)
    applicable_model_limitations: list = field(default_factory=lambda: [
        "Spinel checkpoint is specific to the original synchrotron reaction experiment.",
        "Bundled experimental examples have no independent size/strain ground truth.",
        "General Size/Size_strain checkpoints and test-set reproduction remain pending.",
    ])

    def to_dict(self):
        return asdict(self)


def predict_crystallite_size(xrd_pattern, measurement_metadata, model_config=None):
    """Conservative public API: never activate an unvalidated model by a flag.

    Current audit has no validated model for user uploads. No TensorFlow import,
    checkpoint fallback, random weights, scaling, clipping or interpolation.
    The separately reproducible Spinel research example is not an upload model.
    """
    result = PredictionResult()
    requirements = model_requirements()
    report = assess_compatibility(xrd_pattern, measurement_metadata)
    result.preprocessing_details = {**requirements["preprocessing"],
                                    "expected_input_shape": requirements["input_shape"]}
    result.input_compatibility_status = report["status"]
    result.compatibility_checks = report["checks"]
    result.validation_evidence = requirements["held_out_validation"]
    result.diagnostic_warnings = [c["detail"] for c in report["checks"] if c["status"] != "pass"]
    result.diagnostic_warnings.append("Prediction withheld. Physics analysis remains available under its own assumptions.")
    return result
