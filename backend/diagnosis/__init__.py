"""Diagnosis package exports without circular model initialization."""
from backend.diagnosis.external_dataset_adapter import ExternalDatasetAdapter

__all__ = ["ExternalDatasetAdapter", "DiagnosisModel"]


def __getattr__(name):
    if name == "DiagnosisModel":
        from backend.diagnosis_model import DiagnosisModel
        return DiagnosisModel
    raise AttributeError(name)
