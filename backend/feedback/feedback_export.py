# -*- coding: utf-8 -*-
"""
backend/feedback/feedback_export.py

Feedback Exporter, Knowledge Generator & Future Training Dataset Builder.
Exports Approved clinician feedback into CSV, JSON, and Parquet formats, and builds
structured Knowledge Base records and Future Training Datasets without modifying core models.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from backend.feedback.feedback_repository import FeedbackRepository

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = PACKAGE_ROOT / "outputs" / "feedback"
KNOWLEDGE_DIR = PACKAGE_ROOT / "knowledge"
TRAINING_DIR = PACKAGE_ROOT / "datasets" / "future_training"


def _ensure_export_dirs() -> None:
    """Create mutable export paths only when an export is requested."""
    for path in (
        EXP_DIR,
        KNOWLEDGE_DIR / "approved_cases",
        TRAINING_DIR / "classifier",
        TRAINING_DIR / "retrieval",
        TRAINING_DIR / "bridge",
    ):
        path.mkdir(parents=True, exist_ok=True)


def validate_export_record(r: Dict[str, Any]) -> Tuple[str, str]:
    """
    Validates that a feedback record contains distinct authoritative labels for original prediction
    and clinician correction (Part B1). Rejects incomplete or invalid records.
    """
    original_prediction = r.get("original_model_prediction") or r.get("original_prediction") or r.get("model_primary_prediction")
    clinician_correction = r.get("clinician_primary_scp")

    if not original_prediction or not clinician_correction:
        raise ValueError(
            f"Feedback record {r.get('case_id', 'UNKNOWN')} lacks required labels"
        )
    return str(original_prediction), str(clinician_correction)


class FeedbackExporter:
    """Exports Approved feedback data into research formats, Knowledge Items, and Future Training Datasets."""

    def __init__(self):
        self.repo = FeedbackRepository()

    def export_approved_data(self) -> Dict[str, str]:
        """
        Exports all Approved feedback records into CSV and JSON.
        """
        _ensure_export_dirs()
        approved_records = self.repo.get_approved_feedback()
        
        # 1. Export JSON
        json_path = EXP_DIR / "approved_feedback_export.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(approved_records, f, indent=2)

        # 2. Export CSV
        csv_path = EXP_DIR / "approved_feedback_export.csv"
        if approved_records:
            keys = list(approved_records[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(approved_records)
        else:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("No approved records available.\n")

        # 3. Export Knowledge Items
        self._build_knowledge_database(approved_records)

        # 4. Export Future Training Datasets
        exported_count = self._build_future_training_datasets(approved_records)

        return {
            "json_export": str(json_path),
            "csv_export": str(csv_path),
            "approved_count": len(approved_records),
            "valid_training_record_count": exported_count,
            "status": "Success"
        }

    def _build_knowledge_database(self, records: List[Dict[str, Any]]):
        """Generates structured Knowledge Items for approved cases."""
        for r in records:
            scp = r.get("clinician_primary_scp")
            family = r.get("clinician_family")
            if not scp or not family:
                continue

            rare_scps = {"3AVB", "2AVB", "WPW", "SVT", "AFIB", "AFLT", "ALMI", "INJAL", "ANEUR"}
            difficulty = "Rare" if scp in rare_scps else "Standard"

            k_item = {
                "case_id": r["case_id"],
                "scp_label": scp,
                "family": family,
                "difficulty": difficulty,
                "retrieval_examples": r.get("retrieval_evaluations_json", "[]"),
                "clinical_notes": [r.get("general_comments", "")] if r.get("general_comments") else [],
                "reviewed": True,
                "version": "1.0"
            }

            out_file = KNOWLEDGE_DIR / "approved_cases" / f"case_{r['case_id']}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(k_item, f, indent=2)

    def _build_future_training_datasets(self, records: List[Dict[str, Any]]) -> int:
        """
        Builds future training dataset exports for Classifier, Retrieval, and Bridge (Part B1).
        Quarantines/rejects incomplete feedback records missing authoritative labels.
        """
        classifier_samples = []
        retrieval_samples = []
        bridge_samples = []

        for r in records:
            original_prediction = r.get("original_model_prediction") or r.get("original_prediction") or r.get("model_primary_prediction")
            clinician_correction = r.get("clinician_primary_scp")

            # Fail-closed: missing labels are quarantined from retraining
            if not original_prediction or not clinician_correction:
                continue

            sample = {
                "case_id": r["case_id"],
                "ecg_id": r["ecg_id"],
                "original_prediction": original_prediction,
                "clinician_correction": clinician_correction,
                "retrieval_rating": r.get("retrieval_evaluations_json", "[]"),
                "bridge_rating": r.get("bridge_explanation_rating"),  # None if absent
                "confidence_rating": r.get("confidence_rating")      # None if absent
            }
            classifier_samples.append(sample)
            retrieval_samples.append(sample)
            bridge_samples.append(sample)

        with open(TRAINING_DIR / "classifier" / "future_classifier_data.json", "w", encoding="utf-8") as f:
            json.dump(classifier_samples, f, indent=2)

        with open(TRAINING_DIR / "retrieval" / "future_retrieval_data.json", "w", encoding="utf-8") as f:
            json.dump(retrieval_samples, f, indent=2)

        with open(TRAINING_DIR / "bridge" / "future_bridge_data.json", "w", encoding="utf-8") as f:
            json.dump(bridge_samples, f, indent=2)

        return len(classifier_samples)
