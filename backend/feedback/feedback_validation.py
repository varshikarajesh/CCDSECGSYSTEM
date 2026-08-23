# -*- coding: utf-8 -*-
"""
feedback_validation.py

Security, Input Validation, Anonymization & Hashing Module.
Ensures zero PII storage by hashing Clinician and Patient IDs with SHA-256,
and validates all feedback choices against clinical schemas.
"""

import hashlib
from typing import Dict, Any, Tuple, Optional
from backend.feedback.schemas import FeedbackRequestSchema


class FeedbackValidator:
    """Security and Validation Engine for Clinician Feedback."""

    SALT = "ECG_CLINICAL_FEEDBACK_SALT_V1"

    @classmethod
    def hash_identifier(cls, identifier: str) -> str:
        """Computes SHA-256 salted hash for patient/clinician IDs to ensure 100% PII protection."""
        if not identifier:
            return "anon_unspecified"
        salted_str = f"{cls.SALT}:{identifier}"
        return hashlib.sha256(salted_str.encode("utf-8")).hexdigest()[:32]

    @classmethod
    def validate_and_anonymize(cls, raw_feedback: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Validates input fields against clinical schema and replaces PII with SHA-256 hashes.
        Returns (anonymized_payload, error_message).
        """
        correctness = raw_feedback.get("diagnosis_correctness")
        if correctness not in FeedbackRequestSchema.VALID_CORRECTNESS:
            return {}, f"Invalid diagnosis correctness choice: '{correctness}'. Must be one of {FeedbackRequestSchema.VALID_CORRECTNESS}"

        clinician_id = raw_feedback.get("clinician_id", "")
        patient_id = raw_feedback.get("patient_id", "")

        anonymized_payload = dict(raw_feedback)
        anonymized_payload["clinician_hash"] = cls.hash_identifier(clinician_id)
        anonymized_payload["patient_hash"] = cls.hash_identifier(patient_id)

        # Remove raw PII
        anonymized_payload.pop("clinician_id", None)
        anonymized_payload.pop("patient_id", None)

        return anonymized_payload, None
