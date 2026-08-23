# -*- coding: utf-8 -*-
"""
utils/evidence_fields.py

Shared utility functions for resolving Evidence Bridge and classifier evidence fields
without creating circular dependencies between model and explanation modules.
"""

from typing import Dict, Any, Optional
import numpy as np


def resolve_primary_probability(
    bridge: Dict[str, Any],
    classifier: Dict[str, Any],
) -> Optional[float]:
    """
    Authoritatively resolves primary diagnostic candidate probability across
    Evidence Bridge output, classifier output, and classifier probability maps.
    Returns float probability in [0.0, 1.0] or None if unavailable.
    """
    bridge_dict = bridge if isinstance(bridge, dict) else {}
    classifier_dict = classifier if isinstance(classifier, dict) else {}

    candidates = [
        bridge_dict.get("primary_probability"),
        classifier_dict.get("primary_probability"),
    ]

    primary_label = (
        bridge_dict.get("primary_label")
        or classifier_dict.get("primary_label")
    )

    probabilities = classifier_dict.get("probabilities", {})
    if primary_label and isinstance(probabilities, dict) and primary_label in probabilities:
        candidates.append(probabilities[primary_label])

    for value in candidates:
        if value is None:
            continue
        try:
            probability = float(value)
            if 0.0 <= probability <= 1.0:
                return probability
        except (TypeError, ValueError):
            continue

    return None


def validate_classifier_evidence(classifier: Dict[str, Any]) -> None:
    """
    Validates that a classifier payload contains valid, non-fabricated,
    and internally consistent diagnostic evidence (Part B4).
    """
    if not isinstance(classifier, dict):
        raise ValueError("Classifier payload must be a dictionary")

    label = classifier.get("primary_label")
    probability = classifier.get("primary_probability")
    probabilities = classifier.get("probabilities")

    if not label or label in ("None", "NULL"):
        raise ValueError("Classifier primary label unavailable")

    if probability is None:
        raise ValueError("Classifier primary probability unavailable")

    try:
        prob_float = float(probability)
    except (TypeError, ValueError):
        raise ValueError("Classifier probability must be numeric")

    if not (0.0 <= prob_float <= 1.0):
        raise ValueError("Classifier probability outside [0,1]")

    if isinstance(probabilities, dict) and label in probabilities:
        try:
            mapped_probability = float(probabilities[label])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid probability map entry"
            ) from exc

        if not np.isfinite(mapped_probability):
            raise ValueError(
                "Probability map entry is not finite"
            )

        if not (0.0 <= mapped_probability <= 1.0):
            raise ValueError(
                "Probability map entry outside [0,1]"
            )

        if not np.isclose(
            prob_float,
            mapped_probability,
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                "Primary probability does not match probability map"
            )
