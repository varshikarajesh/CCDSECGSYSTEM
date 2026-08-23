"""Authoritative public entry point for Evidence Bridge V4."""
from __future__ import annotations

from typing import Any, Dict, Optional

from backend.bridge.evidence_bridge_v4 import EvidenceBridgeV4


class EvidenceBridge(EvidenceBridgeV4):
    """Public V4 bridge with a narrow adapter for older ``combine`` callers."""

    VERSION = EvidenceBridgeV4.VERSION

    def combine(
        self,
        classifier_output: Any = None,
        retrieval_output: Optional[Dict[str, Any]] = None,
        holter_output: Optional[Dict[str, Any]] = None,
        statistics_output: Optional[Dict[str, Any]] = None,
        selected_windows: Any = None,
        signal_quality: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **legacy: Any,
    ) -> Dict[str, Any]:
        if classifier_output is None:
            classifier_output = legacy.pop("scp_predictions", None)
        if retrieval_output is None:
            retrieval_output = legacy.pop("retrieval_results", None)
        # V4 does not let the former single-family head override multi-label findings.
        legacy.pop("family_predictions", None)
        return super().combine(
            classifier_output=classifier_output,
            retrieval_output=retrieval_output,
            holter_output=holter_output,
            statistics_output=statistics_output,
            selected_windows=selected_windows,
            signal_quality=signal_quality,
            metadata=metadata,
        )


__all__ = ["EvidenceBridge", "EvidenceBridgeV4"]
