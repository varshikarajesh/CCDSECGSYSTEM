"""Authoritative Bridge V4 deployment interface."""

from backend.bridge.evidence_bridge import EvidenceBridge
from backend.bridge.evidence_bridge_v4 import EvidenceBridgeV4, validate_bridge_result

__all__ = ["EvidenceBridge", "EvidenceBridgeV4", "validate_bridge_result"]
