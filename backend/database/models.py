# -*- coding: utf-8 -*-
"""
backend/database/models.py

Data Models & Table Definitions for Clinician Feedback Platform.
Defines Append-only Feedback Records, Review Queue, and Audit Log structures.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class FeedbackRecordModel:
    """Append-only Clinician Feedback Database Entity."""
    case_id: str
    timestamp: str
    ecg_id: str
    patient_hash: str
    clinician_hash: str
    diagnosis_correctness: str
    clinician_primary_scp: Optional[str] = None
    clinician_secondary_scps: List[str] = field(default_factory=list)
    clinician_family: Optional[str] = None
    confidence_rating: Optional[str] = None
    bridge_explanation_rating: Optional[str] = None
    retrieval_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    general_comments: Optional[str] = None
    classifier_version: str = "v5.0"
    family_head_version: str = "v1.0"
    retrieval_version: str = "v1.0"
    bridge_version: str = "v3.0"
    faiss_version: str = "v1.0"
    signal_quality: float = 1.0
    confidence_score: int = 90
    deployment_version: str = "v7.0"
    review_status: str = "Pending"
    id: Optional[int] = None
    created_at: Optional[str] = None


@dataclass
class ReviewQueueModel:
    """Review Queue State Transition Entry."""
    feedback_id: int
    review_status: str  # 'Pending', 'Approved', 'Rejected', 'Needs Clarification'
    reviewer_hash: str
    review_notes: Optional[str] = None
    reviewed_at: Optional[str] = None
    id: Optional[int] = None


@dataclass
class AuditLogModel:
    """Security Audit Log Entry."""
    action: str
    entity_type: str
    entity_id: str
    performed_by_hash: str
    details: Optional[str] = None
    timestamp: Optional[str] = None
