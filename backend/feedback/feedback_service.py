# -*- coding: utf-8 -*-
"""
feedback_service.py

Public API Service for Clinician Feedback Platform.
Serves as the single integration endpoint for UI, CLI, and API clients.
"""

from typing import Dict, Any, List, Optional, Tuple
from backend.feedback.feedback_validation import FeedbackValidator
from backend.feedback.feedback_repository import FeedbackRepository
from backend.feedback.review_queue import ReviewQueueManager
from backend.feedback.feedback_export import FeedbackExporter
from backend.feedback.analytics import FeedbackAnalyticsEngine


class FeedbackService:
    """Unified Service Interface for Clinician Feedback Processing."""

    def __init__(self):
        self.validator = FeedbackValidator()
        self.repository = FeedbackRepository()
        self.review_queue = ReviewQueueManager()
        self.exporter = FeedbackExporter()
        self.analytics = FeedbackAnalyticsEngine()

    def validate_feedback(self, raw_feedback_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """Validates and anonymizes raw feedback payload."""
        return self.validator.validate_and_anonymize(raw_feedback_payload)

    def submit_feedback(self, raw_feedback_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Public single-call endpoint for UI and API clients.
        Validates, anonymizes PII, inserts into append-only repository, and returns status.
        Does NOT mutate active model checkpoints, FAISS index, or knowledge base.
        """
        anonymized_payload, error_msg = self.validator.validate_and_anonymize(raw_feedback_payload)
        if error_msg:
            return {
                "status": "Error",
                "message": error_msg,
                "feedback_id": None
            }

        feedback_id = self.repository.insert_feedback(anonymized_payload)

        return {
            "status": "Success",
            "message": "Clinician feedback submitted successfully and queued for review.",
            "feedback_id": feedback_id,
            "case_id": raw_feedback_payload.get("case_id"),
            "review_status": "Pending"
        }

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Returns all feedback records currently in Pending review status."""
        return self.review_queue.list_queue(status_filter="Pending")

    def review_feedback(
        self,
        feedback_id: int,
        new_status: str,
        reviewer_id: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transitions review status of a feedback record (e.g. Approved, Rejected)."""
        return self.review_queue.update_review_status(
            feedback_id=feedback_id,
            new_status=new_status,
            reviewer_id=reviewer_id,
            notes=notes
        )

    def get_analytics(self) -> Dict[str, Any]:
        """Returns monthly/total feedback analytics summary."""
        return self.analytics.generate_analytics_report()

    def export_approved_data(self) -> Dict[str, str]:
        """Exports approved records for future research and model versions."""
        return self.exporter.export_approved_data()


# Global singleton instance
feedback_service = FeedbackService()

