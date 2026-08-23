# -*- coding: utf-8 -*-
"""
review_queue.py

Clinical Review Queue Workflow Engine.
Manages state transitions for clinician feedback:
Pending -> Approved | Rejected | Needs Clarification
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from backend.database.connection import DatabaseConnection
from backend.feedback.feedback_validation import FeedbackValidator


class ReviewQueueManager:
    """Manages state transitions and auditing for clinical review of feedback records."""

    ALLOWED_STATUSES = {"Pending", "Approved", "Rejected", "Needs Clarification"}

    def __init__(self):
        self.db = DatabaseConnection()

    def update_review_status(
        self,
        feedback_id: int,
        new_status: str,
        reviewer_id: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transitions a feedback record's status and logs the review action.
        """
        if new_status not in self.ALLOWED_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of {self.ALLOWED_STATUSES}")

        reviewer_hash = FeedbackValidator.hash_identifier(reviewer_id)

        conn = self.db.get_connection()
        cursor = conn.cursor()

        # Update feedback_records table status
        cursor.execute("""
        UPDATE feedback_records SET review_status = ? WHERE id = ?;
        """, (new_status, feedback_id))

        # Insert transition in review_queue table
        cursor.execute("""
        INSERT INTO review_queue (feedback_id, review_status, reviewer_hash, review_notes)
        VALUES (?, ?, ?, ?);
        """, (feedback_id, new_status, reviewer_hash, notes))

        # Insert audit log
        cursor.execute("""
        INSERT INTO audit_logs (action, entity_type, entity_id, performed_by_hash, details)
        VALUES (?, ?, ?, ?, ?);
        """, ("REVIEW_STATUS_UPDATE", "feedback_records", str(feedback_id), reviewer_hash, f"Status updated to '{new_status}'. Notes: {notes}"))

        conn.commit()
        conn.close()

        return {
            "feedback_id": feedback_id,
            "new_status": new_status,
            "reviewer_hash": reviewer_hash,
            "review_notes": notes,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": f"Feedback ID {feedback_id} status updated to '{new_status}'."
        }

    def list_queue(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists records in the review queue, optionally filtered by status."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        if status_filter:
            cursor.execute("SELECT * FROM feedback_records WHERE review_status = ?;", (status_filter,))
        else:
            cursor.execute("SELECT * FROM feedback_records;")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
