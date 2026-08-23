# -*- coding: utf-8 -*-
"""
feedback_repository.py

Append-Only Database Repository for Clinician Feedback Platform.
Provides insert, query, and audit trail functionality using DatabaseConnection.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from backend.database.connection import get_db_connection, DatabaseConnection
from backend.database.models import FeedbackRecordModel, ReviewQueueModel, AuditLogModel


class FeedbackRepository:
    """Append-only Repository for Clinician Feedback & Audit Trails."""

    def __init__(self):
        self.db = DatabaseConnection()
        self.db.init_db()

    def insert_feedback(self, record: Dict[str, Any]) -> int:
        """Inserts a new append-only feedback record and returns the generated feedback_id."""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        retrieval_json = json.dumps(record.get("retrieval_evaluations", []))
        sec_scps_str = ",".join(record.get("clinician_secondary_scps", [])) if isinstance(record.get("clinician_secondary_scps"), list) else str(record.get("clinician_secondary_scps", ""))

        cursor.execute("""
        INSERT INTO feedback_records (
            case_id, timestamp, ecg_id, patient_hash, clinician_hash,
            diagnosis_correctness, clinician_primary_scp, clinician_secondary_scps,
            clinician_family, confidence_rating, bridge_explanation_rating,
            retrieval_evaluations_json, general_comments, classifier_version,
            family_head_version, retrieval_version, bridge_version, faiss_version,
            signal_quality, confidence_score, deployment_version, review_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            record["case_id"],
            record.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            record["ecg_id"],
            record["patient_hash"],
            record["clinician_hash"],
            record["diagnosis_correctness"],
            record.get("clinician_primary_scp"),
            sec_scps_str,
            record.get("clinician_family"),
            record.get("confidence_rating"),
            record.get("bridge_explanation_rating"),
            retrieval_json,
            record.get("general_comments"),
            record.get("classifier_version", "v5.0"),
            record.get("family_head_version", "v1.0"),
            record.get("retrieval_version", "v1.0"),
            record.get("bridge_version", "v3.0"),
            record.get("faiss_version", "v1.0"),
            float(record.get("signal_quality", 1.0)),
            int(record.get("confidence_score", 90)),
            record.get("deployment_version", "v7.0"),
            "Pending"
        ))

        feedback_id = cursor.lastrowid

        # Insert audit log
        cursor.execute("""
        INSERT INTO audit_logs (action, entity_type, entity_id, performed_by_hash, details)
        VALUES (?, ?, ?, ?, ?);
        """, ("CREATE_FEEDBACK", "feedback_records", str(feedback_id), record["clinician_hash"], f"Submitted feedback for case {record['case_id']}"))

        conn.commit()
        conn.close()
        return feedback_id

    def get_feedback_by_id(self, feedback_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single feedback record by ID."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback_records WHERE id = ?;", (feedback_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_approved_feedback(self) -> List[Dict[str, Any]]:
        """Retrieves all feedback records with review_status == 'Approved'."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback_records WHERE review_status = 'Approved';")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """Retrieves all feedback records regardless of status."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback_records;")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
