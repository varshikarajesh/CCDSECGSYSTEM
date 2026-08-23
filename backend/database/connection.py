# -*- coding: utf-8 -*-
"""
backend/database/connection.py

Database Connection Module for Clinician Feedback Platform.
Provides SQLite database engine and session management with easy PostgreSQL migration path.
"""

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = PACKAGE_ROOT / "backend" / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "clinician_feedback.db"


class DatabaseConnection:
    """Provides thread-safe connection handling for SQLite/PostgreSQL."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH

    def get_connection(self) -> sqlite3.Connection:
        """Returns a SQLite connection with row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes database schema tables if they do not exist."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Feedback Records Table (Append-only)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            ecg_id TEXT NOT NULL,
            patient_hash TEXT NOT NULL,
            clinician_hash TEXT NOT NULL,
            diagnosis_correctness TEXT NOT NULL,
            clinician_primary_scp TEXT,
            clinician_secondary_scps TEXT,
            clinician_family TEXT,
            confidence_rating TEXT,
            bridge_explanation_rating TEXT,
            retrieval_evaluations_json TEXT,
            general_comments TEXT,
            classifier_version TEXT NOT NULL,
            family_head_version TEXT NOT NULL,
            retrieval_version TEXT NOT NULL,
            bridge_version TEXT NOT NULL,
            faiss_version TEXT NOT NULL,
            signal_quality REAL NOT NULL,
            confidence_score INTEGER NOT NULL,
            deployment_version TEXT NOT NULL,
            review_status TEXT NOT NULL DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Review Queue State Transitions Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS review_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER NOT NULL,
            review_status TEXT NOT NULL,
            reviewer_hash TEXT NOT NULL,
            review_notes TEXT,
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (feedback_id) REFERENCES feedback_records (id)
        );
        """)

        # Audit Logs Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            performed_by_hash TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()
        conn.close()


def get_db_connection() -> sqlite3.Connection:
    db = DatabaseConnection()
    return db.get_connection()
