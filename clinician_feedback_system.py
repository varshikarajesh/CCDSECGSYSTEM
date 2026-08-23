"""TRACE clinician feedback and next-version learning store.

One-file, standard-library implementation for:
  * immutable snapshots of complete pipeline/LLM responses;
  * component-specific clinician feedback;
  * append-only adjudication and audit history;
  * export of accepted feedback for offline dataset construction.

This module never changes a live diagnosis, model, prompt, FAISS index, or KB.
It stores proposed corrections for later review and offline training only.

Examples (Windows CMD/Anaconda terminal):
    python clinician_feedback_system.py init
    python clinician_feedback_system.py record-response --input response.json
    python clinician_feedback_system.py add-feedback --input feedback.json
    python clinician_feedback_system.py review --feedback-id FB-... --status accepted ^
        --reviewer-id cardiologist-02 --notes "Confirmed from source ECG"
    python clinician_feedback_system.py list --kind feedback --status accepted
    python clinician_feedback_system.py export --output outputs\feedback\accepted.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "backend" / "database" / "clinician_feedback.db"
SCHEMA_VERSION = "2.0.0"

COMPONENTS = {
    "acquisition",
    "preprocessing",
    "window_selector",
    "classifier",
    "retriever",
    "holter",
    "statistics",
    "bridge",
    "knowledge",
    "citation",
    "llm_analysis",
    "llm_language",
    "pipeline",
    "final_report",
}

FEEDBACK_TYPES = {
    "label_confirmation",
    "label_correction",
    "missed_label",
    "false_positive_label",
    "finding_status_correction",
    "window_confirmation",
    "window_timing_correction",
    "missed_window",
    "noise_window",
    "stable_reference_correction",
    "retrieval_relevance",
    "measurement_correction",
    "reasoning_quality",
    "confidence_quality",
    "norm_suppression_error",
    "citation_correctness",
    "citation_relevance",
    "citation_provenance_error",
    "answer_correctness",
    "question_relevance",
    "language_quality",
    "too_technical",
    "too_simple",
    "too_verbose",
    "unsupported_claim",
    "missing_limitation",
    "latency_or_runtime",
    "other",
}

REVIEW_STATUSES = {
    "submitted",
    "pending_adjudication",
    "accepted",
    "rejected",
    "disputed",
    "included_in_dataset",
    "included_in_training",
    "evaluated",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def public_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:10]}"


def hash_identifier(identifier: Optional[str]) -> Optional[str]:
    """Pseudonymize an identifier. This is not a substitute for full de-identification."""
    if not identifier:
        return None
    salt = os.environ.get("TRACE_FEEDBACK_SALT", "TRACE-LOCAL-DEVELOPMENT-SALT")
    return hashlib.sha256(f"{salt}:{identifier}".encode("utf-8")).hexdigest()


def require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


class FeedbackStore:
    """Append-only response, feedback, review, and audit repository."""

    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS trace_schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trace_response_snapshots (
            response_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            recording_hash TEXT NOT NULL,
            patient_hash TEXT,
            session_hash TEXT,
            recording_mode TEXT,
            duration_seconds REAL,
            selected_window_ids_json TEXT NOT NULL,
            pipeline_output_json TEXT NOT NULL,
            bridge_decision_json TEXT NOT NULL,
            llm_analysis_json TEXT NOT NULL,
            llm_answer TEXT NOT NULL,
            citations_json TEXT NOT NULL,
            component_versions_json TEXT NOT NULL,
            prompt_metadata_json TEXT NOT NULL,
            performance_json TEXT NOT NULL,
            evidence_snapshot_sha256 TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trace_feedback_events (
            feedback_id TEXT PRIMARY KEY,
            response_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            clinician_hash TEXT NOT NULL,
            clinician_role TEXT,
            component TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            recording_hash TEXT NOT NULL,
            window_ids_json TEXT NOT NULL,
            finding_labels_json TEXT NOT NULL,
            original_output_json TEXT NOT NULL,
            proposed_correction_json TEXT NOT NULL,
            ratings_json TEXT NOT NULL,
            reason_code TEXT,
            comment TEXT,
            evidence_snapshot_sha256 TEXT NOT NULL,
            initial_status TEXT NOT NULL,
            FOREIGN KEY(response_id) REFERENCES trace_response_snapshots(response_id)
        );

        CREATE TABLE IF NOT EXISTS trace_review_events (
            review_event_id TEXT PRIMARY KEY,
            feedback_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            reviewer_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            notes TEXT,
            adjudication_json TEXT NOT NULL,
            FOREIGN KEY(feedback_id) REFERENCES trace_feedback_events(feedback_id)
        );

        CREATE TABLE IF NOT EXISTS trace_audit_events (
            audit_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            actor_hash TEXT,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            details_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_trace_feedback_response
            ON trace_feedback_events(response_id);
        CREATE INDEX IF NOT EXISTS idx_trace_feedback_component
            ON trace_feedback_events(component, feedback_type);
        CREATE INDEX IF NOT EXISTS idx_trace_reviews_feedback
            ON trace_review_events(feedback_id, created_at);

        CREATE TRIGGER IF NOT EXISTS trace_response_no_update
        BEFORE UPDATE ON trace_response_snapshots
        BEGIN SELECT RAISE(ABORT, 'response snapshots are append-only'); END;

        CREATE TRIGGER IF NOT EXISTS trace_response_no_delete
        BEFORE DELETE ON trace_response_snapshots
        BEGIN SELECT RAISE(ABORT, 'response snapshots are append-only'); END;

        CREATE TRIGGER IF NOT EXISTS trace_feedback_no_update
        BEFORE UPDATE ON trace_feedback_events
        BEGIN SELECT RAISE(ABORT, 'feedback events are append-only'); END;

        CREATE TRIGGER IF NOT EXISTS trace_feedback_no_delete
        BEFORE DELETE ON trace_feedback_events
        BEGIN SELECT RAISE(ABORT, 'feedback events are append-only'); END;

        CREATE TRIGGER IF NOT EXISTS trace_review_no_update
        BEFORE UPDATE ON trace_review_events
        BEGIN SELECT RAISE(ABORT, 'review events are append-only'); END;

        CREATE TRIGGER IF NOT EXISTS trace_review_no_delete
        BEFORE DELETE ON trace_review_events
        BEGIN SELECT RAISE(ABORT, 'review events are append-only'); END;
        """
        with self.connect() as conn:
            conn.executescript(schema)
            conn.execute(
                "INSERT OR REPLACE INTO trace_schema_metadata(key,value) VALUES(?,?)",
                ("schema_version", SCHEMA_VERSION),
            )

    def _audit(
        self,
        conn: sqlite3.Connection,
        action: str,
        entity_type: str,
        entity_id: str,
        payload: Any,
        actor_hash: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        conn.execute(
            """INSERT INTO trace_audit_events(
                audit_id,created_at,actor_hash,action,entity_type,entity_id,
                payload_sha256,details_json
            ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                public_id("AUD"), utc_now(), actor_hash, action, entity_type,
                entity_id, json_hash(payload), canonical_json(details or {}),
            ),
        )

    def record_response(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Save one immutable end-to-end pipeline and LLM response snapshot."""
        payload = require_mapping(payload, "response")
        recording_id = str(payload.get("recording_id", "")).strip()
        if not recording_id:
            raise ValueError("recording_id is required")

        bridge = require_mapping(payload.get("bridge_decision", {}), "bridge_decision")
        versions = require_mapping(payload.get("component_versions", {}), "component_versions")
        response_id = str(payload.get("response_id") or public_id("RSP"))
        recording_hash = hash_identifier(recording_id)
        evidence_material = {
            "pipeline_output": payload.get("pipeline_output", {}),
            "bridge_decision": bridge,
            "llm_analysis": payload.get("llm_analysis", {}),
            "citations": payload.get("citations", []),
            "component_versions": versions,
            "prompt_metadata": payload.get("prompt_metadata", {}),
        }
        evidence_hash = json_hash(evidence_material)

        with self.connect() as conn:
            conn.execute(
                """INSERT INTO trace_response_snapshots VALUES(
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )""",
                (
                    response_id,
                    utc_now(),
                    recording_hash,
                    hash_identifier(payload.get("patient_id")),
                    hash_identifier(payload.get("session_id")),
                    payload.get("recording_mode"),
                    payload.get("duration_seconds"),
                    canonical_json(payload.get("selected_window_ids", [])),
                    canonical_json(payload.get("pipeline_output", {})),
                    canonical_json(bridge),
                    canonical_json(payload.get("llm_analysis", {})),
                    str(payload.get("llm_answer", "")),
                    canonical_json(payload.get("citations", [])),
                    canonical_json(versions),
                    canonical_json(payload.get("prompt_metadata", {})),
                    canonical_json(payload.get("performance", {})),
                    evidence_hash,
                ),
            )
            self._audit(conn, "RECORD_RESPONSE", "response", response_id, evidence_material)

        return {
            "response_id": response_id,
            "recording_hash": recording_hash,
            "evidence_snapshot_sha256": evidence_hash,
            "stored": True,
        }

    def add_feedback(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Append clinician feedback without changing the saved machine response."""
        payload = require_mapping(payload, "feedback")
        response_id = str(payload.get("response_id", "")).strip()
        clinician_id = str(payload.get("clinician_id", "")).strip()
        component = str(payload.get("component", "")).strip().lower()
        feedback_type = str(payload.get("feedback_type", "")).strip().lower()
        if not response_id or not clinician_id:
            raise ValueError("response_id and clinician_id are required")
        if component not in COMPONENTS:
            raise ValueError(f"component must be one of: {sorted(COMPONENTS)}")
        if feedback_type not in FEEDBACK_TYPES:
            raise ValueError(f"feedback_type must be one of: {sorted(FEEDBACK_TYPES)}")

        clinician_hash = hash_identifier(clinician_id)
        feedback_id = str(payload.get("feedback_id") or public_id("FB"))
        with self.connect() as conn:
            response = conn.execute(
                "SELECT recording_hash,evidence_snapshot_sha256 FROM trace_response_snapshots WHERE response_id=?",
                (response_id,),
            ).fetchone()
            if response is None:
                raise ValueError(f"unknown response_id: {response_id}")

            conn.execute(
                """INSERT INTO trace_feedback_events VALUES(
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )""",
                (
                    feedback_id,
                    response_id,
                    utc_now(),
                    clinician_hash,
                    payload.get("clinician_role"),
                    component,
                    feedback_type,
                    response["recording_hash"],
                    canonical_json(payload.get("window_ids", [])),
                    canonical_json(payload.get("finding_labels", [])),
                    canonical_json(payload.get("original_output", {})),
                    canonical_json(payload.get("proposed_correction", {})),
                    canonical_json(payload.get("ratings", {})),
                    payload.get("reason_code"),
                    payload.get("comment"),
                    response["evidence_snapshot_sha256"],
                    "pending_adjudication",
                ),
            )
            self._audit(
                conn, "ADD_FEEDBACK", "feedback", feedback_id, payload,
                actor_hash=clinician_hash,
                details={"component": component, "feedback_type": feedback_type},
            )

        return {
            "feedback_id": feedback_id,
            "response_id": response_id,
            "status": "pending_adjudication",
            "production_decision_modified": False,
        }

    def review_feedback(
        self,
        feedback_id: str,
        status: str,
        reviewer_id: str,
        notes: Optional[str] = None,
        adjudication: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        status = status.strip().lower()
        if status not in REVIEW_STATUSES:
            raise ValueError(f"status must be one of: {sorted(REVIEW_STATUSES)}")
        if status == "submitted":
            raise ValueError("submitted is an initial state, not an adjudication result")
        reviewer_hash = hash_identifier(reviewer_id)
        review_event_id = public_id("REV")
        with self.connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM trace_feedback_events WHERE feedback_id=?", (feedback_id,)
            ).fetchone()
            if not exists:
                raise ValueError(f"unknown feedback_id: {feedback_id}")
            conn.execute(
                "INSERT INTO trace_review_events VALUES(?,?,?,?,?,?,?)",
                (
                    review_event_id, feedback_id, utc_now(), reviewer_hash,
                    status, notes, canonical_json(adjudication or {}),
                ),
            )
            self._audit(
                conn, "REVIEW_FEEDBACK", "feedback", feedback_id,
                {"status": status, "notes": notes, "adjudication": adjudication or {}},
                actor_hash=reviewer_hash,
            )
        return {
            "review_event_id": review_event_id,
            "feedback_id": feedback_id,
            "status": status,
            "production_decision_modified": False,
        }

    def current_feedback_rows(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        query = """
        SELECT f.*,
               COALESCE(
                 (SELECT r.status FROM trace_review_events r
                  WHERE r.feedback_id=f.feedback_id
                  ORDER BY r.created_at DESC, r.rowid DESC LIMIT 1),
                 f.initial_status
               ) AS current_status
        FROM trace_feedback_events f
        """
        params: List[Any] = []
        if status:
            query = f"SELECT * FROM ({query}) WHERE current_status=?"
            params.append(status.lower())
        query += " ORDER BY created_at"
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def response_rows(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            return [
                dict(row) for row in conn.execute(
                    "SELECT * FROM trace_response_snapshots ORDER BY created_at"
                ).fetchall()
            ]

    def export_accepted(self, output: Path | str) -> Dict[str, Any]:
        """Export accepted feedback only; this is not yet a training-ready dataset."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = self.current_feedback_rows("accepted")
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "record_count": len(rows),
            "output": str(output.resolve()),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "warning": "Accepted feedback still requires de-identification, patient-disjoint splitting, leakage checks, and dataset approval before training.",
        }
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


def load_json(path: str) -> Mapping[str, Any]:
    if path == "-":
        return require_mapping(json.load(sys.stdin), "stdin")
    with Path(path).open("r", encoding="utf-8") as handle:
        return require_mapping(json.load(handle), path)


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("init", help="Create/verify the database schema")

    for name in ("record-response", "add-feedback"):
        command = subs.add_parser(name)
        command.add_argument("--input", required=True, help="JSON path, or - for stdin")

    review = subs.add_parser("review")
    review.add_argument("--feedback-id", required=True)
    review.add_argument("--status", required=True, choices=sorted(REVIEW_STATUSES - {"submitted"}))
    review.add_argument("--reviewer-id", required=True)
    review.add_argument("--notes")

    listing = subs.add_parser("list")
    listing.add_argument("--kind", choices=("feedback", "responses"), default="feedback")
    listing.add_argument("--status", choices=sorted(REVIEW_STATUSES))

    export = subs.add_parser("export")
    export.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    store = FeedbackStore(args.db)
    if args.command == "init":
        print_json({"database": str(store.db_path.resolve()), "schema_version": SCHEMA_VERSION, "ready": True})
    elif args.command == "record-response":
        print_json(store.record_response(load_json(args.input)))
    elif args.command == "add-feedback":
        print_json(store.add_feedback(load_json(args.input)))
    elif args.command == "review":
        print_json(store.review_feedback(args.feedback_id, args.status, args.reviewer_id, args.notes))
    elif args.command == "list":
        rows = store.response_rows() if args.kind == "responses" else store.current_feedback_rows(args.status)
        print_json(rows)
    elif args.command == "export":
        print_json(store.export_accepted(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
