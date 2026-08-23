"""Runtime fail-closed knowledge provenance validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "knowledge"


def load_registry() -> Dict[str, Dict[str, Any]]:
    path = K / "source_registry.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("sources", {})


def validated_chunks() -> List[Dict[str, Any]]:
    registry = load_registry()
    path = K / "kb.json"
    if not path.is_file():
        return []
    chunks = json.loads(path.read_text(encoding="utf-8"))
    verified_sources = set()
    for source_id, source in registry.items():
        document = ROOT / str(source.get("document_path", ""))
        if (
            source.get("activation_state") == "active"
            and document.is_file()
            and hashlib.sha256(document.read_bytes()).hexdigest() == source.get("document_sha256")
        ):
            verified_sources.add(source_id)
    required = ("source_id", "source_section", "source_page", "source_fingerprint", "evidence_type", "evidence_strength")
    return [
        chunk for chunk in chunks
        if chunk.get("validation_state") == "validated_exact_source_match"
        and chunk.get("source_id") in verified_sources
        and all(chunk.get(field) for field in required)
    ]


def citation_for(chunk: Dict[str, Any]) -> Dict[str, Any]:
    source = load_registry().get(chunk.get("source_id"), {})
    return {
        "citation_id": f"KB-{chunk.get('id')}",
        "title": source.get("title"),
        "organization_or_authors": source.get("organization_or_authors"),
        "year": source.get("year"),
        "section": chunk.get("source_section"),
        "page_or_locator": chunk.get("source_page"),
        "doi": source.get("doi"),
        "url": source.get("url"),
        "document_sha256": source.get("document_sha256"),
        "evidence_type": chunk.get("evidence_type"),
        "evidence_strength": chunk.get("evidence_strength"),
    }
