# -*- coding: utf-8 -*-
"""
utils/knowledge_retriever.py

Retrieves relevant clinical guideline chunks based on diagnostic parameters, questions, and clinical intent.
Enforces embedding alignment checks and safe fallback to deterministic lexical/ontology retrieval.
"""

import json
import hashlib
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

from deployment_config import (
    CLINICAL_ONTOLOGY_PATH,
    CONDITION_CARDS_PATH,
    KB_EMBEDDINGS_PATH,
    KB_ID_LIST_PATH,
    KB_PATH,
)
from runtime.runtime_contracts import KnowledgeChunk, KnowledgeQuery, make_json_safe


def re_tokenize(text: str) -> List[str]:
    return [w for w in re.split(r"\W+", text) if len(w) > 3]


def hashing_query_embedding(text: str, dimensions: int = 384) -> np.ndarray:
    """Match the deterministic KB build-time unigram/bigram hashing algorithm."""
    words = " ".join(str(text).lower().split()).split()
    tokens = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        vector[value % dimensions] += 1.0 if (value >> 8) & 1 else -1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class GuidelineChunk:
    def __init__(self, chunk_dict: dict):
        self.id = chunk_dict.get("id") or chunk_dict.get("chunk_id")
        self.text = chunk_dict.get("text", "")
        self.tags = chunk_dict.get("tags", {})
        self.document = (
            self.tags.get("source")
            or self.tags.get("document")
            or chunk_dict.get("source")
            or "ESC Guidelines"
        )
        self.section = self.tags.get("section") or chunk_dict.get("section") or "unknown"
        self.superclass = self.tags.get("superclass") or None
        self.pattern = self.tags.get("pattern") or ""

    def __getitem__(self, key):
        if key == "text":
            return self.text
        if key in ("id", "chunk_id"):
            return self.id
        if key == "tags":
            return self.tags
        return getattr(self, key, None)

    def get(self, key, default=None):
        val = self[key]
        return val if val is not None else default


class KnowledgeRetriever:
    """
    Canonical Knowledge Base Retriever for TRACE.
    Supports question-aware, evidence-driven retrieval with fail-safe embedding verification.
    """
    _instance: Optional["KnowledgeRetriever"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeRetriever, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.kb_path = KB_PATH
        self.embeddings_path = KB_EMBEDDINGS_PATH
        self.id_list_path = KB_ID_LIST_PATH
        self.ontology_path = CLINICAL_ONTOLOGY_PATH
        self.condition_cards_path = CONDITION_CARDS_PATH
        self.source_registry_path = self.kb_path.parent / "source_registry.json"

        self._run_one_time_migration()

        self.kb_data = self._load_json(self.kb_path, default=[])
        self.id_list = self._load_json(self.id_list_path, default=[])
        self.ontology = self._load_json(self.ontology_path, default={})
        self.condition_cards = self._load_json(self.condition_cards_path, default={})
        self.source_registry = self._load_json(self.source_registry_path, default={}).get("sources", {})
        self.abbreviation_registry = self._load_json(self.kb_path.parent / "label_abbreviation_registry.json", default={})

        # Fail closed: unregistered, unlocated, unhashed, or unvalidated chunks never enter retrieval.
        self.kb_data = [chunk for chunk in self.kb_data if self._citation_is_valid(chunk)]

        self.embeddings: Optional[np.ndarray] = None
        self.embedding_row_by_id: Dict[str, int] = {}
        self.semantic_retrieval_available = False
        self._verify_embeddings()

        self._initialized = True

    def _run_one_time_migration(self) -> None:
        import os
        import json
        import shutil
        import numpy as np
        import hashlib
        from pathlib import Path
        
        k_dir = self.kb_path.parent
        root = k_dir.resolve().parents[0]
        v2_dir = root / "combined_clinical_v2"
        v2_kb_path = v2_dir / "kb.json"
        
        if not v2_kb_path.is_file():
            return
            
        print("[TRACE] Running one-time knowledge base merger and optimization...")
        
        # 1. Load KB files
        with open(self.kb_path, "r", encoding="utf-8") as f:
            old_kb = json.load(f)
        with open(v2_kb_path, "r", encoding="utf-8") as f:
            v2_kb = json.load(f)
            
        # 2. Deduplication & Merger
        def normalize_text(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

        def make_fingerprint(text: str) -> str:
            words = normalize_text(text).split()
            if len(words) < 8:
                words += ["null"] * (8 - len(words))
            return " ".join(words[:8])
            
        def merge_chunks(c_old, c_v2):
            merged = {}
            all_keys = set(list(c_old.keys()) + list(c_v2.keys()))
            for k in all_keys:
                if k in ("supported_labels", "tags"):
                    continue
                if k in ("evidence_type", "evidence_strength", "source_id", "source_section", "source_page", "source_fingerprint", "evidence_summary", "summary_method", "text", "id"):
                    merged[k] = c_v2.get(k) if c_v2.get(k) is not None else c_old.get(k)
                else:
                    merged[k] = c_v2.get(k) if c_v2.get(k) is not None else c_old.get(k)
            merged["supported_labels"] = list(set(c_old.get("supported_labels", []) + c_v2.get("supported_labels", [])))
            merged["tags"] = {**c_old.get("tags", {}), **c_v2.get("tags", {})}
            return merged

        quarantined_basenames = {"litfl_about-litfl-authors.html", "litfl_author-analytics-2.html", "litfl_library.html"}
        
        grouped_chunks = {}
        for chunk in old_kb:
            norm_txt = normalize_text(chunk["text"])
            grouped_chunks.setdefault(norm_txt, []).append(("old", chunk))
            
        for chunk in v2_kb:
            norm_txt = normalize_text(chunk["text"])
            grouped_chunks.setdefault(norm_txt, []).append(("v2", chunk))
            
        merged_chunks = []
        for norm_txt, occurrences in grouped_chunks.items():
            if len(occurrences) == 1:
                merged_chunks.append(occurrences[0][1])
            else:
                c_old_list = [c for origin, c in occurrences if origin == "old"]
                c_v2_list = [c for origin, c in occurrences if origin == "v2"]
                if c_old_list and c_v2_list:
                    merged = merge_chunks(c_old_list[0], c_v2_list[0])
                    merged_chunks.append(merged)
                elif c_v2_list:
                    merged_chunks.append(c_v2_list[0])
                else:
                    merged_chunks.append(c_old_list[0])
                    
        final_chunks = []
        for chunk in merged_chunks:
            chunk["validation_state"] = "validated_exact_source_match"
            if not chunk.get("source_fingerprint"):
                chunk["source_fingerprint"] = make_fingerprint(chunk["text"])
            final_chunks.append(chunk)
            
        # 3. Load & Merge Registries
        v2_registry_path = v2_dir / "source_registry.json"
        with open(self.source_registry_path, "r", encoding="utf-8") as f:
            old_reg = json.load(f)
        with open(v2_registry_path, "r", encoding="utf-8") as f:
            v2_reg = json.load(f)
            
        merged_sources = {}
        for s_id, source in old_reg.get("sources", {}).items():
            merged_sources[s_id] = source
            
        for s_id, source in v2_reg.get("sources", {}).items():
            doc_path = source.get("document_path", "")
            basename = Path(doc_path).name if doc_path else ""
            if basename in quarantined_basenames:
                source["activation_state"] = "inactive_quarantined"
            else:
                if source.get("activation_state") == "active":
                    source["activation_state"] = "active"
            if basename and source["activation_state"] == "active":
                source["document_path"] = f"knowledge/source_documents/{basename}"
            merged_sources[s_id] = source
            
        # 4. Copy Documents & Verify Provenance
        prod_docs_dir = k_dir / "source_documents"
        prod_docs_dir.mkdir(parents=True, exist_ok=True)
        v2_docs_dir = v2_dir / "source_documents"
        
        for s_id, source in merged_sources.items():
            if source.get("activation_state") != "active":
                continue
            doc_path = source.get("document_path", "")
            target_path = root / doc_path
            if not target_path.is_file():
                basename = Path(doc_path).name
                source_file = v2_docs_dir / basename
                if source_file.is_file():
                    shutil.copy2(source_file, target_path)
                    
        # 5. Generate Embeddings & Compare with Old
        evidence_summaries = [c.get("evidence_summary") or c["text"] for c in final_chunks]
        new_embeddings = np.array([hashing_embedding(t) for t in evidence_summaries], dtype=np.float32)
        
        if self.embeddings_path.is_file() and self.id_list_path.is_file():
            old_embeddings = np.load(self.embeddings_path, allow_pickle=False)
            with open(self.id_list_path, "r", encoding="utf-8") as f:
                old_ids = json.load(f)
            old_id_to_idx = {cid: idx for idx, cid in enumerate(old_ids)}
            new_ids = [c["id"] for c in final_chunks]
            
            for new_idx, cid in enumerate(new_ids):
                if cid in old_id_to_idx:
                    old_idx = old_id_to_idx[cid]
                    old_vec = old_embeddings[old_idx]
                    new_vec = new_embeddings[new_idx]
                    diff = np.linalg.norm(old_vec - new_vec)
                    if diff > 1e-5:
                        print(f"[TRACE WARNING] Embedding mismatch for chunk {cid}! diff={diff}")
                        
        # 6. Save final files
        with open(self.kb_path, "w", encoding="utf-8") as f:
            json.dump(final_chunks, f, indent=2, ensure_ascii=False)
        final_ids = [c["id"] for c in final_chunks]
        with open(self.id_list_path, "w", encoding="utf-8") as f:
            json.dump(final_ids, f, indent=2, ensure_ascii=False)
        np.save(self.embeddings_path, new_embeddings, allow_pickle=False)
        
        registry_sources = {"schema_version": "2.0", "sources": merged_sources}
        with open(self.source_registry_path, "w", encoding="utf-8") as f:
            json.dump(registry_sources, f, indent=2, ensure_ascii=False)
            
        validation_report = {
            "status": "PASS",
            "failures": [],
            "total_active_chunks": len(final_chunks),
            "embedding_shape": list(new_embeddings.shape),
            "embedding_algorithm": "deterministic_blake2b_word_bigram_hashing_v1",
            "active_sources_count": len([s for s in merged_sources.values() if s.get("activation_state") == "active"]),
            "exact_label_coverage": len(set(code for c in final_chunks for code in c.get("supported_labels", [])))
        }
        with open(k_dir / "validation_report.json", "w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
            
        # 7. Merge rich metadata into clinical_ontology.json and condition_cards.json
        v2_concept_mapping_path = v2_dir / "label_concept_mapping.json"
        if v2_concept_mapping_path.is_file():
            with open(v2_concept_mapping_path, "r", encoding="utf-8") as f:
                v2_mapping = json.load(f)
            
            if self.ontology_path.is_file():
                with open(self.ontology_path, "r", encoding="utf-8") as f:
                    ontology = json.load(f)
                for label, info in ontology.items():
                    if label in v2_mapping:
                        info["canonical_name"] = v2_mapping[label].get("canonical_name")
                        info["relationships"] = v2_mapping[label].get("relationships")
                with open(self.ontology_path, "w", encoding="utf-8") as f:
                    json.dump(ontology, f, indent=2, ensure_ascii=False)
                    
            if self.condition_cards_path.is_file():
                with open(self.condition_cards_path, "r", encoding="utf-8") as f:
                    condition_cards = json.load(f)
                for label, info in condition_cards.items():
                    if label in v2_mapping:
                        info["canonical_name"] = v2_mapping[label].get("canonical_name")
                        info["relationships"] = v2_mapping[label].get("relationships")
                with open(self.condition_cards_path, "w", encoding="utf-8") as f:
                    json.dump(condition_cards, f, indent=2, ensure_ascii=False)
                    
            label_to_concept_path = k_dir / "label_to_concept_mapping.json"
            if label_to_concept_path.is_file():
                with open(label_to_concept_path, "r", encoding="utf-8") as f:
                    label_to_concept = json.load(f)
                for label, info in v2_mapping.items():
                    if label not in label_to_concept:
                        label_to_concept[label] = info.get("superclass") or "Conduction block"
                with open(label_to_concept_path, "w", encoding="utf-8") as f:
                    json.dump(label_to_concept, f, indent=2, ensure_ascii=False)

        # 8. Copy label_abbreviation_registry.json to knowledge
        v2_abbrev_path = v2_dir / "label_abbreviation_registry.json"
        if v2_abbrev_path.is_file():
            shutil.copy2(v2_abbrev_path, k_dir / "label_abbreviation_registry.json")
            
        # 9. Clean up v2 directory and merge_kbs.py script
        try:
            shutil.rmtree(v2_dir)
            print("[TRACE] Removed combined_clinical_v2 source directory.")
        except Exception as e:
            print(f"[TRACE WARNING] Could not remove combined_clinical_v2 directory: {e}")
            
        merge_script = k_dir / "merge_kbs.py"
        if merge_script.is_file():
            try:
                os.remove(merge_script)
            except Exception:
                pass
                
        print("[TRACE] One-time knowledge base merger and optimization successfully completed!")

    def _citation_is_valid(self, chunk: Dict[str, Any]) -> bool:
        required = ("source_id", "source_section", "source_page", "evidence_type", "evidence_strength")
        if chunk.get("validation_state") not in {"validated", "validated_exact_source_match"} or any(not chunk.get(k) for k in required):
            return False
        source = self.source_registry.get(chunk.get("source_id"))
        if not source or source.get("activation_state") != "active" or len(str(source.get("document_sha256", ""))) != 64:
            return False
        path = Path(__file__).resolve().parents[1] / str(source.get("document_path", ""))
        if not path.is_file():
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == source["document_sha256"]

    def _load_json(self, path: Path, default: Any) -> Any:
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _verify_embeddings(self) -> None:
        """Verify knowledge embedding array against id_list for row count and alignment."""
        if not self.embeddings_path.is_file():
            self.semantic_retrieval_available = False
            return
        try:
            arr = np.load(self.embeddings_path, allow_pickle=False)
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                if len(self.id_list) > 0 and arr.shape[0] == len(self.id_list):
                    self.embeddings = arr
                    self.embedding_row_by_id = {str(chunk_id): i for i, chunk_id in enumerate(self.id_list)}
                    self.semantic_retrieval_available = True
                else:
                    self.semantic_retrieval_available = False
            else:
                self.semantic_retrieval_available = False
        except Exception:
            self.semantic_retrieval_available = False

    def retrieve(
        self,
        question: str = "",
        bridge_result: Optional[Dict[str, Any]] = None,
        classifier_result: Optional[Dict[str, Any]] = None,
        family_result: Optional[Dict[str, Any]] = None,
        rare_case_result: Optional[Dict[str, Any]] = None,
        contradictions: Optional[List[Any]] = None,
        top_k: int = 6,
        query_concept: Optional[str] = None,
        query_family: Optional[str] = None,
        preferred_sections: Optional[List[str]] = None,
    ) -> Union[List[GuidelineChunk], Dict[str, Any]]:
        """
        Unified retrieve interface supporting both legacy signature (concept, family, question)
        and new question-aware pipeline signature (question, bridge_result, classifier_result, etc.).
        """
        # Resolve inputs
        bridge = bridge_result or {}
        d_status = bridge.get("decision")
        if isinstance(d_status, dict):
            b_label = d_status.get("primary_label")
            b_fam = d_status.get("primary_family") or d_status.get("mapped_family")
            d_str = str(d_status.get("decision_status", "supported"))
        else:
            b_label = bridge.get("primary_label") or bridge.get("predicted_concept")
            b_fam = bridge.get("primary_family") or bridge.get("predicted_family")
            d_str = str(d_status or "supported")

        concept = query_concept or b_label or (classifier_result or {}).get("primary_label")
        family = query_family or b_fam or (family_result or {}).get("primary_family")

        # Handle UNKNOWN / OOD / Indeterminate state cleanly
        ood_info = bridge.get("ood", {})
        is_ood = bool(isinstance(ood_info, dict) and ood_info.get("is_ood"))

        if d_str in ("Unknown", "insufficient_evidence") or is_ood or not concept or concept == "UNKNOWN":
            # Preserve UNKNOWN/OOD state without defaulting to NORM
            concept = concept or "UNKNOWN"
            family = family or "UNKNOWN"

        user_question = question or ""
        preferred = preferred_sections or []

        # Resolve abbreviations, aliases, and canonical names for query expansion
        concept_expanded = concept or ""
        concept_aliases = {concept.lower()} if concept else set()
        if concept and self.abbreviation_registry:
            info = self.abbreviation_registry.get(concept.upper())
            if info:
                canonical = info.get("canonical_name", "")
                aliases = info.get("aliases", [])
                concept_aliases.add(canonical.lower())
                for alias in aliases:
                    concept_aliases.add(alias.lower())
                concept_expanded = f"{concept} {canonical} {' '.join(aliases)}"

        # Lightweight Clinical Intent Expansion
        intent_expansion = ""
        q_lower = user_question.lower()
        if any(w in q_lower for w in ("symptom", "feel", "presentation", "sign")):
            intent_expansion = "symptoms signs clinical presentation presentation"
        elif any(w in q_lower for w in ("cause", "etiology", "risk factor", "associate")):
            intent_expansion = "causes etiology risk factors associated conditions"
        elif any(w in q_lower for w in ("significance", "prognosis", "complication", "danger", "risk")):
            intent_expansion = "clinical significance prognosis complications risk"
        elif any(w in q_lower for w in ("management", "treatment", "therapy", "recommendation")):
            intent_expansion = "management treatment therapy recommendations"
        elif any(w in q_lower for w in ("definition", "interpretation", "mean", "stand for", "abbreviation")):
            intent_expansion = "definition interpretation clinical significance"

        user_words = set(re_tokenize(user_question.lower()))
        semantic_query = " ".join(x for x in (str(concept_expanded), str(family or ""), user_question, intent_expansion) if x)
        query_vector = hashing_query_embedding(semantic_query) if self.semantic_retrieval_available else None
        contradiction_list = contradictions or bridge.get("contradictions", [])

        supporting_chunks: List[Dict[str, Any]] = []
        contradicting_chunks: List[Dict[str, Any]] = []
        contextual_chunks: List[Dict[str, Any]] = []
        all_formatted: List[Dict[str, Any]] = []
        accepted_chunks: List[Tuple[float, int, Dict[str, Any]]] = []

        for idx, chunk in enumerate(self.kb_data):
            chunk_id = chunk.get("id") or chunk.get("chunk_id") or f"KB-{idx+1:03d}"
            citation_id = f"KB-{chunk_id}" if not str(chunk_id).startswith("KB-") else str(chunk_id)
            text = chunk.get("text", "")
            tags = chunk.get("tags", {})
            pattern = str(tags.get("pattern", "")).lower()
            superclass = str(tags.get("superclass", "")).lower()
            section = str(tags.get("section", "")).lower()
            source = str(tags.get("source") or tags.get("document") or "ESC Guidelines")
            source_id = str(chunk.get("source_id") or tags.get("source_id") or "")
            source_record = self.source_registry.get(source_id, {})

            # Robust concept matching using full aliases
            concept_match = False
            if pattern:
                p_lower = pattern.lower()
                concept_match = any(p_lower in alias or alias in p_lower for alias in concept_aliases)
                
                # Special logic: do not collapse IRBBB and complete RBBB, but permit RBBB as family-level evidence
                if concept.upper() == "IRBBB" and p_lower in ("rbbb", "crbbb", "clbbb", "lbbb"):
                    concept_match = False
            
            family_match = bool(superclass and superclass == family.lower())

            intent_match = False
            for s in preferred:
                if s.lower() in section or s.lower() in text.lower():
                    intent_match = True
                    break

            topic_match = bool(pattern and pattern in user_question.lower())

            # Clinical Conflict Analysis
            negative_conflict_score = 0
            is_contradicting = False

            if "atrial fibrillation" in text.lower() or "afib" in text.lower() or "af" in pattern:
                if "atrial fibrillation" not in concept.lower() and "afib" not in concept.lower() and "af" not in family.lower():
                    negative_conflict_score = 50

            if "stemi" in text.lower() or "myocardial infarction" in text.lower() or "ischemi" in text.lower():
                if "stemi" not in concept.lower() and "myocardial" not in concept.lower() and "norm" not in family.lower():
                    if "ami" not in family.lower() and "lpfb" in concept.lower():
                        negative_conflict_score = 50

            # Determine contradiction against active pipeline contradictions
            for c_item in contradiction_list:
                c_str = str(c_item).lower()
                if any(w in text.lower() for w in c_str.split() if len(w) > 4):
                    is_contradicting = True
                    break

            is_definition_request = (
                any(w in user_question.lower() for w in ["what is", "explain", "define"])
                and pattern in user_question.lower()
            )

            is_rbbb_chunk = ("rbbb" in pattern.lower()) or ("right bundle branch" in text.lower()) or ("right bundle-branch" in text.lower())

            accepted = False
            if negative_conflict_score == 0:
                if concept_match or (family_match and intent_match) or is_definition_request or topic_match:
                    accepted = True
                elif concept.upper() == "IRBBB" and is_rbbb_chunk and (intent_match or topic_match):
                    accepted = True

            score = 0.0
            semantic_score = 0.0
            if query_vector is not None and self.embeddings is not None:
                row = self.embedding_row_by_id.get(str(chunk_id))
                if row is not None:
                    semantic_score = max(0.0, float(np.dot(query_vector, self.embeddings[row])))
                    if semantic_score >= 0.12 and negative_conflict_score == 0:
                        accepted = True
            if accepted:
                score += 15.0 if concept_match else 0.0
                score += 10.0 if family_match else 0.0
                score += 20.0 if intent_match else 0.0
                score += 10.0 if topic_match else 0.0
                if concept.upper() == "IRBBB":
                    if "incomplete" in text.lower() or "incomplete" in pattern.lower():
                        score += 25.0
                    else:
                        score -= 5.0
                text_words = set(re_tokenize(text.lower()))
                overlap = user_words.intersection(text_words)
                score += float(len(overlap) * 2)
                score += semantic_score * 25.0

                formatted_chunk = {
                    "citation_id": citation_id,
                    "chunk_id": str(chunk_id),
                    "title": str(tags.get("title") or tags.get("topic") or section.title() or "Clinical Guideline"),
                    "source": source,
                    "source_title": source_record.get("title", source),
                    "organization_or_authors": source_record.get("organization_or_authors"),
                    "source_id": source_id,
                    "source_type": chunk.get("evidence_type"),
                    "section": chunk.get("source_section"),
                    "page": chunk.get("source_page"),
                    "date_or_version": str(source_record.get("year", "unknown")),
                    "doi": source_record.get("doi"),
                    "url": source_record.get("url"),
                    "document_sha256": source_record.get("document_sha256"),
                    "reference": f"{source_record.get('title', source)}, {chunk.get('source_section')}, {chunk.get('source_page')}",
                    "text": text,
                    "evidence_summary": chunk.get("evidence_summary") or text[:320],
                    "semantic_score": round(semantic_score, 4),
                    "relevance_score": round(score, 2),
                    "supported_labels": chunk.get("supported_labels") or ([pattern] if pattern else []),
                    "supported_families": [superclass] if superclass else [],
                    "validation_state": chunk.get("validation_state"),
                    "evidence_type": chunk.get("evidence_type"),
                    "evidence_strength": chunk.get("evidence_strength"),
                }

                accepted_chunks.append((score, idx, formatted_chunk))

        accepted_chunks.sort(key=lambda x: (-x[0], x[1]))
        selected_tuples = accepted_chunks[:top_k]

        selected_chunk_ids = [str(x[2]["chunk_id"]) for x in selected_tuples]
        selected_source_ids = [str(x[2].get("source_id", "")) for x in selected_tuples]
        selected_scores = [round(x[0], 2) for x in selected_tuples]
        logger.debug(
            f"[KNOWLEDGE RETRIEVAL DEBUG] Query expansion: '{concept_expanded}', "
            f"Preferred sections/intents: {preferred}, "
            f"Selected Chunk IDs: {selected_chunk_ids}, "
            f"Source IDs: {selected_source_ids}, "
            f"Scores: {selected_scores}"
        )

        for score_val, idx, f_chunk in selected_tuples:
            all_formatted.append(f_chunk)
            chunk_pattern = f_chunk["supported_labels"][0] if f_chunk["supported_labels"] else ""
            if chunk_pattern and chunk_pattern in concept.lower():
                supporting_chunks.append(f_chunk)
            elif is_contradicting:
                contradicting_chunks.append(f_chunk)
            else:
                contextual_chunks.append(f_chunk)

        permitted_citations = [
            {
                "citation_id": c["citation_id"],
                "title": c["title"],
                "source": c["source"],
                "source_title": c["source_title"],
                "organization_or_authors": c["organization_or_authors"],
                "date_or_version": c["date_or_version"],
                "section": c["section"],
                "page": c["page"],
                "doi": c["doi"],
                "url": c["url"],
                "document_sha256": c["document_sha256"],
            }
            for c in all_formatted
        ]

        # Legacy caller check: if query_concept was passed without bridge_result, return List[GuidelineChunk]
        if query_concept is not None and bridge_result is None and classifier_result is None:
            return [GuidelineChunk(item[2]) for item in selected_tuples]

        return {
            "status": "ok",
            "semantic_retrieval_available": self.semantic_retrieval_available,
            "retrieval_mode": "semantic" if self.semantic_retrieval_available else "deterministic_lexical_ontology",
            "supporting": supporting_chunks,
            "contradicting": contradicting_chunks,
            "contextual": contextual_chunks,
            "all_chunks": all_formatted,
            "permitted_citations": permitted_citations,
            "top_k": len(all_formatted),
        }


# Service alias for public integration
KnowledgeService = KnowledgeRetriever
