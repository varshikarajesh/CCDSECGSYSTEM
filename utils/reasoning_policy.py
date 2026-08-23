# -*- coding: utf-8 -*-
"""
utils/reasoning_policy.py

Reasoning policy definitions and query classification rules.
"""

import re
from enum import Enum
from typing import Any
from utils.question_router import QuestionRoute

class AnswerMode(str, Enum):
    PIPELINE_ONLY = "PIPELINE_ONLY"
    PIPELINE_EXPLANATION = "PIPELINE_EXPLANATION"
    PIPELINE_PLUS_KB = "PIPELINE_PLUS_KB"
    PIPELINE_PLUS_GENERAL = "PIPELINE_PLUS_GENERAL"
    PIPELINE_KB_GENERAL = "PIPELINE_KB_GENERAL"
    GENERAL_ONLY = "GENERAL_ONLY"
    HYBRID = "hybrid"


class DiagnosisAnchor:
    def __init__(self, concept: str, family: str, confidence: float, bridge_concept: str, evidence_findings: list):
        self.concept = concept
        self.family = family
        self.confidence = confidence
        self.bridge_concept = bridge_concept
        self.evidence_findings = evidence_findings or []
        self.prohibited_competing_diagnoses = self._determine_prohibited()

    def _determine_prohibited(self) -> list:
        all_diseases = ["stemi", "nstemi", "myocardial infarction", "atrial fibrillation", "afib", "af", "nsvt", "brugada", "wpw"]
        concept_lower = self.concept.lower()
        family_lower = self.family.lower()
        
        prohibited = []
        for d in all_diseases:
            if d not in concept_lower and d not in family_lower:
                prohibited.append(d)
        return prohibited

class ReasoningPolicy:
    @staticmethod
    def classify_question(question: str, anchor: DiagnosisAnchor) -> AnswerMode:
        q = question.lower().strip()
        
        is_about_patient = any(w in q for w in ["patient", "this", "my", "current", "here", "shown", "me", "case", "record"])
        
        # Check general definition matching
        general_patterns = [
            r"^(what\s+(is|are)|explain|define)\s+(a\s+|an\s+|the\s+)?(pci|cabg|statin|statins|dapt|stent|stents|aspirin|beta\s*blocker|beta\s*blockers|revascularization|ablation|treatment|therapy|management|procedure|intervention|symptom|symptoms|prognosis|recovery|stemi|mi|myocardial\s*infarction|heart\s*attack|af|afib|atrial\s*fibrillation|lpfb|lafb|lvh)\b"
        ]
        is_general_medical = any(re.search(pat, q) for pat in general_patterns)
        is_pure_educational = is_general_medical and not is_about_patient
        
        if is_pure_educational:
            return AnswerMode.GENERAL_ONLY
            
        pipeline_keywords = ["label", "diagnosis", "predict", "confidence", "uncertain", "neighbour", "neighbor", "similar ecg", "faiss", "bridge", "evidence", "trace", "dump", "developer"]
        if any(k in q for k in pipeline_keywords):
            if "why" in q or "explain" in q:
                return AnswerMode.PIPELINE_EXPLANATION
            return AnswerMode.PIPELINE_ONLY
            
        kb_keywords = ["stent", "pci", "cabg", "revascularization", "medication", "drug", "prescribe", "aspirin", "beta blocker", "statin", "dapt", "guideline", "management"]
        if any(k in q for k in kb_keywords):
            return AnswerMode.PIPELINE_PLUS_KB
            
        return AnswerMode.PIPELINE_PLUS_GENERAL

    @staticmethod
    def determine_sources(route: QuestionRoute, anchor: DiagnosisAnchor) -> str:
        return getattr(route, "required_sources", getattr(route, "required_source", "COMBINED"))

    @staticmethod
    def is_general_knowledge_permitted(route: QuestionRoute, anchor: DiagnosisAnchor, kb_available: bool = False) -> bool:
        if getattr(route, "intent", None) in ["educational_definitions", "symptoms", "treatment", "patient_education"]:
            return True
        return not kb_available
