# -*- coding: utf-8 -*-
"""
utils/evidence_hierarchy.py

Defines the four-level evidence hierarchy and source attribution for clinical reasoning.
"""

from enum import Enum

class AnswerSource(str, Enum):
    PIPELINE = "PIPELINE"
    KNOWLEDGE_BASE = "KNOWLEDGE_BASE"
    GENERAL_MEDICAL = "GENERAL_MEDICAL"
    COMBINED = "COMBINED"

class ReasoningMode(str, Enum):
    PIPELINE_ONLY = "PIPELINE_ONLY"
    PIPELINE_PLUS_KB = "PIPELINE_PLUS_KB"
    KB_ONLY = "KB_ONLY"
    GENERAL_MEDICAL = "GENERAL_MEDICAL"
    HYBRID = "HYBRID"

class EvidenceHierarchy:
    @staticmethod
    def get_sources_used(mode: ReasoningMode, kb_used: bool, general_knowledge_used: bool) -> list:
        """
        Determines the list of clinical evidence sources used.
        """
        sources = []
        # Pipeline is always Level 1, authoritative and included in almost all answers
        if mode != ReasoningMode.KB_ONLY:
            sources.append("Pipeline Evidence")
            
        # Knowledge Base is Level 2
        if kb_used:
            sources.append("Knowledge Base")
            
        # General Medical Knowledge is Level 3
        if general_knowledge_used:
            sources.append("General Medical Knowledge")
            
        return sources

    @staticmethod
    def resolve_answer_source(mode: ReasoningMode, kb_used: bool, general_knowledge_used: bool) -> AnswerSource:
        """
        Resolves the overall AnswerSource category.
        """
        sources = EvidenceHierarchy.get_sources_used(mode, kb_used, general_knowledge_used)
        if len(sources) > 1:
            return AnswerSource.COMBINED
        elif len(sources) == 1:
            if sources[0] == "Pipeline Evidence":
                return AnswerSource.PIPELINE
            elif sources[0] == "Knowledge Base":
                return AnswerSource.KNOWLEDGE_BASE
            elif sources[0] == "General Medical Knowledge":
                return AnswerSource.GENERAL_MEDICAL
        return AnswerSource.PIPELINE  # Default fallback

    @staticmethod
    def format_attribution_footer(sources: list) -> str:
        """
        Formats the reasoning source attribution footer.
        """
        if not sources:
            return ""
        lines = ["\n\nReasoning Sources"]
        for src in sources:
            lines.append(f"✓ {src}")
        return "\n".join(lines)
