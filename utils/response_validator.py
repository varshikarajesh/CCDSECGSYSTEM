# -*- coding: utf-8 -*-
"""
utils/response_validator.py

Answer validation logic to enforce reasoning policy compliance.
"""

import json
import re
from typing import List, Dict, Tuple, Any
from utils.reasoning_policy import DiagnosisAnchor, AnswerMode
from typing import Optional

def parse_confidence_percent(text: str) -> Optional[str]:
    text_lower = text.lower()
    
    # 1. Match percentages like "55%", "71.3%", "71%"
    match_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", text_lower)
    if match_pct:
        return match_pct.group(1)
        
    # 2. Match word representation of numbers, e.g. "fifty-five percent" or "fifty five"
    word_to_num = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
        "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
        "eighty": "80", "ninety": "90"
    }
    
    words = re.findall(r"\b[a-z]+\b", text_lower)
    for i in range(len(words)):
        w = words[i]
        if w in ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]:
            base = int(word_to_num[w])
            if i + 1 < len(words) and words[i+1] in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]:
                val = base + int(word_to_num[words[i+1]])
                return str(val)
            return str(base)
        if w in word_to_num:
            return word_to_num[w]
            
    # 3. Match decimal representations like "0.55"
    match_dec = re.search(r"\b0\.(\d{1,4})\b", text_lower)
    if match_dec:
        dec_val = float(match_dec.group(0))
        return f"{dec_val * 100:.1f}".rstrip('0').rstrip('.')
        
    return None

class ValidationContext:
    def __init__(
        self,
        intent: str,
        sub_intent: str,
        composer_class_name: str,
        composer_input_class_name: str,
        selected_fields: List[str],
        current_prompt: str,
        current_answer: str,
        anchor: DiagnosisAnchor,
        mode: AnswerMode,
        pipeline_result: Any = None,
        composer_used: Any = None,
        fallback_used: bool = False,
        composer_input: Any = None,
        kb_summary: Any = None
    ):
        self.intent = intent
        self.sub_intent = sub_intent
        self.composer_class_name = composer_class_name
        self.composer_input_class_name = composer_input_class_name
        self.selected_fields = selected_fields
        self.current_prompt = current_prompt
        self.current_answer = current_answer
        self.anchor = anchor
        self.mode = mode
        self.pipeline_result = pipeline_result
        self.composer_used = composer_used
        self.fallback_used = fallback_used
        self.composer_input = composer_input
        self.kb_summary = kb_summary


EXPECTED_COMPOSER_BY_ROUTE = {
    ("diagnosis", None): "DiagnosisComposer",
    ("diagnosis_reasoning", None): "DiagnosisReasoningComposer",
    ("symptoms", None): "SymptomsComposer",
    ("patient_symptoms", None): "PatientSymptomsComposer",
    ("treatment", None): "TreatmentComposer",
    ("treatment", "PACEMAKER_DECISION"): "TreatmentComposer",
    ("educational_definitions", None): "EducationalComposer",
    ("retrieval_explanation", None): "NeighbourExplanationComposer",
    ("confidence_reasoning", None): "ConfidenceComposer",
}

COMPOSER_ALIASES = {
    "ConfidenceExplanationComposer": "ConfidenceComposer",
    "NeighbourExplanationComposer": "NeighbourExplanationComposer",
    "NeighbourComposer": "NeighbourExplanationComposer",
}

def normalize_composer_name(name: str) -> str:
    if not name:
        return ""
    name_str = str(name).strip()
    return COMPOSER_ALIASES.get(name_str, name_str)


class ResponseValidator:
    @staticmethod
    def normalize_generated_answer(answer: str) -> str:
        """Extract a natural answer from harmless model wrappers before safety validation."""
        if not answer:
            return ""
        text = str(answer).strip()
        fenced = re.fullmatch(r"```json\s*(\{.*\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            try:
                obj = json.loads(fenced.group(1))
                for key in ("answer", "final_answer", "text"):
                    if isinstance(obj.get(key), str) and obj[key].strip():
                        return obj[key].strip()
            except Exception:
                pass
        text = re.sub(r"^```(?:natural_answer|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"</?(?:response|natural[-_]?answer|natural[-_]?answer_requirements)\s*>", "", text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def select_validators(context: ValidationContext) -> Tuple[List[str], List[str]]:
        all_possible = [
            "no_raw_json", "no_raw_pipeline_objects", "no_canned_fallback",
            "diagnosis_preservation", "no_neighbour_contamination", 
            "no_patient_treatment_without_support", "single_source_footer", 
            "disease_category_consistency", "no_prohibited_scaffolding", 
            "no_empty_bullets", "composer_matches_intent", 
            "condition_aware_general_knowledge", "confidence_grounding", 
            "neighbour_grounding", "pacemaker_decision", "treatment_validation", 
            "regeneration_text_removed", "diagnosis_reasoning_depth", 
            "correct_composer_used", "composer_input_complete", 
            "fallback_mismatches_intent", "source_attribution", 
            "educational_contains_no_pipeline", "educational_factual",
            "educational_relevance", "retrieval_target", "confidence_query_value",
            "treatment_urgency_grounding", "patient_symptom_separation"
        ]
        selected = []
        
        # Educational/GENERAL_ONLY isolation rules
        if context.mode == AnswerMode.GENERAL_ONLY or context.intent == "educational_definitions":
            selected = [
                "no_raw_json", "no_raw_pipeline_objects", "no_prohibited_scaffolding", 
                "no_empty_bullets", "educational_contains_no_pipeline", "educational_factual", 
                "source_attribution", "educational_relevance"
            ]
        else:
            selected = [
                "no_raw_json", "no_raw_pipeline_objects", "no_canned_fallback", 
                "diagnosis_preservation", "no_neighbour_contamination", 
                "no_patient_treatment_without_support", "single_source_footer", 
                "disease_category_consistency", "no_prohibited_scaffolding", 
                "no_empty_bullets", "composer_matches_intent", 
                "condition_aware_general_knowledge", "regeneration_text_removed", 
                "correct_composer_used", "composer_input_complete", 
                "fallback_mismatches_intent", "source_attribution"
            ]
            
            if context.intent in ["diagnosis", "final_label"]:
                pass
            elif context.intent == "diagnosis_reasoning":
                selected.append("diagnosis_reasoning_depth")
            elif context.intent == "retrieval_explanation":
                selected.append("neighbour_grounding")
                selected.append("retrieval_target")
            elif context.intent == "confidence_reasoning":
                selected.append("confidence_grounding")
                selected.append("confidence_query_value")
            elif context.intent == "treatment":
                selected.append("treatment_urgency_grounding")
                is_pacemaker = (
                    context.sub_intent == "PACEMAKER_DECISION" or 
                    "pacemaker" in (context.current_prompt or "").lower() or 
                    "pacemaker" in (context.current_answer or "").lower()
                )
                if is_pacemaker:
                    selected.append("pacemaker_decision")
                else:
                    selected.append("treatment_validation")
            elif context.intent == "patient_symptoms":
                selected.append("patient_symptom_separation")
            elif context.intent == "symptoms":
                pass
                
            # Confidence check is only run under specific scope rules:
            # - diagnosis / final_label
            # - confidence_reasoning
            # - diagnosis_reasoning, but only if confidence fields are serialized in current_prompt
            should_run_confidence = False
            if context.intent in ["diagnosis", "final_label", "confidence_reasoning"]:
                should_run_confidence = True
            elif context.intent == "diagnosis_reasoning":
                prompt_lower = (context.current_prompt or "").lower()
                if "confidence" in prompt_lower or "calibrated" in prompt_lower or "bucket" in prompt_lower:
                    should_run_confidence = True
                    
            if should_run_confidence:
                selected.append("confidence_grounding")
                
        skipped = [v for v in all_possible if v not in selected]
        return selected, skipped

    @staticmethod
    def validate_no_raw_json(context: ValidationContext, violations: List[str]):
        if "{" in context.current_answer or "}" in context.current_answer or '":' in context.current_answer:
            violations.append("no_raw_json")

    @staticmethod
    def validate_no_raw_pipeline_objects(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        for word in ["pipeline_context", "evidence_outputs", "confidence_outputs", "bridge_outputs", "rare_case_outputs"]:
            if word in ans_lower:
                violations.append("no_raw_pipeline_objects")
                break

    @staticmethod
    def validate_no_canned_fallback(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        canned_phrases = [
            "there are no available symptom records",
            "there are no available actionable guidance records",
            "the current knowledge base context does not support that conclusion"
        ]
        for phrase in canned_phrases:
            if phrase in ans_lower:
                violations.append("no_canned_fallback_only_answer")
                break

    @staticmethod
    def validate_diagnosis_preservation(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        for disease in context.anchor.prohibited_competing_diagnoses:
            disease_pat = r"\b" + re.escape(disease) + r"\b"
            if re.search(disease_pat, ans_lower):
                patient_claims = [
                    f"patient has {disease}", f"patient's ecg shows {disease}", 
                    f"diagnosed with {disease}", f"indicates {disease}", f"consistent with {disease}",
                    f"patient demonstrates {disease}", f"patient's condition is {disease}"
                ]
                if any(claim in ans_lower for claim in patient_claims):
                    violations.append("diagnosis_anchor_preserved")
                if "guideline" in ans_lower and disease in ans_lower and not ("general" in ans_lower or "educational" in ans_lower or disease in context.current_prompt.lower()):
                    violations.append("irrelevant_kb_absent")

    @staticmethod
    def validate_no_neighbour_contamination(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        if "neighbour" in ans_lower or "neighbor" in ans_lower:
            finding_words = ["st elevation", "st depression", "q-wave", "left bundle", "right bundle", "t-wave inversion"]
            for f in finding_words:
                if f in ans_lower:
                    patient_claims = [f"patient has {f}", f"patient's ecg shows {f}", f"patient ecg shows {f}", f"patient demonstrates {f}"]
                    if any(claim in ans_lower for claim in patient_claims) and not any(claim in context.anchor.bridge_concept.lower() or claim in context.anchor.concept.lower() for claim in [f]):
                        violations.append("no_neighbour_contamination")
                        break

    @staticmethod
    def validate_no_patient_treatment_without_support(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        if context.anchor.family.upper() == "NORM":
            treatment_words = ["stent", "pci", "cabg", "dapt", "statin", "prescribe", "procedure", "intervention"]
            for word in treatment_words:
                if word in ans_lower:
                    patient_claims = [
                        f"patient should receive {word}", f"prescribe {word} for the patient",
                        f"patient requires {word}", f"indicate {word} for this patient"
                    ]
                    if any(claim in ans_lower for claim in patient_claims):
                        violations.append("no_patient_specific_treatment_without_support")
                        break

    @staticmethod
    def validate_single_source_footer(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        source_matches = re.findall(r"\b(?:source|sources)\s*:", ans_lower)
        if len(source_matches) > 1:
            violations.append("single_normalized_source_footer")

    @staticmethod
    def validate_disease_category_consistency(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        is_mi_or_ischemia = any(x in context.anchor.family.lower() or x in context.anchor.concept.lower() for x in ["mi", "myocardial infarction", "ischemia"])
        if is_mi_or_ischemia:
            if "conduction disturbance" in ans_lower or "conduction block" in ans_lower or "heart block" in ans_lower:
                violations.append("disease_category_consistency")

    @staticmethod
    def validate_no_prohibited_scaffolding(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        prohibited_scaffolds = [
            "generated output:", "output description:", "source units:", 
            "this query relates to", "medical concept", "event label match", 
            "supporting chunks", "chunk ids", "section unknown", "developer trace"
        ]
        if any(pat in ans_lower for pat in prohibited_scaffolds):
            violations.append("no_prohibited_scaffolding")

    @staticmethod
    def validate_no_empty_bullets(context: ValidationContext, violations: List[str]):
        if re.search(r"^\s*[\-\*•]\s*$", context.current_answer, re.MULTILINE) or re.search(r"^\s*[\-\*•]\s*-\s*$", context.current_answer, re.MULTILINE):
            violations.append("no_empty_bullets")

    @staticmethod
    def validate_composer_matches_intent(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        if context.intent == "confidence_reasoning":
            if not any(w in ans_lower for w in ["confidence", "calibrated", "score", "agreement", "boost", "penalty"]):
                violations.append("composer_matches_intent")
        elif context.intent == "retrieval_explanation":
            if not any(w in ans_lower for w in ["neighbour", "neighbor", "matched", "embedding", "retrieved", "similarity"]):
                violations.append("composer_matches_intent")
        elif context.intent == "diagnosis_reasoning":
            if not any(w in ans_lower for w in ["bridge", "evidence", "morpholog", "conduction", "consensus"]):
                violations.append("composer_matches_intent")
        elif context.intent in ["symptoms", "patient_symptoms"]:
            if not any(w in ans_lower for w in ["symptom", "pain", "fatigue", "syncope", "discomfort", "lightheadedness", "breath"]):
                violations.append("composer_matches_intent")
        elif context.intent == "treatment":
            is_pacemaker = context.sub_intent == "PACEMAKER_DECISION" or "pacemaker" in (context.current_prompt or "").lower() or "pacemaker" in ans_lower
            if is_pacemaker:
                if not any(w in ans_lower for w in ["pacemaker", "pacing", "bradycardia", "block", "conduction", "treatment"]):
                    violations.append("composer_matches_intent")
            else:
                if not any(w in ans_lower for w in ["guideline", "management", "treatment", "monitoring", "limit", "missing", "cardiologist"]):
                    violations.append("composer_matches_intent")

    @staticmethod
    def validate_condition_aware_general_knowledge(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        diagnosis_lower = context.anchor.concept.lower()
        if "lpfb" in diagnosis_lower or "left posterior" in diagnosis_lower:
            for term in ["bundle branch block", "ivcd", "af", "atrial fibrillation", "pacemaker"]:
                if term in ans_lower and term not in (context.current_prompt or "").lower() and term not in diagnosis_lower:
                    violations.append("condition_aware_general_knowledge")

    @staticmethod
    def validate_confidence_grounding(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        prompt_lower = (context.current_prompt or "").lower()
        
        # authoritative value consistency
        boosts_val = None
        penalties_val = None
        bridge_support_val = None
        evidence_support_val = None
        neighbour_agreement_val = None
        morphology_findings = []
        authoritative_score = None
        
        if context.pipeline_result:
            if hasattr(context.pipeline_result, "confidence_outputs") and context.pipeline_result.confidence_outputs:
                co = context.pipeline_result.confidence_outputs
                boosts_val = co.get("boosts") or co.get("positive_contributors")
                penalties_val = co.get("penalties") or co.get("negative_contributors")
                bridge_support_val = co.get("bridge_support") or co.get("knowledge_support")
                evidence_support_val = co.get("evidence_support") or co.get("morphology_support")
                neighbour_agreement_val = co.get("neighbor_consensus") or co.get("neighbour_agreement")
                authoritative_score = co.get("score") or co.get("overall_confidence")
            if hasattr(context.pipeline_result, "evidence_outputs") and context.pipeline_result.evidence_outputs:
                eo = context.pipeline_result.evidence_outputs
                morphology_findings = eo.get("morphology_findings") or []
        
        if context.composer_input:
            if boosts_val is None: boosts_val = getattr(context.composer_input, "boosts", None)
            if penalties_val is None: penalties_val = getattr(context.composer_input, "penalties", None)
            if bridge_support_val is None: bridge_support_val = getattr(context.composer_input, "bridge_support", None)
            if evidence_support_val is None: evidence_support_val = getattr(context.composer_input, "evidence_support", None)
            if neighbour_agreement_val is None: neighbour_agreement_val = getattr(context.composer_input, "neighbour_agreement", None)
            if authoritative_score is None: authoritative_score = getattr(context.composer_input, "confidence_score", None)

        def is_val_available(v) -> bool:
            if v is None:
                return False
            v_str = str(v).strip().lower()
            if v_str in ["", "none", "not available", "[]"]:
                return False
            return True

        available_contributors = {
            "neighbour_agreement": is_val_available(neighbour_agreement_val),
            "bridge_support": is_val_available(bridge_support_val),
            "evidence_support": is_val_available(evidence_support_val),
            "boosts": is_val_available(boosts_val),
            "penalties": is_val_available(penalties_val),
        }

        if authoritative_score is not None:
            try:
                score_float = float(authoritative_score)
                pct_val = score_float * 100 if score_float <= 1.0 else score_float
                pct_str1 = f"{pct_val:.1f}"
                pct_str2 = f"{round(pct_val)}"
                pct_str3 = f"{pct_val:.2f}"
                
                # If serialized, verify score consistency
                if "confidence" in prompt_lower or context.intent == "confidence_reasoning":
                    if not (pct_str1 in context.current_answer or pct_str2 in context.current_answer or pct_str3 in context.current_answer or f"{score_float}" in context.current_answer):
                        violations.append("confidence_value_mismatch")
            except (ValueError, TypeError):
                pass

        # Check for unsupported mention of contributors
        unsupported_keywords = {
            "bridge": "bridge_support",
            "evidence": "evidence_support",
            "boost": "boosts",
            "penal": "penalties"
        }
        for kw, field in unsupported_keywords.items():
            if kw in ans_lower and not available_contributors[field]:
                if not any(ph in ans_lower for ph in ["not available", "unavailable", "no separate", "no explicit"]):
                    violations.append("confidence_response_contains_unsupported_fields")

        # Check for completeness only where required (Section 2E, Refinement 3)
        has_conf_in_prompt = any(w in prompt_lower for w in ["confidence", "calibrated", "bucket", "score"])
        requires_contributor_explanation = (
            context.intent == "confidence_reasoning" or
            (context.intent == "diagnosis_reasoning" and has_conf_in_prompt)
        )

        if requires_contributor_explanation:
            # check explicit statements when Bridge/Evidence support is unavailable
            if not available_contributors["bridge_support"]:
                if not any(ph in ans_lower for ph in ["bridge", "not available", "unavailable", "no separate"]):
                    violations.append("insufficient_confidence_contributors")
            if not available_contributors["evidence_support"]:
                if not any(ph in ans_lower for ph in ["evidence", "not available", "unavailable", "no separate"]):
                    violations.append("insufficient_confidence_contributors")

            # Check that available contributors are explained
            explained = 0
            avail_total = 0
            for field, avail in available_contributors.items():
                if avail:
                    avail_total += 1
                    kw = "neighbor" if field == "neighbour_agreement" else field.split("_")[0]
                    if kw == "neighbor":
                        if any(w in ans_lower for w in ["neighbor", "neighbour", "consensus", "agreement"]):
                            explained += 1
                    else:
                        if kw in ans_lower:
                            explained += 1

            if explained < avail_total and avail_total > 0:
                violations.append("insufficient_confidence_contributors")

    @staticmethod
    def validate_neighbour_grounding(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        if "similarity" not in ans_lower and "similarity score" not in ans_lower:
            violations.append("neighbour_fields_present")
        if not (any(pat in ans_lower for pat in ["ecg_", "id:"]) and ("similarity:" in ans_lower or "score:" in ans_lower or "similarity score" in ans_lower)):
            violations.append("neighbour_response_contains_real_fields")

    @staticmethod
    def validate_pacemaker_decision(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        direct_ans_patterns = [
            "does not by itself establish that a pacemaker is required",
            "cannot determine whether pacing is required from this diagnosis alone",
            "the current trace record is insufficient to determine pacing need",
            "insufficient to determine pacemaker need",
            "not by itself establish",
            "not by itself determine"
        ]
        has_direct_ans = any(pat in ans_lower for pat in direct_ans_patterns)
        
        pacing_vars = ["bradycardia", "av block", "conduction failure", "instability", "pauses", "escape rhythm", "transient", "persistent", "symptomatic", "pacing"]
        pacing_var_matches = [w for w in pacing_vars if w in ans_lower]
        
        if "complete heart block" in ans_lower or "advanced av block" in ans_lower:
            pacing_var_matches.append("av block")
        if "blood-pressure instability" in ans_lower or "haemodynamic" in ans_lower:
            pacing_var_matches.append("instability")
            
        pacing_var_matches = list(set(pacing_var_matches))
        
        if not has_direct_ans:
            violations.append("pacemaker_indecisive")
            
        if len(pacing_var_matches) < 2:
            violations.append("pacemaker_missing_variables")

    @staticmethod
    def validate_treatment_validation(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        is_limitation_answer = any(pat in ans_lower for pat in [
            "does not contain enough patient-specific information",
            "insufficient patient-specific information",
            "lack of patient-specific",
            "treatment cannot be recommended",
            "cannot recommend an individualized"
        ])
        
        if is_limitation_answer:
            has_urgent = any(w in ans_lower for w in ["urgent", "cardiologist", "clinical assessment", "evaluation"])
            if not has_urgent:
                violations.append("treatment_question_answered")
        else:
            if not any(w in ans_lower for w in ["cardiologist", "evaluation", "principles", "monitoring", "limit", "missing"]):
                violations.append("treatment_question_answered")
                
            has_kb = any(w in ans_lower for w in ["guideline", "section", "kb"]) or (context.kb_summary and any(w in ans_lower for w in context.kb_summary.lower().split()[:20]))
            has_general_limit = "management" in ans_lower and any(w in ans_lower for w in ["missing", "limitation", "cardiologist", "limit"])
            if not (has_kb or has_general_limit):
                violations.append("treatment_is_supported")

    @staticmethod
    def validate_regeneration_text_removed(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        regeneration_patterns = ["i apologize", "i corrected", "my previous response", "the previous answer", "i revised"]
        if any(pat in ans_lower for pat in regeneration_patterns):
            violations.append("regeneration_text_removed")

    @staticmethod
    def validate_diagnosis_reasoning_depth(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        if not any(w in ans_lower for w in ["bridge", "evidence", "morpholog", "conduction", "consensus"]):
            violations.append("diagnosis_reasoning_depth")

    @staticmethod
    def validate_correct_composer_used(context: ValidationContext, violations: List[str]):
        expected_composer = EXPECTED_COMPOSER_BY_ROUTE.get((context.intent, context.sub_intent)) or EXPECTED_COMPOSER_BY_ROUTE.get((context.intent, None))
        
        if expected_composer and context.composer_class_name:
            actual_norm = normalize_composer_name(context.composer_class_name)
            expected_norm = normalize_composer_name(expected_composer)
            if actual_norm != expected_norm:
                violations.append("composer_class_mismatch")

    @staticmethod
    def validate_composer_input_complete(context: ValidationContext, violations: List[str]):
        if context.composer_input:
            missing_reqs = context.composer_input.validate()
            if missing_reqs:
                violations.append("composer_input_complete")

    @staticmethod
    def validate_fallback_mismatches_intent(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        if context.fallback_used:
            if context.intent == "confidence_reasoning" and "confidence" not in ans_lower and "calibrated" not in ans_lower:
                violations.append("fallback_mismatches_intent")
            if context.intent == "retrieval_explanation" and "neighbour" not in ans_lower and "neighbor" not in ans_lower and "similarity" not in ans_lower:
                violations.append("fallback_mismatches_intent")
            if context.intent == "treatment" and "management" not in ans_lower and "treatment" not in ans_lower and "clinical assessment" not in ans_lower and "pacemaker" not in ans_lower:
                violations.append("fallback_mismatches_intent")

    @staticmethod
    def validate_source_attribution(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        if "sources:" in ans_lower:
            footer_part = ans_lower.split("sources:")[-1]
            actual_sources = []
            for line in footer_part.split("\n"):
                if line.strip().startswith("•"):
                    actual_sources.append(line.replace("•", "").strip().lower())
            
            if context.intent == "diagnosis_reasoning":
                if "trace pipeline evidence" not in actual_sources:
                    violations.append("source_attribution_incorrect")
            elif context.mode == AnswerMode.GENERAL_ONLY or context.intent == "educational_definitions":
                if not all(s == "general medical knowledge" for s in actual_sources):
                    violations.append("source_attribution_incorrect")
            elif context.intent in ["diagnosis", "final_label"]:
                if "trace pipeline evidence" not in actual_sources:
                    violations.append("source_attribution_incorrect")
            elif context.intent == "patient_symptoms":
                if not any(s in actual_sources for s in ["trace pipeline evidence", "insufficient patient symptom history"]):
                    violations.append("source_attribution_incorrect")
            elif context.intent == "treatment" and context.fallback_used:
                if any("esc" in s for s in actual_sources):
                    violations.append("source_attribution_incorrect")

    @staticmethod
    def validate_educational_contains_no_pipeline(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        pipeline_terms = ["trace", "pipeline", "calibrated confidence", "neighbor", "neighbour", "ecg_id"]
        if any(term in ans_lower for term in pipeline_terms):
            violations.append("educational_contains_no_pipeline")

    @staticmethod
    def validate_educational_factual(context: ValidationContext, violations: List[str]):
        ans_lower = context.current_answer.lower()
        q_lower = context.current_prompt.lower() if context.current_prompt else ""
        if "af" in q_lower or "atrial fibrillation" in q_lower:
            forbidden_af_terms = ["outflow obstruction", "functional shunting", "lvh", "left ventricular hypertrophy", "diastolic dysfunction"]
            if any(term in ans_lower for term in forbidden_af_terms):
                violations.append("educational_factual_inaccuracy")

    @staticmethod
    def validate_educational_relevance(context: ValidationContext, violations: List[str]):
        if context.intent == "educational_definitions":
            q_lower = (context.current_prompt or "").lower()
            ans_lower = (context.current_answer or "").lower()
            if "af" in q_lower or "atrial fibrillation" in q_lower:
                if "atrial fibrillation" not in ans_lower and "af" not in ans_lower:
                    violations.append("educational_relevance_fail")

    @staticmethod
    def validate_retrieval_target(context: ValidationContext, violations: List[str]):
        if context.intent == "retrieval_explanation":
            q_lower = (context.current_prompt or "").lower()
            ans_lower = (context.current_answer or "").lower()
            match = re.search(r"neighbour\s*#?\s*(\d+)", q_lower)
            if not match:
                match = re.search(r"neighbor\s*#?\s*(\d+)", q_lower)
            if match:
                idx = match.group(1)
                expected_phrase = f"neighbour #{idx}"
                expected_phrase_alt = f"neighbor #{idx}"
                if expected_phrase not in ans_lower and expected_phrase_alt not in ans_lower:
                    violations.append("retrieval_target_mismatch")

    @staticmethod
    def validate_confidence_query_value(context: ValidationContext, violations: List[str]):
        if context.intent == "confidence_reasoning":
            q_lower = (context.current_prompt or "").lower()
            ans_lower = (context.current_answer or "").lower()
            user_score = parse_confidence_percent(q_lower)
            if user_score:
                user_score_pct = user_score if user_score.endswith("%") else f"{user_score}%"
                if user_score_pct not in ans_lower:
                    violations.append("confidence_query_value_mismatch")

    @staticmethod
    def validate_treatment_urgency_grounding(context: ValidationContext, violations: List[str]):
        if context.intent == "treatment":
            ans_lower = (context.current_answer or "").lower()
            concept_lower = (context.anchor.concept or "").lower()
            if "normal" in concept_lower or "sinus rhythm" in concept_lower:
                if "urgent" in ans_lower or "emergency" in ans_lower or "immediate" in ans_lower:
                    violations.append("treatment_urgency_ungrounded")

    @staticmethod
    def validate_patient_symptom_separation(context: ValidationContext, violations: List[str]):
        if context.intent == "patient_symptoms":
            ans_lower = (context.current_answer or "").lower()
            if "no patient-reported symptom history" not in ans_lower:
                violations.append("patient_symptom_grounding_fail")

    @staticmethod
    def validate(
        answer: str, 
        question: str, 
        anchor: DiagnosisAnchor, 
        mode: AnswerMode, 
        pipeline_result=None,
        route=None,
        composer_used=None,
        fallback_used=False,
        composer_input=None,
        kb_summary=None
    ) -> Tuple[str, List[str]]:
        intent = route.intent if route else "diagnosis"
        sub_intent = getattr(route, "sub_intent", None)
        
        composer_class_name = None
        if composer_used:
            if isinstance(composer_used, str):
                composer_class_name = composer_used
            else:
                composer_class_name = composer_used.__class__.__name__
                if hasattr(composer_used, "__name__") and (not hasattr(composer_used, "__class__") or composer_used.__class__.__name__ == 'type'):
                    composer_class_name = composer_used.__name__
                    
        composer_input_class_name = None
        if composer_input:
            if isinstance(composer_input, str):
                composer_input_class_name = composer_input
            else:
                composer_input_class_name = composer_input.__class__.__name__
                if hasattr(composer_input, "__name__") and (not hasattr(composer_input, "__class__") or composer_input.__class__.__name__ == 'type'):
                    composer_input_class_name = composer_input.__name__

        # Build validation context
        context = ValidationContext(
            intent=intent,
            sub_intent=sub_intent,
            composer_class_name=composer_class_name,
            composer_input_class_name=composer_input_class_name,
            selected_fields=[],
            current_prompt=question,
            current_answer=answer,
            anchor=anchor,
            mode=mode,
            pipeline_result=pipeline_result,
            composer_used=composer_used,
            fallback_used=fallback_used,
            composer_input=composer_input,
            kb_summary=kb_summary
        )
        
        # Select validators
        selected, skipped = ResponseValidator.select_validators(context)
        
        violations = []
        # Execute each selected validator
        for val_name in selected:
            val_fn = getattr(ResponseValidator, f"validate_{val_name}", None)
            if val_fn:
                val_fn(context, violations)
                
        # Internal Exception exposed validation (user-visible only - Part 5B, Refinement 5)
        ERROR_PATTERNS = [
            r"\bTraceback\b",
            r"\b(?:NameError|UnboundLocalError|TypeError|AttributeError|KeyError)\b",
            r"cannot access local variable",
            r'File ".+?", line \d+',
            r"^\s*ERROR\s*:",
            r"^\s*Exception\s*:",
        ]
        internal_exception_exposed = False
        for pat in ERROR_PATTERNS:
            if re.search(pat, answer):
                internal_exception_exposed = True
                violations.append("internal_exception_exposed")
                break

        # Check permitted citation IDs (KB-...)
        if permitted_citations := getattr(context, "permitted_citations", None) or []:
            permitted_ids = {c.get("citation_id") if isinstance(c, dict) else getattr(c, "citation_id", None) for c in permitted_citations}
            permitted_ids = {pid for pid in permitted_ids if pid}
            extracted_ids = set(re.findall(r"\bKB-[A-Za-z0-9_\-]+\b", context.current_answer))
            unsupported = extracted_ids - permitted_ids
            if unsupported:
                violations.append("unpermitted_citation_id")

        violations = list(set(violations))
        
        # Structured Composer Match Evidence print out
        expected_composer = EXPECTED_COMPOSER_BY_ROUTE.get((intent, sub_intent)) or EXPECTED_COMPOSER_BY_ROUTE.get((intent, None))
        composer_passed = (normalize_composer_name(composer_class_name) == normalize_composer_name(expected_composer))
        composer_evidence = {
            "passed": composer_passed,
            "expected_composer": expected_composer,
            "actual_composer": composer_class_name,
            "intent": intent,
            "sub_intent": sub_intent
        }
        if not composer_passed:
            composer_evidence["reason"] = "composer_class_mismatch"
            
        print(f"\n[VALIDATOR LINEAGE LOG]")
        print(f"- Intent: {intent}")
        print(f"- Sub-Intent: {sub_intent or 'None'}")
        print(f"- Composer Match Evidence: {composer_evidence}")
        print(f"- Selected Validators: {', '.join(selected)}")
        print(f"- Skipped Validators: {', '.join(skipped)}")
        print(f"- Validation Result: {'FAIL' if violations else 'PASS'}")
        print(f"- Failure Reasons: {violations or 'None'}\n")
        
        # Strict validation result invariants (Section 6)
        if violations:
            return "REGENERATE", violations
        return "PASS", []

    @staticmethod
    def validate_citations(
        answer: str,
        permitted_citations: Optional[List[Dict[str, Any]]] = None,
        corrective_retry_performed: bool = False
    ) -> Any:
        from runtime.runtime_contracts import CitationValidationResult
        permitted_ids = set()
        for c in (permitted_citations or []):
            cid = c.get("citation_id") if isinstance(c, dict) else getattr(c, "citation_id", None)
            if cid:
                permitted_ids.add(cid)

        extracted_ids = set(re.findall(r"\bKB-[A-Za-z0-9_\-]+\b", answer))
        unsupported = sorted(list(extracted_ids - permitted_ids))
        has_sources = "Sources:" in answer or "Sources :" in answer

        is_valid = len(unsupported) == 0

        return CitationValidationResult(
            valid=is_valid,
            extracted_citations=sorted(list(extracted_ids)),
            permitted_citations=sorted(list(permitted_ids)),
            unsupported_citations=unsupported,
            sources_section_added=has_sources,
            corrective_retry_performed=corrective_retry_performed
        )

    @staticmethod
    def strip_unsupported_citations(
        answer: str,
        permitted_citations: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Strips unpermitted citation markers (e.g. [KB-XXX]) and empty Sources blocks
        from generated text without discarding the underlying clinical response.
        """
        if not answer:
            return answer
        permitted_ids = set()
        for c in (permitted_citations or []):
            cid = c.get("citation_id") if isinstance(c, dict) else getattr(c, "citation_id", None)
            if cid:
                permitted_ids.add(str(cid))

        extracted_ids = set(re.findall(r"\bKB-[A-Za-z0-9_\-]+\b", answer))
        unsupported_ids = extracted_ids - permitted_ids

        cleaned = answer
        for uid in unsupported_ids:
            cleaned = re.sub(r"\[" + re.escape(uid) + r"\]", "", cleaned)
            cleaned = re.sub(r"\b" + re.escape(uid) + r"\b", "", cleaned)

        cleaned = re.sub(r"\[\s*;\s*\]", "", cleaned)
        cleaned = re.sub(r"\[\s*\]", "", cleaned)
        cleaned = re.sub(r"  +", " ", cleaned)

        if "sources:" in cleaned.lower():
            parts = re.split(r"(?i)\bsources\s*:", cleaned)
            main_part = parts[0].strip()
            footer_part = parts[1] if len(parts) > 1 else ""
            footer_citations = re.findall(r"\bKB-[A-Za-z0-9_\-]+\b", footer_part)
            valid_footer_cits = [fc for fc in footer_citations if fc in permitted_ids]
            if not valid_footer_cits:
                cleaned = main_part

        return cleaned.strip()

    @staticmethod
    def validate_post_generation(
        answer: str,
        question: str,
        anchor: Any,
        mode: Any,
        bridge_result: Dict[str, Any],
        permitted_citations: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        from runtime.runtime_contracts import PostGenerationValidation
        violations = []

        cit_res = ResponseValidator.validate_citations(answer, permitted_citations)
        if not cit_res.valid:
            violations.append("unpermitted_citation_id")

        concept = (getattr(anchor, "concept", None) or "UNKNOWN").lower()
        decision = bridge_result.get("decision", "supported")
        ood_info = bridge_result.get("ood", {})
        is_ood = ood_info.get("is_ood", False)

        ans_lower = answer.lower()
        q_lower = (question or "").lower().strip()

        # A supported deterministic decision must never be weakened into an
        # invented UNKNOWN/indeterminate result by the explanation model.
        supported_decision = str(decision).lower() in {
            "supported", "multi_label_supported", "probable", "partially_supported"
        }
        false_unknown_claim = any(phrase in ans_lower for phrase in (
            "primary finding is currently unknown",
            "primary finding is unknown",
            "decision is indeterminate",
            "diagnosis is unknown",
            "primary label and family are both unknown",
            "no diagnostic confidence or support strength is available",
        ))
        if supported_decision and concept != "unknown" and false_unknown_claim:
            violations.append("diagnosis_mismatch")

        # Resolve the same deterministic intent used by inference so the final
        # answer must actually satisfy the user's question.
        try:
            from utils.question_router import route_question
            answer_route = route_question(question, conversation_state={"history": [{"role": "assistant", "content": "case active"}]})
        except Exception:
            answer_route = None

        patient_procedure_question = bool(
            answer_route
            and answer_route.intent == "treatment"
            and getattr(answer_route, "patient_specific", False)
        )
        if patient_procedure_question:
            ecg_limit = any(term in ans_lower for term in (
                "ecg alone cannot determine", "ecg by itself cannot determine",
                "ecg trace alone cannot determine", "trace alone cannot determine",
                "cannot determine from the ecg", "cannot be determined from this ecg",
                "does not establish whether", "insufficient to determine",
                "result alone cannot determine", "this result alone cannot determine",
            ))
            missing_inputs = any(term in ans_lower for term in (
                "coronary anatomy", "angiography", "troponin", "biomarker",
                "clinical assessment", "clinical evaluation", "clinical correlation",
                "symptom history",
            ))
            if not ecg_limit or not missing_inputs:
                violations.append("patient_procedure_question_not_answered_safely")

        patient_symptom_question = (
            any(term in q_lower for term in ("symptom", "pain", "dyspnea", "shortness of breath", "palpitation"))
            and any(term in q_lower for term in ("this patient", "the patient", "patient have", "my symptoms"))
        )
        if answer_route and answer_route.intent == "patient_symptoms":
            patient_symptom_question = True
        if patient_symptom_question:
            grounded_absence_language = any(term in ans_lower for term in (
                "not provided", "no patient symptoms", "no symptoms were supplied", "history is unavailable",
                "cannot infer", "cannot be inferred", "symptom history"
            ))
            if not grounded_absence_language:
                violations.append("patient_symptoms_not_grounded")

        # Check if question is pure educational or general definition
        is_educational = any(q_lower.startswith(prefix) for prefix in ["what is", "explain", "define", "what are"]) and not any(w in q_lower for w in ["this patient", "patient", "this ecg", "my", "here", "case"])

        if (decision in ("Unknown", "insufficient_evidence") or is_ood) and not is_educational:
            valid_uncertainty_phrases = [
                "unknown", "out-of-distribution", "ood", "atypical", "manual clinical review",
                "stronger conclusion", "disagreement", "safety check", "unverified", "candidate",
                "cannot determine", "insufficient", "additional clinical assessment", "evidence does not establish",
                "result alone cannot", "not established", "exploratory", "limitation", "requires repeat",
                "educational", "definition", "general", "cannot be determined", "does not establish",
                "missing", "record", "requires clinical", "guideline", "uncertain"
            ]
            if not any(phrase in ans_lower for phrase in valid_uncertainty_phrases):
                violations.append("unknown_ood_not_preserved")

            # Context-aware certainty check (allow percentages when quoting candidate probabilities)
            definite_claims = [
                r"\bdefinitely\s+(has|requires|needs)\b",
                r"\bcertainly\s+(has|requires|needs)\b",
                r"\bconfirmed\s+(diagnosis|mi|stemi|lvh)\b",
                r"\bmust\s+receive\s+a?\s*(stent|pci|cabg|bypass)\b",
                r"\bdefinitely\s+(needs|does\s+not\s+need)\b",
                r"\bpatient\s+definitely\b",
                r"\bconfirmed\s+that\s+the\s+patient\b",
            ]
            if any(re.search(pat, ans_lower) for pat in definite_claims):
                violations.append("unknown_strengthened_with_certainty")

        # Inventions check for patient specific claims
        invented_labs_patterns = [
            r"\btroponin\s+is\s+(elevated|high|normal|\d+)\b",
            r"\bangiography\s+(shows|revealed|demonstrates)\b",
            r"\bechocardiogram\s+shows\b",
        ]
        if any(re.search(pat, ans_lower) for pat in invented_labs_patterns):
            violations.append("invented_clinical_data")

        # A diagnostic family is supporting metadata, not an additional label.
        family = str(bridge_result.get("primary_family") or "").strip().lower()
        if family and family not in {"unknown", "norm", concept}:
            family_as_diagnosis = (
                f"diagnosis of {family}" in ans_lower
                or f"secondary diagnosis is {family}" in ans_lower
                or f"secondary diagnosis of {family}" in ans_lower
            )
            if family_as_diagnosis:
                violations.append("family_presented_as_diagnosis")

        prohibited = [
            r"---", r"SYSTEM INSTRUCTION", r"\[1\. IMMUTABLE", r"ACTIVE SKILL INSTRUCTIONS",
            r"GARBAGE_COLS", r"pipeline_context_v2_1", r"SASH_(MARKERS|BACK_CORS)",
            r"SOUGHT_IDSS", r"diagnostic trace advisory", r"embedding_id",
            r"retrieval_kneighbors", r"TOP_ECG_SHFTS", r"Request Shape:",
            r"Trace ID:", r"In trace chains", r"Sub-messages", r"<NMI_",
            r"</?natural[-_]?answer", r"</?natural_answer_requirements", r"</?response>",
        ]
        for p in prohibited:
            if re.search(p, answer):
                violations.append("prompt_leakage")

        critical = (
            "internal_exception_exposed" in violations
            or "prompt_leakage" in violations
            or "unknown_strengthened_with_certainty" in violations
            or "diagnosis_mismatch" in violations
        )

        return PostGenerationValidation(
            valid=len(violations) == 0,
            diagnosis_preserved="diagnosis_mismatch" not in violations,
            uncertainty_preserved="unknown_ood_not_preserved" not in violations,
            ood_preserved="unknown_ood_not_preserved" not in violations,
            confidence_language_valid="confidence_mismatch" not in violations,
            neighbor_attribution_valid="neighbor_contamination" not in violations,
            signal_limitations_valid=True,
            contradictions_disclosed=True,
            treatment_grounded="unsupported_treatment" not in violations,
            citations_valid=cit_res.valid,
            critical_failure=critical,
            violations=violations
        )



    @staticmethod
    def validate_relevance(answer: str, question: str, anchor: Any, mode: Any, route: Any) -> Tuple[bool, List[str]]:
        ans_lower = answer.lower()
        failures = []
        intent = route.intent if route else "diagnosis"
        sub_intent = getattr(route, "sub_intent", None)
        
        if intent == "treatment" and sub_intent == "PACEMAKER_DECISION":
            intent = "pacemaker_decision"
            
        concept = (getattr(anchor, "concept", "") or "inferoposterior myocardial infarction").lower()
        
        if intent in ["diagnosis", "final_label"]:
            diag_keywords = [concept, "myocardial infarction", "lpfb", "block", "normal", "norm"]
            if not any(kw in ans_lower for kw in diag_keywords):
                failures.append(f"Diagnosis answer does not contain the pipeline diagnosis concept/keywords matching '{concept}'")
                
        elif intent == "diagnosis_reasoning":
            diag_keywords = [concept, "myocardial infarction", "lpfb", "block", "normal", "norm", "diagnosis"]
            if not any(kw in ans_lower for kw in diag_keywords):
                failures.append("Diagnosis reasoning does not contain the diagnosis concept")
            support_components = ["retrieval", "bridge", "evidence", "morpholog", "alignment", "confidence", "consensus", "similarity"]
            if not any(sc in ans_lower for sc in support_components):
                failures.append("Diagnosis reasoning does not contain any support components")
                
        elif intent == "symptoms":
            symptom_words = ["chest pressure", "pain", "shortness of breath", "dyspnoea", "sweating", "nausea", "dizziness", "fatigue", "syncope", "lightheadedness", "palpitations"]
            symptom_matches = [w for w in symptom_words if w in ans_lower]
            bullets = ans_lower.count("-") + ans_lower.count("•") + ans_lower.count("*")
            if len(symptom_matches) < 2 and bullets < 2:
                failures.append("General symptoms answer does not contain at least two symptom items")
                
        elif intent == "patient_symptoms":
            if not any(pat in ans_lower for pat in ["symptom history", "symptom record", "not available", "no patient-reported", "recorded symptoms"]):
                failures.append("Patient symptoms answer does not state whether patient symptom history is available")
                
        elif intent == "treatment":
            is_limitation = "patient-specific" in ans_lower and "insufficient" in ans_lower
            if not is_limitation:
                if not any(w in ans_lower for w in ["reperfusion", "antiplatelet", "pci", "monitor", "clinical assessment", "urgent", "cardiologist"]):
                    failures.append("Treatment answer does not contain management direction")
                if not any(w in ans_lower for w in ["limit", "missing", "urgency", "urgent", "cardiologist", "in-person", "assess"]):
                    failures.append("Treatment answer does not contain a patient-specific limitation or urgency statement")
                
        elif intent == "pacemaker_decision":
            direct_ans_patterns = [
                "does not by itself establish that a pacemaker is required",
                "cannot determine whether pacing is required from this diagnosis alone",
                "the current trace record is insufficient to determine pacing need",
                "insufficient to determine pacemaker need",
                "not by itself establish",
                "not by itself determine"
            ]
            if not any(pat in ans_lower for pat in direct_ans_patterns):
                failures.append("Pacemaker answer does not address whether pacing can be determined from the diagnosis alone")
            
        elif intent == "educational_definitions":
            if "patient" in ans_lower or "trace diagnosis" in ans_lower:
                failures.append("Educational definition contains patient-specific TRACE details")
                
        elif intent == "retrieval_explanation":
            if not (any(pat in ans_lower for pat in ["ecg_", "neighbour #", "neighbor #", "rank"]) or re.search(r"neighbour \d+", ans_lower)):
                failures.append("Retrieval explanation does not contain neighbour ID or rank")
            if not any(w in ans_lower for w in ["similarity", "matched", "reason", "query", "faiss"]):
                failures.append("Retrieval explanation does not contain similarity or retrieval reason")
                
        elif intent == "confidence_reasoning":
            if not any(w in ans_lower for w in ["%", "score", "confidence value"]):
                failures.append("Confidence reasoning does not contain confidence score")
            if not any(w in ans_lower for w in ["high", "moderate", "low", "bucket", "category"]):
                failures.append("Confidence reasoning does not contain confidence bucket")
                
        return len(failures) == 0, failures
