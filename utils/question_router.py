# -*- coding: utf-8 -*-
"""
utils/question_router.py

Deterministic intent classifier and context router for the TRACE system.
"""

import re
from typing import Dict, Any, Optional, List
from utils.evidence_hierarchy import ReasoningMode

PROCEDURE_TERMS = {
    "stent",
    "stents",
    "pci",
    "angioplasty",
    "cabg",
    "bypass",
    "revascularization",
    "catheterization",
    "coronary intervention",
}


class QuestionRoute:
    def __init__(self, intent: str, required_context_blocks: list, kb_required: bool = False,
                 trace_level: str = "compact", context_budget_tokens: int = 1000,
                 allow_expansion: bool = False, developer_mode: bool = False, preferred_sections: Optional[list] = None,
                 required_source: Any = ReasoningMode.PIPELINE_PLUS_KB,
                 required_sources: str = "COMBINED", sub_intent: Optional[str] = None,
                 patient_specific: bool = False, question_subjects: Optional[List[str]] = None,
                 comparison_requested: bool = False, required_inputs: Optional[List[str]] = None,
                 original_question: str = "", normalized_question: str = "",
                 corrections: Optional[List[dict]] = None, correction_confidence: float = 1.0):
        self.intent = intent
        self.required_context_blocks = required_context_blocks
        self.kb_required = kb_required
        self.trace_level = trace_level
        self.context_budget_tokens = context_budget_tokens
        self.allow_expansion = allow_expansion
        self.developer_mode = developer_mode
        self.preferred_sections = preferred_sections or []
        self.required_source = required_source
        self.required_sources = required_sources
        self.sub_intent = sub_intent
        self.patient_specific = patient_specific
        self.question_subjects = question_subjects or []
        self.comparison_requested = comparison_requested
        self.required_inputs = required_inputs or []
        self.original_question = original_question
        self.normalized_question = normalized_question
        self.corrections = corrections or []
        self.correction_confidence = correction_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "sub_intent": self.sub_intent,
            "patient_specific": self.patient_specific,
            "required_sources": self.required_sources,
            "required_context_blocks": self.required_context_blocks,
            "kb_required": self.kb_required,
            "trace_level": self.trace_level,
            "context_budget_tokens": self.context_budget_tokens,
            "allow_expansion": self.allow_expansion,
            "developer_mode": self.developer_mode,
            "preferred_sections": self.preferred_sections,
            "question_subjects": self.question_subjects,
            "comparison_requested": self.comparison_requested,
            "required_inputs": self.required_inputs,
            "original_question": self.original_question,
            "normalized_question": self.normalized_question,
            "corrections": self.corrections,
            "correction_confidence": self.correction_confidence,
        }


def normalize_spelling_typos(question: str) -> dict[str, Any]:
    original = question.strip()
    normalized = original

    stop_words = {
        "the", "a", "an", "they", "them", "their", "this", "that", "these", "those",
        "your", "my", "our", "his", "her", "its", "here", "there", "what", "which",
        "show", "give", "tell", "what", "who", "whom", "whose", "where", "when",
        "why", "how", "are", "is", "was", "were", "been", "being", "have", "has",
        "had", "having", "do", "does", "did", "doing", "would", "should", "could",
        "ought", "must", "may", "might", "can", "will", "shall", "and", "but",
        "or", "if", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down", "in",
        "out", "on", "off", "over", "under", "again", "further", "then", "once"
    }
    
    typo_map = {
        "symtoms": "symptoms",
        "symptopys": "symptoms",
        "symptom": "symptoms",
        "symptomy": "symptoms",
        "retival": "retrieval",
        "retreival": "retrieval",
        "retreived": "retrieved",
        "segement": "segment",
        "segemnt": "segment",
        "classifer": "classifier",
        "diagonsis": "diagnosis",
        "diognosis": "diagnosis",
        "abnormalitoes": "abnormalities",
        "evidance": "evidence",
        "citaiton": "citation",
        "rythm": "rhythm",
        "confidance": "confidence",
        "irbb": "IRBBB",
        "irbbb": "IRBBB",
        "i-rbbb": "IRBBB",
        "incomplete rbbb": "incomplete right bundle branch block",
        "uncertian": "uncertain",
        "secound": "second",
        "wich": "which",
        "mater": "matter",
    }
    
    corrections = []
    
    # Tokenize the question to perform replacement on word boundaries
    words = re.findall(r"\b[a-zA-Z0-9\-\(\)]+\b", original)
    
    for word in words:
        w_lower = word.lower()
        if w_lower in typo_map:
            replacement = typo_map[w_lower]
            # Match the casing
            if word.isupper() and len(replacement) <= 5:
                replacement = replacement.upper()
            normalized = re.sub(rf"\b{re.escape(word)}\b", replacement, normalized)
            corrections.append({"original": word, "replacement": replacement})
            
    # Calculate simple confidence
    confidence = 1.0 if not corrections else round(1.0 - (0.04 * len(corrections)), 2)
    
    # 2. Conservative fuzzy matching on known concepts
    known_concepts = [
        "symptoms", "retrieval", "retrieved", "segment", "classifier", 
        "diagnosis", "abnormalities", "evidence", "citation", "rhythm", 
        "confidence", "uncertain", "first", "second", "third", "which",
        "what", "why", "who", "when", "where", "how", "evidence summary"
    ]
    
    ont_mapping = {
        "NORM": "normal ECG", "NDT": "non-diagnostic T abnormalities",
        "NST_": "non-specific ST changes", "DIG": "digitalis-effect",
        "LNGQT": "long QT-interval", "LVH": "left ventricular hypertrophy",
        "LAFB": "left anterior fascicular block", "LPFB": "left posterior fascicular block",
        "IRBBB": "incomplete right bundle branch block", "1AVB": "first-degree atrioventricular block",
        "2AVB": "second-degree atrioventricular block", "3AVB": "third-degree atrioventricular block",
        "CRBBB": "complete right bundle branch block", "CLBBB": "complete left bundle branch block",
        "ISCAL": "ischemic changes in anterolateral leads", "SR": "sinus rhythm",
        "AFIB": "atrial fibrillation", "AFLT": "atrial flutter", "STACH": "sinus tachycardia",
        "SBRAD": "sinus bradycardia", "SARRH": "sinus arrhythmia", "PVC": "premature ventricular complex",
        "PAC": "premature atrial complex", "IMI": "inferior myocardial infarction",
        "AMI": "anterior myocardial infarction", "ASMI": "anteroseptal myocardial infarction",
        "ALMI": "anterolateral myocardial infarction"
    }
    
    def levenshtein_distance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1+1], distances_[-1])))
            distances = distances_
        return distances[-1]
        
    normalized_words = re.findall(r"\b[a-zA-Z0-9\-\(\)]+\b", normalized)
    for word in normalized_words:
        w_lower = word.lower()
        if word.isdigit() or len(word) < 4 or any(c.isdigit() for c in word) or w_lower in stop_words:
            continue
        if w_lower in known_concepts or w_lower.upper() in ont_mapping or any(w_lower == k.lower() for k in ont_mapping.values()):
            continue
            
        best_match = None
        min_dist = 999
        
        for concept in known_concepts:
            dist = levenshtein_distance(w_lower, concept)
            if dist <= 2 and dist < min_dist:
                min_dist = dist
                best_match = concept
                
        for key in ont_mapping.keys():
            dist = levenshtein_distance(w_lower, key.lower())
            if dist <= 1 and dist < min_dist:
                min_dist = dist
                best_match = key
                
        if best_match and min_dist <= 2:
            plausible_matches = []
            for concept in known_concepts:
                if levenshtein_distance(w_lower, concept) == min_dist:
                    plausible_matches.append(concept)
            for key in ont_mapping.keys():
                if levenshtein_distance(w_lower, key.lower()) == min_dist:
                    plausible_matches.append(key)
                    
            if len(set(plausible_matches)) > 1:
                normalized = f"CLARIFICATION_REQUIRED: {', '.join(set(plausible_matches))}"
                return {
                    "original_question": original,
                    "normalized_question": normalized,
                    "corrections": corrections,
                    "correction_confidence": 0.0,
                    "clarification_options": list(set(plausible_matches))
                }
                
            normalized = re.sub(rf"\b{re.escape(word)}\b", best_match, normalized)
            corrections.append({"original": word, "replacement": best_match})
            
    confidence = 1.0 if not corrections else round(1.0 - (0.04 * len(corrections)), 2)
    return {
        "original_question": original,
        "normalized_question": normalized,
        "corrections": corrections,
        "correction_confidence": confidence
    }


def _route_question_internal(user_question: str, conversation_state=None) -> QuestionRoute:
    q = user_question.lower().strip()
    is_about_patient = any(w in q for w in ["patient", "this", "my", "current", "here", "shown", "case", "record", "need a stent", "does this"])

    # In a case-scoped chatbot, terse symptom follow-ups refer to the current
    # patient unless the user explicitly asks for general associations.
    has_case_context = conversation_state is not None
    if has_case_context and q in {
        "what are the symptoms", "what are symptoms", "symptoms", "any symptoms",
        "what symptoms", "what symptoms are there",
    }:
        is_about_patient = True

    # Resolve terse comparison follow-ups against case-scoped conversation state.
    history_text = ""
    if conversation_state:
        history = conversation_state.get("history", []) if isinstance(conversation_state, dict) else getattr(conversation_state, "history", [])
        history_text = " ".join(str(item.get("content", "")) if isinstance(item, dict) else str(item) for item in history[-4:]).lower()
    comparison_context = any(x in history_text for x in ("ecg 1", "ecg1", "ecg 2", "ecg2", "compare"))
    explicit_comparison = (
        (any(x in q for x in ("ecg 1", "ecg1", "first ecg")) and any(x in q for x in ("ecg 2", "ecg2", "second ecg")))
        or any(x in q for x in ("compare the ecg", "compare ecg", "difference between", "different between", "why are") ) and "ecg" in q
    )
    comparison_followup = comparison_context and any(x in q for x in ("what about", "how about", "change", "different", "compare", "qt", "qrs", "rate", "rhythm", "lead"))
    if explicit_comparison or comparison_followup:
        subjects = ["ECG_1", "ECG_2"]
        subjects.extend([name for name, terms in {
            "QTc": ("qt", "qtc"), "QRS": ("qrs",), "heart_rate": ("rate", "heart rate"),
            "rhythm": ("rhythm",), "morphology": ("lead", "waveform", "morphology")
        }.items() if any(term in q for term in terms)])
        return QuestionRoute(
            intent="ecg_comparison", sub_intent="COMPARE_SELECTED_FEATURES" if len(subjects) > 2 else "COMPARE_COMPLETE_ECG",
            patient_specific=True, comparison_requested=True, question_subjects=subjects,
            required_inputs=["ecg_1", "ecg_2"],
            required_context_blocks=["paired_signal_quality", "paired_statistics", "paired_classifier", "paired_retrieval", "paired_bridge"],
            kb_required=True, trace_level="compact", context_budget_tokens=1600,
            preferred_sections=["ecg_diagnostic_criteria", "differential_diagnosis"],
            required_source=ReasoningMode.PIPELINE_PLUS_KB, required_sources="COMBINED",
        )

    # Measurement-specific questions need measured ECG values plus validated criteria.
    statistic_terms = ("qtc", "qt interval", "qrs duration", "pr interval", "rmssd", "sdnn", "pnn50", "pnn20", "heart rate variability", "hrv")
    if any(term in q for term in statistic_terms):
        return QuestionRoute(
            intent="statistics", sub_intent="ECG_STATISTIC_INTERPRETATION", patient_specific=is_about_patient,
            question_subjects=[term for term in statistic_terms if term in q], required_inputs=["ecg_statistics"],
            required_context_blocks=["ecg_statistics", "signal_quality", "diagnosis"], kb_required=True,
            preferred_sections=["ecg_diagnostic_criteria", "risk_red_flags"], context_budget_tokens=1200,
            required_source=ReasoningMode.PIPELINE_PLUS_KB, required_sources="COMBINED",
        )

    if any(term in q for term in ("which window", "what time", "when did", "five minute", "5 minute", "two minute", "2 minute", "abnormal segment")):
        return QuestionRoute(
            intent="long_recording", sub_intent="TEMPORAL_LOCALIZATION", patient_specific=True,
            required_inputs=["temporal_summary"], required_context_blocks=["temporal_summary", "abnormal_windows", "diagnosis"],
            kb_required=False, context_budget_tokens=1200, required_source=ReasoningMode.PIPELINE_ONLY, required_sources="PIPELINE",
        )
    
    # Check if query is about patient or general
    # Procedure terms check
    has_procedure_term = any(term in q for term in PROCEDURE_TERMS)
    
    # Definition / General medical pattern check
    general_patterns = [
        r"^what\s+(is|are)\s+(a\s+|an\s+|the\s+)?(pci|cabg|statin|statins|dapt|stent|stents|aspirin|beta\s*blocker|beta\s*blockers|revascularization|ablation|treatment|therapy|management|procedure|intervention|symptom|symptoms|prognosis|recovery|stemi|mi|myocardial\s*infarction|heart\s*attack|af|afib|atrial\s*fibrillation|lpfb|lafb|lvh)\b",
        r"^explain\s+(a\s+|an\s+|the\s+)?(pci|cabg|statin|statins|dapt|stent|stents|aspirin|beta\s*blocker|beta\s*blockers|revascularization|ablation|treatment|therapy|management|procedure|intervention|symptom|symptoms|prognosis|recovery|stemi|mi|myocardial\s*infarction|heart\s*attack|af|afib|atrial\s*fibrillation|lpfb|lafb|lvh)\b",
        r"^define\s+(a\s+|an\s+|the\s+)?(pci|cabg|statin|statins|dapt|stent|stents|aspirin|beta\s*blocker|beta\s*blockers|revascularization|ablation|treatment|therapy|management|procedure|intervention|symptom|symptoms|prognosis|recovery|stemi|mi|myocardial\s*infarction|heart\s*attack|af|afib|atrial\s*fibrillation|lpfb|lafb|lvh)\b"
    ]
    is_general_medical = any(re.search(pat, q) for pat in general_patterns)
    is_pure_educational = is_general_medical and not is_about_patient

    # Patient-specific procedure question check (e.g., "Does this patient need a stent?")
    if has_procedure_term and is_about_patient:
        return QuestionRoute(
            intent="treatment",
            sub_intent="PATIENT_SPECIFIC_PROCEDURE",
            patient_specific=True,
            required_context_blocks=["diagnosis", "kb_summary", "safety_note"],
            kb_required=True,
            trace_level="compact",
            context_budget_tokens=1200,
            preferred_sections=["revascularization", "stent", "antiplatelet", "pci"],
            required_source=ReasoningMode.PIPELINE_PLUS_KB,
            required_sources="COMBINED"
        )
    
    # 0. Pure Educational Definitions (e.g. "What is a stent?")
    if is_pure_educational:
        return QuestionRoute(
            intent="educational_definitions",
            sub_intent="EDUCATIONAL_DEFINITION",
            patient_specific=False,
            required_context_blocks=[],
            kb_required=False,
            trace_level="compact",
            context_budget_tokens=400,
            required_source=ReasoningMode.GENERAL_MEDICAL,
            required_sources="GENERAL_MEDICAL"
        )

    # Pacemaker sub-intent check
    pacing_words = ["pacemaker", "pacing", "indicate", "permanent", "temporary"]
    if any(w in q for w in pacing_words):
        return QuestionRoute(
            intent="treatment",
            sub_intent="PACEMAKER_DECISION",
            patient_specific=is_about_patient,
            required_context_blocks=["diagnosis", "kb_summary", "safety_note"],
            kb_required=True,
            trace_level="compact",
            context_budget_tokens=1200,
            preferred_sections=["pacing", "pacemaker", "av block", "bradycardia"],
            required_source=ReasoningMode.GENERAL_MEDICAL if is_pure_educational else ReasoningMode.PIPELINE_PLUS_KB,
            required_sources="GENERAL_MEDICAL" if is_pure_educational else "COMBINED"
        )

    # Developer Mode
    if "developer" in q or "dev mode" in q or "dump" in q:
        return QuestionRoute(
            intent="developer",
            patient_specific=is_about_patient,
            required_context_blocks=["diagnosis", "confidence", "evidence", "kb_summary", "retrieved_neighbors", "trace_summary"],
            kb_required=True,
            trace_level="developer",
            context_budget_tokens=2500,
            allow_expansion=True,
            developer_mode=True,
            required_source=ReasoningMode.PIPELINE_PLUS_KB,
            required_sources="COMBINED"
        )
        
    # Final Label
    final_label_keywords = ["what is final label", "final diagnosis", "what did pipeline predict", "final concept", "final label"]
    if any(k in q for k in final_label_keywords):
        return QuestionRoute(
            intent="final_label",
            patient_specific=True,
            required_context_blocks=["diagnosis"],
            kb_required=False,
            trace_level="compact",
            context_budget_tokens=400,
            required_source=ReasoningMode.PIPELINE_ONLY,
            required_sources="PIPELINE"
        )
        
    # Trace Commands
    if q == "trace" or "show full trace" in q or "pipeline trace" in q:
        return QuestionRoute(
            intent="full_trace",
            patient_specific=True,
            required_context_blocks=["trace_summary"],
            kb_required=False,
            trace_level="detailed",
            context_budget_tokens=1500,
            required_source=ReasoningMode.PIPELINE_ONLY,
            required_sources="PIPELINE"
        )
        
    # Source Check
    if q == "source" or "show source" in q or "guideline source" in q or "citation" in q:
        return QuestionRoute(
            intent="show_source",
            patient_specific=is_about_patient,
            required_context_blocks=["kb_summary"],
            kb_required=True,
            trace_level="compact",
            context_budget_tokens=500,
            required_source=ReasoningMode.KB_ONLY,
            required_sources="KB"
        )
        
    # Expansion Request
    if q == "expand" or "explain in detail" in q or "more detail" in q:
        return QuestionRoute(
            intent="expand",
            patient_specific=is_about_patient,
            required_context_blocks=["diagnosis", "kb_summary", "safety_note"],
            kb_required=True,
            trace_level="compact",
            context_budget_tokens=1800,
            allow_expansion=True,
            required_source=ReasoningMode.PIPELINE_PLUS_KB,
            required_sources="COMBINED"
        )
        
    # Procedure / Revascularization check (when not specifically matching patient flag earlier)
    if has_procedure_term:
        sub = "PATIENT_SPECIFIC_PROCEDURE" if is_about_patient else "GENERAL_PROCEDURE_EDUCATION"
        source_mode = ReasoningMode.GENERAL_MEDICAL if not is_about_patient else ReasoningMode.PIPELINE_PLUS_KB
        return QuestionRoute(
            intent="treatment",
            sub_intent=sub,
            patient_specific=is_about_patient,
            required_context_blocks=["diagnosis", "kb_summary", "safety_note"],
            kb_required=True,
            trace_level="compact",
            context_budget_tokens=1200,
            preferred_sections=["revascularization", "stent", "antiplatelet", "pci"],
            required_source=source_mode,
            required_sources="GENERAL_MEDICAL" if not is_about_patient else "COMBINED"
        )
        
    if "medication" in q or "drug" in q or "prescribe" in q or "aspirin" in q or "beta blocker" in q or "statin" in q or "dose" in q or "contraindication" in q or "regimen" in q:
        source_mode = ReasoningMode.GENERAL_MEDICAL if is_pure_educational else ReasoningMode.PIPELINE_PLUS_KB
        return QuestionRoute(
            intent="medication",
            patient_specific=is_about_patient,
            required_context_blocks=["diagnosis", "kb_summary", "safety_note"],
            kb_required=True,
            trace_level="compact",
            context_budget_tokens=1200,
            preferred_sections=["pharmacotherapy", "medication", "drug"],
            required_source=source_mode,
            required_sources="GENERAL_MEDICAL" if is_pure_educational else "COMBINED"
        )
        
    if "treat" in q or "therapy" in q or "management" in q or "ablation" in q or "procedure" in q or "intervention" in q:
        source_mode = ReasoningMode.GENERAL_MEDICAL if is_pure_educational else ReasoningMode.PIPELINE_PLUS_KB
        return QuestionRoute(
            intent="treatment",
            sub_intent="PATIENT_SPECIFIC_PROCEDURE" if is_about_patient else "GENERAL_PROCEDURE_EDUCATION",
            patient_specific=is_about_patient,
            required_context_blocks=["diagnosis", "kb_summary", "safety_note"],
            kb_required=True,
            trace_level="compact",
            context_budget_tokens=1200,
            preferred_sections=["treatment", "therapy", "management"],
            required_source=source_mode,
            required_sources="GENERAL_MEDICAL" if is_pure_educational else "COMBINED"
        )
        
    # Recovery & Rehabilitation
    if "recovery" in q or "rehabilitation" in q or "lifestyle" in q or "rest" in q or "exercise" in q or "activity" in q or "follow up" in q:
        source_mode = ReasoningMode.GENERAL_MEDICAL if is_pure_educational else ReasoningMode.HYBRID
        return QuestionRoute(
            intent="recovery",
            patient_specific=is_about_patient,
            required_context_blocks=["diagnosis", "kb_summary", "safety_note"],
            kb_required=True,
            trace_level="compact",
            context_budget_tokens=1000,
            preferred_sections=["recovery", "lifestyle", "rehabilitation"],
            required_source=source_mode,
            required_sources="GENERAL_MEDICAL" if is_pure_educational else "COMBINED"
        )
        
    # Prognosis & Outcomes
    if "prognosis" in q or "outlook" in q or "outcome" in q or "survival" in q or "future" in q or "recurrence" in q or "long term" in q or "chronic" in q:
        source_mode = ReasoningMode.GENERAL_MEDICAL if is_pure_educational else ReasoningMode.HYBRID
        return QuestionRoute(
            intent="prognosis",
            patient_specific=is_about_patient,
            required_context_blocks=["diagnosis", "kb_summary", "safety_note"],
            kb_required=True,
            trace_level="compact",
            context_budget_tokens=1000,
            preferred_sections=["prognosis", "outcomes"],
            required_source=source_mode,
            required_sources="GENERAL_MEDICAL" if is_pure_educational else "COMBINED"
        )
        
    # Patient Symptoms Check
    is_symptom_query = any(w in q for w in ["symptom", "pain", "dyspnea", "shortness of breath", "angina", "syncope", "palpitations", "presentation", "dizziness", "fainting"])
    is_patient_specific_symptom = is_symptom_query and (is_about_patient or any(w in q for w in ["this patient", "the patient", "patient have", "reported", "this case", "experiencing", "symptomatic"]))
    
    if is_patient_specific_symptom:
        return QuestionRoute(
            intent="patient_symptoms",
            sub_intent="PATIENT_SPECIFIC_SYMPTOMS",
            patient_specific=True,
            required_context_blocks=["diagnosis", "evidence"],
            kb_required=False,
            trace_level="compact",
            context_budget_tokens=1000,
            required_source=ReasoningMode.PIPELINE_ONLY,
            required_sources="PIPELINE"
        )

    # General Symptoms Education
    if is_symptom_query:
        return QuestionRoute(
            intent="symptoms",
            sub_intent="GENERAL_SYMPTOM_EDUCATION",
            patient_specific=False,
            required_context_blocks=["diagnosis", "kb_summary"],
            kb_required=True,
            trace_level="compact",
            context_budget_tokens=1000,
            preferred_sections=["symptoms", "presentation"],
            required_source=ReasoningMode.GENERAL_MEDICAL,
            required_sources="GENERAL_MEDICAL"
        )
        
    # Investigations
    if "investigation" in q or "test" in q or "echo" in q or "troponin" in q or "mri" in q or "holter" in q or "blood" in q or "angiography" in q or "imaging" in q:
        source_mode = ReasoningMode.GENERAL_MEDICAL if not is_about_patient else ReasoningMode.HYBRID
        return QuestionRoute(
            intent="investigations",
            patient_specific=is_about_patient,
            required_context_blocks=["diagnosis", "kb_summary"],
            kb_required=True,
            trace_level="compact",
            context_budget_tokens=1000,
            preferred_sections=["investigations", "diagnosis"],
            required_source=source_mode,
            required_sources="GENERAL_MEDICAL" if not is_about_patient else "COMBINED"
        )
 
    # ECG Waveform Interpretation
    if "interpret" in q or "waveform" in q or "lead" in q or "qrs" in q or "st segment" in q or "t wave" in q or "p wave" in q or "interval" in q or "rhythm" in q or "bridge" in q:
        return QuestionRoute(
            intent="ecg_interpretation",
            patient_specific=True,
            required_context_blocks=["diagnosis", "evidence"],
            kb_required=False,
            trace_level="compact",
            context_budget_tokens=1000,
            required_source=ReasoningMode.PIPELINE_ONLY,
            required_sources="PIPELINE"
        )
        
    # Retrieval & FAISS Neighbors Explanation
    if "neighbour" in q or "neighbor" in q or "retrieved" in q or "retrieval" in q or "similar ecg" in q or "faiss" in q or "excluded" in q:
        return QuestionRoute(
            intent="retrieval_explanation",
            sub_intent="RETRIEVAL_EXPLANATION",
            patient_specific=True,
            required_context_blocks=["retrieved_neighbors", "trace_summary"],
            kb_required=False,
            trace_level="compact",
            context_budget_tokens=1200,
            required_source=ReasoningMode.PIPELINE_ONLY,
            required_sources="PIPELINE"
        )
        
    # Confidence Scorer calibration
    if "confidence" in q or "uncertain" in q or "why low" in q or "why medium" in q or "reliability" in q:
        return QuestionRoute(
            intent="confidence_reasoning",
            sub_intent="CONFIDENCE_REASONING",
            patient_specific=True,
            required_context_blocks=["diagnosis", "confidence", "evidence"],
            kb_required=False,
            trace_level="compact",
            context_budget_tokens=1000,
            required_source=ReasoningMode.PIPELINE_ONLY,
            required_sources="PIPELINE"
        )

    # OOD explanation
    if "ood" in q or "out of distribution" in q or "out-of-distribution" in q:
        return QuestionRoute(
            intent="ood_explanation",
            sub_intent="OOD_EXPLANATION",
            patient_specific=True,
            required_context_blocks=["diagnosis", "evidence", "confidence"],
            kb_required=False,
            trace_level="compact",
            context_budget_tokens=1000,
            required_source=ReasoningMode.PIPELINE_ONLY,
            required_sources="PIPELINE"
        )
        
    # Rare Case Finder check
    if "rare" in q or "unusual" in q or "atypical" in q:
        return QuestionRoute(
            intent="rare_case",
            patient_specific=True,
            required_context_blocks=["diagnosis", "confidence"],
            kb_required=False,
            trace_level="compact",
            context_budget_tokens=1000,
            required_source=ReasoningMode.PIPELINE_ONLY,
            required_sources="PIPELINE"
        )
        
    # Patient Education / Simplified Language
    if "explain to patient" in q or "tell patient" in q or "education" in q or "simplify" in q or "medical student" in q or "layman" in q or "patient perspective" in q:
        return QuestionRoute(
            intent="patient_education",
            patient_specific=is_about_patient,
            required_context_blocks=["diagnosis", "kb_summary"],
            kb_required=True,
            trace_level="compact",
            context_budget_tokens=1000,
            required_source=ReasoningMode.PIPELINE_PLUS_KB,
            required_sources="COMBINED"
        )
        
    # Default Diagnosis Explanation & Diagnosis Reasoning
    if "why" in q or "evidence supports" in q or "chosen" in q or "chosen?" in q or "diagnosis chosen" in q:
        return QuestionRoute(
            intent="diagnosis_reasoning",
            sub_intent="DIAGNOSIS_REASONING",
            patient_specific=True,
            required_context_blocks=["diagnosis", "bridge", "evidence", "confidence", "trace_summary"],
            kb_required=False,
            trace_level="compact",
            context_budget_tokens=1200,
            required_source=ReasoningMode.PIPELINE_ONLY,
            required_sources="PIPELINE"
        )
        
    if "diagnosis" in q or "diagnose" in q or "stemi" in q or "mi" in q or "myocardial" in q:
        return QuestionRoute(
            intent="diagnosis",
            patient_specific=True,
            required_context_blocks=["diagnosis", "bridge", "evidence", "confidence"],
            kb_required=False,
            trace_level="compact",
            context_budget_tokens=1200,
            required_source=ReasoningMode.PIPELINE_ONLY,
            required_sources="PIPELINE"
        )
        
    # General Conversation
    return QuestionRoute(
        intent="diagnosis",
        patient_specific=is_about_patient,
        required_context_blocks=["diagnosis", "evidence", "confidence"],
        kb_required=False,
        trace_level="compact",
        context_budget_tokens=800,
        required_source=ReasoningMode.PIPELINE_ONLY,
        required_sources="PIPELINE"
    )


def route_question(user_question: str, conversation_state=None) -> QuestionRoute:
    norm_res = normalize_spelling_typos(user_question)
    norm_q = norm_res["normalized_question"]
    
    if norm_q.startswith("CLARIFICATION_REQUIRED:"):
        route = QuestionRoute(
            intent="clarification_request",
            required_context_blocks=[],
            kb_required=False,
            trace_level="compact",
            context_budget_tokens=400,
            required_source=ReasoningMode.GENERAL_MEDICAL,
            required_sources="GENERAL_MEDICAL"
        )
        route.original_question = user_question
        route.normalized_question = norm_q
        route.corrections = norm_res.get("corrections", [])
        route.correction_confidence = norm_res.get("correction_confidence", 0.0)
        route.clarification_options = norm_res.get("clarification_options", [])
        return route
        
    route = _route_question_internal(norm_q, conversation_state)
    route.original_question = user_question
    route.normalized_question = norm_q
    route.corrections = norm_res.get("corrections", [])
    route.correction_confidence = norm_res.get("correction_confidence", 1.0)
    return route
