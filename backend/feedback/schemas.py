# -*- coding: utf-8 -*-
"""
schemas.py

API Schemas and Data Transfer Objects (DTOs) for Feedback Platform.
"""

from typing import Dict, Any, List, Optional


class FeedbackRequestSchema:
    """Validator and DTO schema for incoming feedback API requests."""

    VALID_CORRECTNESS = {
        "Correct", "Partially Correct", "Incorrect", "Cannot Determine", "Poor ECG Quality"
    }
    
    VALID_FAMILIES = {
        "Normal", "Rhythm", "Conduction", "Infarction", "Hypertrophy",
        "Repolarization", "Ischemia", "Pacing", "Other"
    }

    VALID_CONFIDENCE_RATINGS = {"Too High", "Appropriate", "Too Low"}
    
    VALID_EXPLANATION_RATINGS = {
        "Very Useful", "Useful", "Neutral", "Misleading", "Incorrect"
    }

    VALID_RELEVANCE = {
        "Highly Relevant", "Relevant", "Somewhat Relevant", "Not Relevant", "Misleading"
    }
