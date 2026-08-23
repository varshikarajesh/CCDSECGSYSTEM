# -*- coding: utf-8 -*-
"""
streamlit_components/session_state.py

Session state management for the TRACE Streamlit testing interface.
Initializes persistent UI state variables without triggering unnecessary inference re-execution.
"""

import streamlit as st


def init_session_state() -> None:
    """Initializes Streamlit session state keys with default values if not present."""
    defaults = {
        "uploaded_file_bytes": None,
        "uploaded_filename": None,
        "parsed_ecg_signal": None,
        "inference_result": None,
        "upstream_snapshot": None,
        "selected_neighbor_id": None,
        "selected_neighbor_waveform": None,
        "api_health": None,
        "api_config": None,
        "feedback_submitted": False,
        "feedback_response": None,
        "chat_messages": None,
    }

    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val
