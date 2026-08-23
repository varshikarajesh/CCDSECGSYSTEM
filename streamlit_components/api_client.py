# -*- coding: utf-8 -*-
"""
streamlit_components/api_client.py

API Client wrapper for connecting Streamlit to the TRACE FastAPI backend.
Handles health checks, inference requests, neighbor waveform lookups, and clinician feedback submission.
"""

from typing import Any, Dict, Optional, Tuple
import requests


class ApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout_seconds: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get_health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/api/health"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_config(self) -> Dict[str, Any]:
        url = f"{self.base_url}/api/config"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def run_inference(
        self,
        file_bytes: bytes,
        filename: str,
        sampling_rate_hz: int = 100,
        top_k: int = 5,
        include_retrieval: bool = True,
        include_knowledge: bool = True,
        include_explanation: bool = True,
        llm_mode: Optional[str] = None,
        llm_backend: Optional[str] = None,
        ecg_id: Optional[int] = None,
        case_id: Optional[str] = None,
        question: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], int]:
        url = f"{self.base_url}/api/inference"
        files = {
            "file": (filename, file_bytes, "application/octet-stream")
        }
        data = {
            "sampling_rate_hz": str(sampling_rate_hz),
            "top_k": str(top_k),
            "include_retrieval": str(include_retrieval).lower(),
            "include_knowledge": str(include_knowledge).lower(),
            "include_explanation": str(include_explanation).lower(),
        }
        if question:
            data["question"] = question
        if llm_mode:
            data["llm_mode"] = llm_mode
        if llm_backend:
            data["llm_backend"] = llm_backend
        if ecg_id is not None:
            data["ecg_id"] = str(ecg_id)
        if case_id:
            data["conversation_id"] = case_id

        try:
            resp = requests.post(url, files=files, data=data, timeout=self.timeout_seconds)
            status_code = resp.status_code
            try:
                payload = resp.json()
            except Exception:
                payload = {"detail": resp.text}

            if status_code == 200:
                return True, payload, status_code
            else:
                return False, payload, status_code
        except requests.exceptions.RequestException as exc:
            return False, {"detail": f"Network/Connection Error: {exc}"}, 503

    def fetch_neighbor_waveform(self, ecg_id: int) -> Tuple[bool, Dict[str, Any]]:
        url = f"{self.base_url}/api/retrieval/neighbors/{ecg_id}/waveform"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return True, resp.json()
            else:
                try:
                    err = resp.json()
                except Exception:
                    err = {"detail": resp.text}
                return False, err
        except requests.exceptions.RequestException as exc:
            return False, {"detail": f"Request failed: {exc}"}

    def run_recording_inference(
        self,
        file_bytes: bytes,
        filename: str,
        recording_mode: str,
        sampling_rate_hz: int = 100,
        top_k: int = 5,
        manual_window_indices: Optional[list] = None,
    ) -> Tuple[bool, Dict[str, Any], int]:
        url = f"{self.base_url}/api/recording-inference"
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        data = {
            "recording_mode": recording_mode,
            "sampling_rate_hz": str(sampling_rate_hz),
            "top_k": str(top_k),
            "manual_window_indices_json": __import__("json").dumps(manual_window_indices or []),
        }
        try:
            resp = requests.post(url, files=files, data=data, timeout=self.timeout_seconds)
            try:
                payload = resp.json()
            except Exception:
                payload = {"detail": resp.text}
            return resp.status_code == 200, payload, resp.status_code
        except requests.exceptions.RequestException as exc:
            return False, {"detail": f"Network/Connection Error: {exc}"}, 503

    def fetch_recording_window(self, recording_id: str, start_seconds: float, end_seconds: float):
        url = f"{self.base_url}/api/recordings/{recording_id}/window"
        try:
            response = requests.get(
                url,
                params={"start_seconds": start_seconds, "end_seconds": end_seconds},
                timeout=self.timeout_seconds,
            )
            payload = response.json()
            return response.ok, payload
        except requests.RequestException as exc:
            return False, {"detail": f"Network/Connection Error: {exc}"}

    def submit_feedback(self, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], int]:
        url = f"{self.base_url}/api/feedback"
        try:
            resp = requests.post(url, json=payload, timeout=10)
            status_code = resp.status_code
            try:
                res_payload = resp.json()
            except Exception:
                res_payload = {"detail": resp.text}

            if status_code in (200, 201):
                return True, res_payload, status_code
            else:
                return False, res_payload, status_code
        except requests.exceptions.RequestException as exc:
            return False, {"detail": f"Feedback submission error: {exc}"}, 503
