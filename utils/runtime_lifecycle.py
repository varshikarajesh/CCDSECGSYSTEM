# -*- coding: utf-8 -*-
"""
utils/runtime_lifecycle.py

Authoritative thread-safe process-wide lifecycle counter registry.
Tracks model loads, runtime instances, wrapper instances, and backend instances dynamically.
"""

from threading import Lock

_LOCK = Lock()

_COUNTERS = {
    "pipeline_runner_instances": 0,
    "llm_wrapper_instances": 0,
    "real_llm_backend_instances": 0,
    "model_load_count": 0,
}


def increment_counter(name: str) -> int:
    """Increments a counter by 1 in a thread-safe manner."""
    with _LOCK:
        if name not in _COUNTERS:
            _COUNTERS[name] = 0
        _COUNTERS[name] += 1
        return _COUNTERS[name]


def get_counter(name: str) -> int:
    """Gets the current integer value of a counter."""
    with _LOCK:
        return int(_COUNTERS.get(name, 0))


def get_lifecycle_snapshot() -> dict:
    """Returns a dictionary snapshot of all current lifecycle counters."""
    with _LOCK:
        return dict(_COUNTERS)


def reset_counters_for_testing():
    """Resets counters for test suite initialization (never called during normal inference)."""
    with _LOCK:
        for k in _COUNTERS:
            _COUNTERS[k] = 0