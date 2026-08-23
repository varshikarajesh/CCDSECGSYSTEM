# Evidence Bridge V4 contract

V4 is a deterministic, multi-label, window-aware decision-support bridge. It does not use OOD and the LLM cannot change its output.

## Input object

```json
{
  "classifier": {
    "probabilities": {"AFIB": 0.82, "CRBBB": 0.18},
    "windows": [{"window_id": "w3", "probabilities": {"AFIB": 0.88}}]
  },
  "retrieval": {
    "neighbors": [{"similarity": 0.90, "scp_codes": ["AFIB"], "query_window_id": "w3"}]
  },
  "holter": {
    "recording_probabilities": {"AF": 0.84},
    "windows": [{"window_id": "w3", "probabilities": {"AF": 0.91}}]
  },
  "statistics": {
    "overall": {"nn_intervals_ms": [700, 1020, 610, 950]},
    "windows": {"w3": {"rr_irregularly_irregular": true, "distinct_p_waves": false}}
  },
  "selected_windows": [
    {"window_id": "w3", "start_seconds": 30, "end_seconds": 40, "role": "abnormal_episode", "selected": true},
    {"window_id": "w0", "start_seconds": 0, "end_seconds": 10, "role": "stable_reference", "selected": true}
  ],
  "signal_quality": {"overall_quality_score": 0.94},
  "metadata": {"mode": "5min", "sampling_rate_hz": 100}
}
```

## Evidence policy

- A supported finding requires a classifier or retrieval anchor and at least two independent evidence sources.
- Statistics and Holter outputs can support or contradict a candidate but cannot independently establish a final diagnosis.
- Missing data is `not_evaluable`, never negative evidence.
- Supported abnormal findings suppress `NORM`; coexisting abnormal labels remain independent.
- Finding-to-window attribution is preserved.
- Every attributed finding reports exact start/end seconds and a `MM:SS.s-MM:SS.s` display interval when timing is supplied.
- Stable reference intervals explain what appeared normal/stable without treating absence of an alert as proof of normality.
- Only knowledge chunks that pass document hash and fingerprint validation can be cited.
- Each citation includes the supporting source passage as `evidence_text`, together with title, organization/authors, section, locator, DOI/URL, and document hash.
- Signal quality can limit or block the decision; OOD is not calculated.

## Command

```powershell
python run_bridge_v4.py --input evidence_package.json --output bridge_v4_result.json
```
