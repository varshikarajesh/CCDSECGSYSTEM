"""Deterministic ECG statistics lookup and bridge-support evaluation.

This module deliberately performs no diagnosis and contains no OOD logic. Missing
measurements are returned as not evaluable rather than interpreted as normal.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional


DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "knowledge" / "ecg_statistics_rules.json"


class ECGStatisticsKnowledge:
    def __init__(self, rules_path: Optional[Path] = None):
        self.rules_path = Path(rules_path or DEFAULT_RULES_PATH)
        with self.rules_path.open("r", encoding="utf-8") as handle:
            self.db = json.load(handle)

    @staticmethod
    def _numbers(values: Iterable[Any]) -> List[float]:
        return [float(v) for v in values if v is not None and math.isfinite(float(v)) and float(v) > 0]

    def calculate(self, measurements: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        warnings: List[str] = []
        rr = self._numbers(measurements.get("nn_intervals_ms", []))
        duration = float(measurements.get("duration_seconds", 0) or 0)
        beats = measurements.get("beat_count")

        if rr:
            out["mean_rr_ms"] = mean(rr)
            out["median_rr_ms"] = median(rr)
            if len(rr) >= 3:
                diffs = [rr[i] - rr[i - 1] for i in range(1, len(rr))]
                out["sdnn_ms"] = stdev(rr)
                out["rmssd_ms"] = math.sqrt(mean(d * d for d in diffs))
                out["sdsd_ms"] = stdev(diffs) if len(diffs) >= 2 else 0.0
                out["pnn50_percent"] = 100.0 * sum(abs(d) > 50 for d in diffs) / len(diffs)
                out["pnn20_percent"] = 100.0 * sum(abs(d) > 20 for d in diffs) / len(diffs)
                out["cvrr_percent"] = 100.0 * out["sdnn_ms"] / out["mean_rr_ms"]
                sorted_rr = sorted(rr)
                q1 = self._percentile(sorted_rr, 25)
                q3 = self._percentile(sorted_rr, 75)
                med = out["median_rr_ms"]
                out["rr_iqr_ms"] = q3 - q1
                out["rr_mad_ms"] = median([abs(v - med) for v in rr])
            out["heart_rate_bpm"] = 60000.0 / out["mean_rr_ms"]
        elif beats is not None and duration > 0:
            out["heart_rate_bpm"] = 60.0 * float(beats) / duration

        qt = measurements.get("qt_ms")
        rr_for_qt = measurements.get("rr_ms", out.get("mean_rr_ms"))
        hr = measurements.get("heart_rate_bpm", out.get("heart_rate_bpm"))
        if qt is not None and rr_for_qt is not None and float(qt) > 0 and float(rr_for_qt) > 0:
            qt, rr_for_qt = float(qt), float(rr_for_qt)
            rr_s = rr_for_qt / 1000.0
            out["qtc_bazett_ms"] = qt / math.sqrt(rr_s)
            out["qtc_fridericia_ms"] = qt / (rr_s ** (1.0 / 3.0))
            out["qtc_framingham_ms"] = qt + 154.0 * (1.0 - rr_s)
            if hr is not None and float(hr) > 0:
                out["qtc_hodges_ms"] = qt + 1.75 * (float(hr) - 60.0)
            if float(hr or 0) > 100 or float(hr or 0) < 60:
                warnings.append("Bazett QTc is rate-sensitive; prefer Fridericia and retain all formula outputs.")

        voltage_fields = ("s_v1_mv", "r_v5_mv", "r_v6_mv")
        if all(measurements.get(k) is not None for k in voltage_fields):
            out["sokolow_lyon_mv"] = abs(float(measurements["s_v1_mv"])) + max(
                float(measurements["r_v5_mv"]), float(measurements["r_v6_mv"])
            )
        if measurements.get("s_v3_mv") is not None and measurements.get("r_avl_mv") is not None:
            out["cornell_voltage_mv"] = abs(float(measurements["s_v3_mv"])) + float(measurements["r_avl_mv"])

        if rr and duration and duration < 120:
            warnings.append("Short-window HRV is descriptive only and must not be compared with long-recording reference ranges.")
        return {"values": {k: round(v, 6) for k, v in out.items()}, "warnings": warnings}

    def evaluate(self, measurements: Mapping[str, Any], scope: str = "representative_window") -> Dict[str, Any]:
        calculated = self.calculate(measurements)
        merged = dict(measurements)
        merged.update(calculated["values"])
        supporting, contradicting, not_evaluable = [], [], []

        for rule in self.db["support_rules"]:
            state = self._rule_state(rule, merged)
            if state is None:
                not_evaluable.append(rule["id"])
            elif state:
                supporting.append(self._evidence(rule, scope))
        for rule in self.db["contradiction_rules"]:
            state = self._conditions(rule.get("when_any", []), merged, mode="any")
            if state is None:
                not_evaluable.append(rule["id"])
            elif state:
                contradicting.append({
                    "rule_id": rule["id"], "labels": rule["contradicts"],
                    "weight": rule["weight"], "scope": scope
                })

        return {
            "schema_version": self.db["schema_version"],
            "scope": scope,
            "calculated_metrics": {**{k: v for k, v in measurements.items() if isinstance(v, (int, float))}, **calculated["values"]},
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "not_evaluable": sorted(set(not_evaluable)),
            "warnings": calculated["warnings"],
            "ood_used": False,
            "decision_authority": self.db["safety"]["authority"]
        }

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * percentile / 100.0
        lo, hi = math.floor(position), math.ceil(position)
        if lo == hi:
            return values[lo]
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    def _rule_state(self, rule: Mapping[str, Any], values: Mapping[str, Any]) -> Optional[bool]:
        if "all" in rule:
            return self._conditions(rule["all"], values, "all")
        return self._conditions(rule.get("any", []), values, "any")

    def _conditions(self, conditions: List[Mapping[str, Any]], values: Mapping[str, Any], mode: str) -> Optional[bool]:
        states = []
        for condition in conditions:
            field = condition["field"]
            if field not in values or values[field] is None:
                states.append(None)
                continue
            states.append(self._compare(values[field], condition["operator"], condition["value"]))
        if mode == "all":
            if False in states:
                return False
            return None if None in states else True
        if True in states:
            return True
        return None if None in states else False

    @staticmethod
    def _compare(actual: Any, operator: str, expected: Any) -> bool:
        if operator == "<": return actual < expected
        if operator == "<=": return actual <= expected
        if operator == ">": return actual > expected
        if operator == ">=": return actual >= expected
        if operator == "==": return actual == expected
        if operator == "between": return expected[0] <= actual <= expected[1]
        raise ValueError(f"Unsupported rule operator: {operator}")

    @staticmethod
    def _evidence(rule: Mapping[str, Any], scope: str) -> Dict[str, Any]:
        return {
            "rule_id": rule["id"], "labels": rule["supports"],
            "weight": rule["weight"], "scope": scope,
            "specificity": rule.get("specificity", "moderate"),
            "red_flag": rule.get("red_flag", False),
            "caution": rule.get("caution"),
            "source_ids": rule.get("source_ids", [])
        }

    def lookup_metric(self, metric_id: str) -> Optional[Dict[str, Any]]:
        return next((m for m in self.db["metrics"] if m["id"] == metric_id), None)

    def lookup_label_rules(self, label: str) -> List[Dict[str, Any]]:
        label = label.upper()
        return [r for r in self.db["support_rules"] if label in r.get("supports", [])]
