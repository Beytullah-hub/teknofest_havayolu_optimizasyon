"""Rotation matching coverage and skip diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ModelConfig
from .model import normalize_flight_no


SKIP_ANALYSIS_COLUMNS = [
    "rotation_id",
    "pair_id",
    "outbound_flight_no_raw",
    "inbound_flight_no_raw",
    "outbound_flight_no_normalized",
    "inbound_flight_no_normalized",
    "outbound_origin",
    "outbound_destination",
    "inbound_origin",
    "inbound_destination",
    "skip_reason",
    "candidate_in_selected_subset",
    "outbound_found_in_schedule",
    "inbound_found_in_schedule",
    "outbound_found_in_od_table",
    "inbound_found_in_od_table",
    "route_match_found",
    "day_match_found",
    "possible_reason_category",
]


def build_rotation_skip_analysis(
    rotation_candidates: pd.DataFrame | None,
    optimized_schedule: pd.DataFrame | None,
    raw_connections: pd.DataFrame | None,
    config: ModelConfig,
) -> pd.DataFrame:
    """Build per-skipped-candidate rotation matching diagnostics."""

    if rotation_candidates is None or rotation_candidates.empty:
        return pd.DataFrame(columns=SKIP_ANALYSIS_COLUMNS)

    candidates = rotation_candidates.copy()
    skipped = candidates[
        (~_bool_series(candidates.get("rotation_constraint_available"), len(candidates)))
        | (candidates.get("skip_reason", pd.Series([""] * len(candidates))).fillna("").astype(str) != "")
    ].copy()
    if skipped.empty:
        return pd.DataFrame(columns=SKIP_ANALYSIS_COLUMNS)

    schedule_legs = _schedule_leg_frame(optimized_schedule, config)
    od_legs = _od_leg_frame(raw_connections, config)
    rows = []
    for cand in skipped.itertuples(index=False):
        outbound_raw = getattr(cand, "outbound_flight_no", None)
        inbound_raw = getattr(cand, "inbound_flight_no", None)
        outbound_norm = normalize_flight_no(outbound_raw)
        inbound_norm = normalize_flight_no(inbound_raw)
        out_origin = _airport(getattr(cand, "outbound_origin", None))
        out_dest = _airport(getattr(cand, "outbound_destination", None))
        in_origin = _airport(getattr(cand, "inbound_origin", None))
        in_dest = _airport(getattr(cand, "inbound_destination", None))
        day = _safe_int(getattr(cand, "day", None))

        out_sched = _has_leg(schedule_legs, "outbound", outbound_norm, out_origin, out_dest, day)
        in_sched = _has_leg(schedule_legs, "inbound", inbound_norm, in_origin, in_dest, day)
        out_od_no = _has_flight(od_legs, "outbound", outbound_norm)
        in_od_no = _has_flight(od_legs, "inbound", inbound_norm)
        out_route = _has_leg(od_legs, "outbound", outbound_norm, out_origin, out_dest, None)
        in_route = _has_leg(od_legs, "inbound", inbound_norm, in_origin, in_dest, None)
        out_day = _has_leg(od_legs, "outbound", outbound_norm, out_origin, out_dest, day)
        in_day = _has_leg(od_legs, "inbound", inbound_norm, in_origin, in_dest, day)
        route_match = bool(out_route and in_route)
        day_match = bool(route_match and (day is None or (out_day and in_day)))
        in_subset = bool(out_sched and in_sched)
        raw_skip = _clean_text(getattr(cand, "skip_reason", ""))
        format_mismatch = _format_mismatch(od_legs, "outbound", outbound_raw, outbound_norm) or _format_mismatch(
            od_legs, "inbound", inbound_raw, inbound_norm
        )
        category = _classify_skip(
            raw_skip,
            in_subset,
            bool(out_od_no),
            bool(in_od_no),
            route_match,
            day_match,
            format_mismatch,
        )
        effective_skip = "subset_filter_excluded" if category == "subset_filter_excluded" else raw_skip
        rows.append(
            {
                "rotation_id": getattr(cand, "rotation_id", None),
                "pair_id": getattr(cand, "pair_id", None),
                "outbound_flight_no_raw": outbound_raw,
                "inbound_flight_no_raw": inbound_raw,
                "outbound_flight_no_normalized": outbound_norm,
                "inbound_flight_no_normalized": inbound_norm,
                "outbound_origin": out_origin,
                "outbound_destination": out_dest,
                "inbound_origin": in_origin,
                "inbound_destination": in_dest,
                "skip_reason": effective_skip,
                "candidate_in_selected_subset": in_subset,
                "outbound_found_in_schedule": bool(out_sched),
                "inbound_found_in_schedule": bool(in_sched),
                "outbound_found_in_od_table": bool(out_od_no),
                "inbound_found_in_od_table": bool(in_od_no),
                "route_match_found": route_match,
                "day_match_found": day_match,
                "possible_reason_category": category,
            }
        )
    return pd.DataFrame(rows, columns=SKIP_ANALYSIS_COLUMNS)


def apply_rotation_skip_categories(rotation_candidates: pd.DataFrame | None, skip_analysis: pd.DataFrame | None) -> pd.DataFrame | None:
    """Mark subset-driven skips directly on rotation_candidates for clearer downstream reports."""

    if rotation_candidates is None or rotation_candidates.empty or skip_analysis is None or skip_analysis.empty:
        return rotation_candidates
    out = rotation_candidates.copy()
    subset_ids = set(
        skip_analysis.loc[
            skip_analysis["possible_reason_category"].astype(str) == "subset_filter_excluded",
            "rotation_id",
        ].astype(str)
    )
    if subset_ids and "rotation_id" in out.columns:
        mask = out["rotation_id"].astype(str).isin(subset_ids)
        out.loc[mask, "skip_reason"] = "subset_filter_excluded"
    return out


def summarize_rotation_skip_analysis(
    rotation_candidates: pd.DataFrame | None,
    skip_analysis: pd.DataFrame | None,
) -> dict:
    """Return rotation coverage metrics for summaries and experiment logs."""

    total = int(len(rotation_candidates)) if rotation_candidates is not None and not rotation_candidates.empty else 0
    if rotation_candidates is not None and not rotation_candidates.empty and "rotation_constraint_available" in rotation_candidates:
        active = int(rotation_candidates["rotation_constraint_available"].fillna(False).astype(bool).sum())
    else:
        active = 0
    coverage = 100.0 * active / total if total else 0.0

    counts = {}
    if skip_analysis is not None and not skip_analysis.empty:
        counts = skip_analysis["possible_reason_category"].fillna("unknown").astype(str).value_counts().to_dict()

    elapsed_exact = 0
    legacy_exact = 0
    fallback = 0
    if rotation_candidates is not None and not rotation_candidates.empty:
        active_candidates = rotation_candidates[
            rotation_candidates.get("rotation_constraint_available", pd.Series([False] * len(rotation_candidates))).fillna(False).astype(bool)
        ]
        if "block_time_source" in active_candidates.columns:
            sources = active_candidates["block_time_source"].fillna("").astype(str)
            elapsed_exact = int((sources == "elapsedtime_exact_match").sum())
            legacy_exact = int((sources == "od_exact_match_legacy").sum())
            fallback = int((sources == "schedule_elapsed_fallback").sum())
        elif "derivation_method" in active_candidates.columns:
            derivation = active_candidates["derivation_method"].fillna("").astype(str)
            elapsed_exact = int(derivation.str.startswith("ElapsedTime1/ElapsedTime2 exact match").sum())
            legacy_exact = int(derivation.str.startswith("O&D connection block total").sum())
            fallback = int(derivation.str.startswith("approximated from schedule elapsed").sum())
    exact = elapsed_exact + legacy_exact

    return {
        "rotation_coverage_pct": coverage,
        "rotation_skip_subset_excluded": int(counts.get("subset_filter_excluded", 0)),
        "rotation_skip_format_mismatch": int(counts.get("flight_number_format_mismatch", 0)),
        "rotation_skip_route_mismatch": int(counts.get("route_mismatch", 0)),
        "rotation_skip_missing_block_time": int(counts.get("missing_block_time", 0)),
        "rotation_block_elapsedtime_match_count": elapsed_exact,
        "rotation_block_legacy_exact_match_count": legacy_exact,
        "rotation_block_exact_match_count": exact,
        "rotation_block_fallback_count": fallback,
    }


def write_rotation_skip_outputs(
    run_dir: Path,
    rotation_candidates: pd.DataFrame | None,
    skip_analysis: pd.DataFrame | None,
    metrics: dict,
) -> None:
    """Write rotation skip analysis workbook and text summary."""

    run_dir.mkdir(parents=True, exist_ok=True)
    if skip_analysis is None:
        skip_analysis = pd.DataFrame(columns=SKIP_ANALYSIS_COLUMNS)
    skip_analysis.to_excel(run_dir / "rotation_skip_analysis.xlsx", index=False)

    total = int(metrics.get("rotation_candidates_total", 0))
    active = int(metrics.get("rotation_constraints_active", 0))
    skipped = int(metrics.get("rotation_constraints_skipped", 0))
    skip_rate = 100.0 * skipped / total if total else 0.0
    lines = [
        "ROTATION SKIP SUMMARY",
        "---------------------",
        f"total candidates: {total}",
        f"active constraints: {active}",
        f"skipped constraints: {skipped}",
        f"skip rate: {skip_rate:.6f}%",
        f"outbound found count: {_sum_bool(skip_analysis, 'outbound_found_in_schedule')}",
        f"inbound found count: {_sum_bool(skip_analysis, 'inbound_found_in_schedule')}",
        f"both legs found count: {int((skip_analysis.get('outbound_found_in_schedule', pd.Series(dtype=bool)).fillna(False) & skip_analysis.get('inbound_found_in_schedule', pd.Series(dtype=bool)).fillna(False)).sum()) if not skip_analysis.empty else 0}",
        f"elapsedtime exact block match count: {metrics.get('rotation_block_elapsedtime_match_count', 0)}",
        f"legacy exact block match count: {metrics.get('rotation_block_legacy_exact_match_count', 0)}",
        f"exact block match count total: {metrics.get('rotation_block_exact_match_count', 0)}",
        f"fallback block count: {metrics.get('rotation_block_fallback_count', 0)}",
        "",
        "skip counts by reason:",
    ]
    if skip_analysis.empty:
        lines.append("- none")
    else:
        for reason, count in skip_analysis["skip_reason"].fillna("unknown").astype(str).value_counts().items():
            lines.append(f"- {reason}: {int(count)}")
    lines.append("")
    lines.append("skip counts by possible_reason_category:")
    if skip_analysis.empty:
        lines.append("- none")
    else:
        for reason, count in skip_analysis["possible_reason_category"].fillna("unknown").astype(str).value_counts().items():
            lines.append(f"- {reason}: {int(count)}")
    (run_dir / "rotation_skip_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _schedule_leg_frame(schedule: pd.DataFrame | None, config: ModelConfig) -> pd.DataFrame:
    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=["direction", "flight_norm", "flight_raw_normless", "origin", "destination", "day"])
    rows = []
    hub = _airport(config.hub)
    for row in schedule.itertuples(index=False):
        origin = _airport(getattr(row, "origin", None))
        destination = _airport(getattr(row, "destination", None))
        direction = "outbound" if origin == hub else "inbound" if destination == hub else ""
        if not direction:
            continue
        raw = getattr(row, "flight_no", None)
        rows.append(
            {
                "direction": direction,
                "flight_norm": normalize_flight_no(raw),
                "flight_raw_normless": _raw_flight_text(raw),
                "origin": origin,
                "destination": destination,
                "day": _safe_int(getattr(row, "day", None)),
            }
        )
    return pd.DataFrame(rows)


def _od_leg_frame(connections: pd.DataFrame | None, config: ModelConfig) -> pd.DataFrame:
    if connections is None or connections.empty:
        return pd.DataFrame(columns=["direction", "flight_norm", "flight_raw_normless", "origin", "destination", "day"])
    rows = []
    hub = _airport(config.hub)
    for row in connections.itertuples(index=False):
        day = _safe_int(getattr(row, "day", getattr(row, "Gün", None)))
        cr1 = _clean_text(getattr(row, "Cr1", "")).upper()
        cr2 = _clean_text(getattr(row, "Cr2", "")).upper()
        dep1 = _airport(getattr(row, "Dep1", None))
        arr1 = _airport(getattr(row, "Arr1", None))
        dep2 = _airport(getattr(row, "Dep2", None))
        arr2 = _airport(getattr(row, "Arr2", None))
        inbound_flight = getattr(row, "inbound_flight", getattr(row, "FlNo1", None))
        outbound_flight = getattr(row, "outbound_flight", getattr(row, "FlNo2", None))
        if cr1 == "TK" and arr1 == hub:
            rows.append(_leg_row("inbound", inbound_flight, dep1, arr1, day))
        if cr2 == "TK" and dep2 == hub:
            rows.append(_leg_row("outbound", outbound_flight, dep2, arr2, day))
    return pd.DataFrame(rows)


def _leg_row(direction: str, flight_no, origin: str, destination: str, day: int | None) -> dict:
    return {
        "direction": direction,
        "flight_norm": normalize_flight_no(flight_no),
        "flight_raw_normless": _raw_flight_text(flight_no),
        "origin": origin,
        "destination": destination,
        "day": day,
    }


def _has_flight(legs: pd.DataFrame, direction: str, flight_norm: str) -> bool:
    if legs.empty or not flight_norm:
        return False
    return bool(((legs["direction"] == direction) & (legs["flight_norm"] == flight_norm)).any())


def _has_leg(legs: pd.DataFrame, direction: str, flight_norm: str, origin: str, destination: str, day: int | None) -> bool:
    if legs.empty or not flight_norm or not origin or not destination:
        return False
    mask = (
        (legs["direction"] == direction)
        & (legs["flight_norm"] == flight_norm)
        & (legs["origin"] == origin)
        & (legs["destination"] == destination)
    )
    if day is not None and "day" in legs.columns:
        mask = mask & (legs["day"] == day)
    return bool(mask.any())


def _format_mismatch(legs: pd.DataFrame, direction: str, raw_flight, flight_norm: str) -> bool:
    if legs.empty or not flight_norm:
        return False
    raw_text = _raw_flight_text(raw_flight)
    matching_norm = legs[(legs["direction"] == direction) & (legs["flight_norm"] == flight_norm)]
    if matching_norm.empty:
        return False
    return not bool((matching_norm["flight_raw_normless"] == raw_text).any())


def _classify_skip(
    skip_reason: str,
    in_subset: bool,
    out_od_no: bool,
    in_od_no: bool,
    route_match: bool,
    day_match: bool,
    format_mismatch: bool,
) -> str:
    text = skip_reason.lower()
    if "block" in text:
        return "missing_block_time"
    if route_match and not day_match:
        return "missing_day_match"
    if route_match and not in_subset:
        return "subset_filter_excluded"
    if format_mismatch:
        return "flight_number_format_mismatch"
    if (out_od_no or in_od_no) and not route_match:
        return "route_mismatch"
    if "missing modeled" in text or "missing schedule" in text:
        return "missing_schedule_time"
    return "unknown"


def _bool_series(series, length: int) -> pd.Series:
    if series is None:
        return pd.Series([False] * length)
    if isinstance(series, pd.Series):
        if series.dtype == object:
            return series.fillna(False).map(_to_bool)
        return series.fillna(False).astype(bool)
    return pd.Series([_to_bool(series)] * length)


def _to_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _airport(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _raw_flight_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper().replace(" ", "")
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _safe_int(value) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _sum_bool(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].fillna(False).astype(bool).sum())
