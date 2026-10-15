"""MVP-vs-baseline analysis utilities.

These functions answer whether the MVP is doing meaningful optimization or
mostly reproducing the baseline on the same filtered subset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline import BaselineResult
from .metrics import compute_run_metrics


def compare_to_baseline(
    run_dir: Path,
    baseline_result: BaselineResult,
    current_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Write a deep baseline-current comparison and return summary metrics."""

    baseline_metrics = compute_run_metrics(
        baseline_result.od_metrics,
        baseline_result.selected_connections,
        baseline_result.beaten_competitors,
    )
    baseline_objective = float(baseline_metrics["total_objective"])
    current_objective = float(current_metrics["total_objective"])
    improvement = current_objective - baseline_objective
    improvement_pct = 100.0 * improvement / abs(baseline_objective) if abs(baseline_objective) > 1e-12 else 0.0
    current_od = pd.read_excel(run_dir / "od_metrics.xlsx", engine="openpyxl")
    od_counts = _od_reward_status_counts(baseline_result.od_metrics, current_od)
    summary = {
        "baseline_objective": baseline_objective,
        "current_objective": current_objective,
        "absolute_improvement": improvement,
        "percentage_improvement": improvement_pct,
        "baseline_active_connections": int(baseline_metrics["active_connections"]),
        "current_active_connections": int(current_metrics["active_connections"]),
        "baseline_beaten_competitors": int(baseline_metrics["beaten_competitors"]),
        "current_beaten_competitors": int(current_metrics["beaten_competitors"]),
        "baseline_average_connection_gap": float(baseline_metrics["average_connection_gap"]),
        "current_average_connection_gap": float(current_metrics["average_connection_gap"]),
        "baseline_ranking_reward": float(baseline_metrics["ranking_reward"]),
        "current_ranking_reward": float(current_metrics["ranking_reward"]),
        "baseline_connection_reward": float(baseline_metrics["connection_reward"]),
        "current_connection_reward": float(current_metrics["connection_reward"]),
        "improved_od_count": od_counts["improved_od_count"],
        "worsened_od_count": od_counts["worsened_od_count"],
        "unchanged_od_count": od_counts["unchanged_od_count"],
    }
    lines = [
        "COMPARISON TO BASELINE",
        "----------------------",
        f"baseline objective: {summary['baseline_objective']}",
        f"current objective: {summary['current_objective']}",
        f"absolute improvement: {summary['absolute_improvement']}",
        f"percentage improvement: {summary['percentage_improvement']:.6f}%",
        f"baseline active connections: {summary['baseline_active_connections']}",
        f"current active connections: {summary['current_active_connections']}",
        f"baseline beaten competitors: {summary['baseline_beaten_competitors']}",
        f"current beaten competitors: {summary['current_beaten_competitors']}",
        f"baseline average connection gap: {summary['baseline_average_connection_gap']}",
        f"current average connection gap: {summary['current_average_connection_gap']}",
        f"baseline ranking reward: {summary['baseline_ranking_reward']}",
        f"current ranking reward: {summary['current_ranking_reward']}",
        f"baseline connection reward: {summary['baseline_connection_reward']}",
        f"current connection reward: {summary['current_connection_reward']}",
        f"improved O-D count: {summary['improved_od_count']}",
        f"worsened O-D count: {summary['worsened_od_count']}",
        f"unchanged O-D count: {summary['unchanged_od_count']}",
    ]
    (run_dir / "comparison_to_baseline.txt").write_text("\n".join(lines), encoding="utf-8")
    return summary


def analyze_schedule_changes(run_dir: Path) -> dict[str, Any]:
    """Analyze whether optimized_schedule.xlsx actually changes flight times."""

    schedule = pd.read_excel(run_dir / "optimized_schedule.xlsx", engine="openpyxl")
    if schedule.empty:
        shifts = pd.Series(dtype=float)
    else:
        shifts = pd.to_numeric(schedule["time_change_min"], errors="coerce").fillna(0.0)
    total = int(len(shifts))
    changed = int((shifts != 0).sum())
    changed_pct = 100.0 * changed / total if total else 0.0
    summary = {
        "total_adjustable_flights": total,
        "flights_with_nonzero_shift": changed,
        "changed_flight_pct": changed_pct,
        "average_absolute_shift": float(shifts.abs().mean()) if total else 0.0,
        "max_absolute_shift": float(shifts.abs().max()) if total else 0.0,
        "min_shift": float(shifts.min()) if total else 0.0,
        "max_shift": float(shifts.max()) if total else 0.0,
    }
    lines = [
        "SCHEDULE CHANGE ANALYSIS",
        "------------------------",
        f"total adjustable flights: {summary['total_adjustable_flights']}",
        f"flights with nonzero shift: {summary['flights_with_nonzero_shift']}",
        f"percentage of changed flights: {summary['changed_flight_pct']:.6f}%",
        f"average absolute shift: {summary['average_absolute_shift']}",
        f"max absolute shift: {summary['max_absolute_shift']}",
        f"min shift: {summary['min_shift']}",
        f"max shift: {summary['max_shift']}",
    ]
    if changed == 0:
        lines.extend(
            [
                "",
                "WARNING: MVP did not change any flight times. This may indicate the optimization is only reproducing baseline.",
            ]
        )
    (run_dir / "schedule_change_analysis.txt").write_text("\n".join(lines), encoding="utf-8")
    return summary


def analyze_shift_distribution(run_dir: Path, max_shift: int | float | None = None) -> dict[str, Any]:
    """Write detailed shift distribution outputs from optimized_schedule.xlsx."""

    schedule = pd.read_excel(run_dir / "optimized_schedule.xlsx", engine="openpyxl")
    expected_columns = [
        "flight_id",
        "flight_no",
        "day",
        "origin",
        "destination",
        "old_ist_time",
        "new_ist_time",
        "time_change_min",
    ]
    if schedule.empty:
        distribution = pd.DataFrame(columns=expected_columns + ["abs_time_change_min", "shift_bucket"])
        shifts = pd.Series(dtype=float)
    else:
        distribution = schedule.copy()
        for col in expected_columns:
            if col not in distribution.columns:
                distribution[col] = None
        shifts = pd.to_numeric(distribution["time_change_min"], errors="coerce").fillna(0.0)
        distribution["time_change_min"] = shifts
        distribution["abs_time_change_min"] = shifts.abs()
        distribution["shift_bucket"] = distribution["abs_time_change_min"].map(_shift_bucket)
        distribution = distribution[expected_columns + ["abs_time_change_min", "shift_bucket"]]

    distribution.to_excel(run_dir / "shift_distribution.xlsx", index=False)

    total = int(len(shifts))
    changed = int((shifts != 0).sum()) if total else 0
    changed_pct = 100.0 * changed / total if total else 0.0
    zero_count = int((shifts == 0).sum()) if total else 0
    nonzero_count = changed
    abs_shifts = shifts.abs()
    average_shift = float(shifts.mean()) if total else 0.0
    average_abs_shift = float(abs_shifts.mean()) if total else 0.0
    median_abs_shift = float(abs_shifts.median()) if total else 0.0
    min_shift = float(shifts.min()) if total else 0.0
    max_shift_value = float(shifts.max()) if total else 0.0
    max_shift_limit = abs(float(max_shift)) if max_shift is not None else float(abs_shifts.max()) if total else 0.0
    if total and max_shift_limit > 0:
        at_neg_bound = int((shifts <= -max_shift_limit).sum())
        at_pos_bound = int((shifts >= max_shift_limit).sum())
    else:
        at_neg_bound = 0
        at_pos_bound = 0
    at_bound_count = at_neg_bound + at_pos_bound
    pct_at_shift_bounds = 100.0 * at_bound_count / total if total else 0.0
    small_shift_le_5 = int((abs_shifts <= 5).sum()) if total else 0
    pct_small_shift_le_5min = 100.0 * small_shift_le_5 / total if total else 0.0

    summary = {
        "total_adjustable_flights": total,
        "changed_flights_count": changed,
        "changed_flight_pct": changed_pct,
        "zero_shift_count": zero_count,
        "nonzero_shift_count": nonzero_count,
        "average_shift": average_shift,
        "avg_abs_shift": average_abs_shift,
        "median_abs_shift": median_abs_shift,
        "min_shift": min_shift,
        "max_shift": max_shift_value,
        "flights_at_negative_max_shift": at_neg_bound,
        "flights_at_positive_max_shift": at_pos_bound,
        "pct_at_shift_bounds": pct_at_shift_bounds,
        "flights_abs_shift_le_1min": int((abs_shifts <= 1).sum()) if total else 0,
        "flights_abs_shift_le_5min": small_shift_le_5,
        "pct_small_shift_le_5min": pct_small_shift_le_5min,
        "flights_abs_shift_5_15min": int(((abs_shifts > 5) & (abs_shifts <= 15)).sum()) if total else 0,
        "flights_abs_shift_15_30min": int(((abs_shifts > 15) & (abs_shifts <= 30)).sum()) if total else 0,
        "flights_abs_shift_gt_30min": int((abs_shifts > 30).sum()) if total else 0,
    }

    lines = [
        "SHIFT DISTRIBUTION ANALYSIS",
        "---------------------------",
        f"total adjustable flights: {summary['total_adjustable_flights']}",
        f"changed flights count: {summary['changed_flights_count']}",
        f"changed flight pct: {summary['changed_flight_pct']:.6f}%",
        f"zero shift count: {summary['zero_shift_count']}",
        f"nonzero shift count: {summary['nonzero_shift_count']}",
        f"average shift: {summary['average_shift']}",
        f"average absolute shift: {summary['avg_abs_shift']}",
        f"median absolute shift: {summary['median_abs_shift']}",
        f"min shift: {summary['min_shift']}",
        f"max shift: {summary['max_shift']}",
        f"number of flights at -max_shift: {summary['flights_at_negative_max_shift']}",
        f"number of flights at +max_shift: {summary['flights_at_positive_max_shift']}",
        f"number of flights with abs shift <= 1 minute: {summary['flights_abs_shift_le_1min']}",
        f"number of flights with abs shift <= 5 minutes: {summary['flights_abs_shift_le_5min']}",
        f"number of flights with abs shift between 5 and 15 minutes: {summary['flights_abs_shift_5_15min']}",
        f"number of flights with abs shift between 15 and 30 minutes: {summary['flights_abs_shift_15_30min']}",
        f"number of flights with abs shift > 30 minutes: {summary['flights_abs_shift_gt_30min']}",
        f"pct at shift bounds: {summary['pct_at_shift_bounds']:.6f}%",
        f"pct small shift <= 5 min: {summary['pct_small_shift_le_5min']:.6f}%",
    ]
    warnings = []
    if total and changed_pct == 100.0 and median_abs_shift <= 5:
        warnings.append("WARNING: changed_flight_pct is 100% but median_abs_shift is very low; changes may be tiny.")
    if total and pct_at_shift_bounds >= 50.0:
        warnings.append(
            "WARNING: Many flights are pushed to max shift bounds. Consider adding schedule disruption penalty or stronger operational constraints."
        )
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(warnings)

    (run_dir / "shift_distribution_analysis.txt").write_text("\n".join(lines), encoding="utf-8")
    return summary


def compare_connection_sets(run_dir: Path, baseline_selected: pd.DataFrame) -> dict[str, Any]:
    """Compare active connection identities between baseline and MVP."""

    current_selected = pd.read_excel(run_dir / "selected_connections.xlsx", engine="openpyxl")
    baseline_set = _connection_identity_set(baseline_selected)
    current_set = _connection_identity_set(current_selected)
    common = baseline_set & current_set
    new = current_set - baseline_set
    deactivated = baseline_set - current_set
    union = baseline_set | current_set
    jaccard = len(common) / len(union) if union else 1.0
    diff = pd.DataFrame(
        [{"status": "common", **_identity_to_row(x)} for x in sorted(common)]
        + [{"status": "newly_activated", **_identity_to_row(x)} for x in sorted(new)]
        + [{"status": "deactivated", **_identity_to_row(x)} for x in sorted(deactivated)]
    )
    diff.to_excel(run_dir / "connection_set_diff.xlsx", index=False)
    summary = {
        "baseline_active_connection_count": len(baseline_set),
        "mvp_active_connection_count": len(current_set),
        "common_active_connections": len(common),
        "newly_activated_connections": len(new),
        "deactivated_connections": len(deactivated),
        "connection_set_jaccard_similarity": float(jaccard),
    }
    lines = [
        "CONNECTION CHANGE ANALYSIS",
        "--------------------------",
        f"baseline active connection count: {summary['baseline_active_connection_count']}",
        f"MVP active connection count: {summary['mvp_active_connection_count']}",
        f"common active connections: {summary['common_active_connections']}",
        f"newly activated connections: {summary['newly_activated_connections']}",
        f"deactivated connections: {summary['deactivated_connections']}",
        f"connection set Jaccard similarity: {summary['connection_set_jaccard_similarity']:.6f}",
    ]
    if jaccard == 1.0:
        lines.append("Interpretation: MVP active connection set is identical to baseline.")
    if not new and not deactivated:
        lines.append("Interpretation: MVP did not change the active connection structure.")
    (run_dir / "connection_change_analysis.txt").write_text("\n".join(lines), encoding="utf-8")
    return summary


def compare_competitor_results(run_dir: Path, baseline_beaten: pd.DataFrame) -> dict[str, Any]:
    """Compare competitor beating status between baseline and MVP."""

    current = pd.read_excel(run_dir / "beaten_competitors.xlsx", engine="openpyxl")
    baseline_set = _beaten_identity_set(baseline_beaten)
    current_set = _beaten_identity_set(current)
    newly = current_set - baseline_set
    no_longer = baseline_set - current_set
    unchanged = baseline_set & current_set
    total_baseline = len(baseline_set)
    total_current = len(current_set)
    denominator = total_baseline if total_baseline else 1
    improvement_ratio = (total_current - total_baseline) / denominator
    diff = pd.DataFrame(
        [{"status": "unchanged_beaten", **_competitor_identity_to_row(x)} for x in sorted(unchanged)]
        + [{"status": "newly_beaten", **_competitor_identity_to_row(x)} for x in sorted(newly)]
        + [{"status": "no_longer_beaten", **_competitor_identity_to_row(x)} for x in sorted(no_longer)]
    )
    diff.to_excel(run_dir / "competitor_diff.xlsx", index=False)
    summary = {
        "baseline_beaten_competitors": total_baseline,
        "mvp_beaten_competitors": total_current,
        "newly_beaten_competitors": len(newly),
        "no_longer_beaten_competitors": len(no_longer),
        "unchanged_beaten_competitors": len(unchanged),
        "beat_improvement_ratio": float(improvement_ratio),
    }
    lines = [
        "COMPETITOR CHANGE ANALYSIS",
        "--------------------------",
        f"baseline beaten competitors: {summary['baseline_beaten_competitors']}",
        f"MVP beaten competitors: {summary['mvp_beaten_competitors']}",
        f"newly beaten competitors: {summary['newly_beaten_competitors']}",
        f"no longer beaten competitors: {summary['no_longer_beaten_competitors']}",
        f"unchanged beaten competitors: {summary['unchanged_beaten_competitors']}",
        f"beat improvement ratio: {summary['beat_improvement_ratio']:.6f}",
    ]
    (run_dir / "competitor_change_analysis.txt").write_text("\n".join(lines), encoding="utf-8")
    return summary


def compare_od_metrics(run_dir: Path, baseline_od_metrics: pd.DataFrame) -> dict[str, Any]:
    """Write O-D/day comparison workbooks and return aggregate counts."""

    current = pd.read_excel(run_dir / "od_metrics.xlsx", engine="openpyxl")
    base = baseline_od_metrics.copy()
    base = base.rename(
        columns={
            "active_connections": "baseline_active_connections",
            "new_best_journey_time": "baseline_best_journey_time",
            "beaten_competitors": "baseline_beaten_competitors",
            "new_rank": "baseline_rank",
            "total_weighted_reward": "baseline_total_reward",
        }
    )
    current = current.rename(
        columns={
            "active_connections": "mvp_active_connections",
            "new_best_journey_time": "mvp_best_journey_time",
            "beaten_competitors": "mvp_beaten_competitors",
            "new_rank": "mvp_rank",
            "total_weighted_reward": "mvp_total_reward",
        }
    )
    merged = base[
        [
            "od",
            "day",
            "importance_factor",
            "baseline_active_connections",
            "baseline_best_journey_time",
            "baseline_beaten_competitors",
            "baseline_rank",
            "baseline_total_reward",
        ]
    ].merge(
        current[
            [
                "od",
                "day",
                "mvp_active_connections",
                "mvp_best_journey_time",
                "mvp_beaten_competitors",
                "mvp_rank",
                "mvp_total_reward",
            ]
        ],
        on=["od", "day"],
        how="outer",
    )
    merged["importance_factor"] = merged["importance_factor"].fillna(0)
    merged["connection_delta"] = merged["mvp_active_connections"].fillna(0) - merged["baseline_active_connections"].fillna(0)
    merged["journey_time_delta"] = merged["mvp_best_journey_time"] - merged["baseline_best_journey_time"]
    merged["beaten_delta"] = merged["mvp_beaten_competitors"].fillna(0) - merged["baseline_beaten_competitors"].fillna(0)
    merged["rank_delta"] = merged["baseline_rank"].fillna(0) - merged["mvp_rank"].fillna(0)
    merged["reward_delta"] = merged["mvp_total_reward"].fillna(0) - merged["baseline_total_reward"].fillna(0)
    merged["status"] = np.where(
        merged["reward_delta"] > 1e-9,
        "improved",
        np.where(merged["reward_delta"] < -1e-9, "worsened", "unchanged"),
    )
    ordered = [
        "od",
        "day",
        "importance_factor",
        "baseline_active_connections",
        "mvp_active_connections",
        "connection_delta",
        "baseline_best_journey_time",
        "mvp_best_journey_time",
        "journey_time_delta",
        "baseline_beaten_competitors",
        "mvp_beaten_competitors",
        "beaten_delta",
        "baseline_rank",
        "mvp_rank",
        "rank_delta",
        "baseline_total_reward",
        "mvp_total_reward",
        "reward_delta",
        "status",
    ]
    merged = merged[ordered]
    merged.to_excel(run_dir / "od_level_comparison.xlsx", index=False)
    merged.sort_values("reward_delta", ascending=False).head(25).to_excel(run_dir / "top_improved_od.xlsx", index=False)
    merged.sort_values("reward_delta", ascending=True).head(25).to_excel(run_dir / "top_worsened_od.xlsx", index=False)
    merged.sort_values("importance_factor", ascending=False).head(25).to_excel(run_dir / "top_important_od_comparison.xlsx", index=False)
    return {
        "improved_od_count": int((merged["status"] == "improved").sum()),
        "worsened_od_count": int((merged["status"] == "worsened").sum()),
        "unchanged_od_count": int((merged["status"] == "unchanged").sum()),
    }


def write_meaningfulness_report(
    run_dir: Path,
    comparison: dict[str, Any],
    schedule: dict[str, Any],
    connections: dict[str, Any],
    competitors: dict[str, Any],
    od_summary: dict[str, Any],
) -> dict[str, Any]:
    """Write automatic MVP meaningfulness decision report."""

    weak_conditions = [
        comparison["percentage_improvement"] <= 0.1,
        schedule["changed_flight_pct"] == 0,
        connections["newly_activated_connections"] == 0,
        connections["deactivated_connections"] == 0,
        competitors["newly_beaten_competitors"] == 0,
        od_summary["improved_od_count"] == 0,
        abs(connections["connection_set_jaccard_similarity"] - 1.0) <= 1e-12,
    ]
    weak_count = sum(bool(x) for x in weak_conditions)
    if weak_count >= 5:
        result = "FORMALITY-LIKE"
        interpretation = (
            "The MVP is feasible, but most improvement signals are weak in this test. "
            "Schedule shifts, connection-set changes, competitor wins, and objective metrics together look close to a formality run."
        )
    elif weak_count >= 3:
        result = "WEAK"
        interpretation = (
            "The MVP is feasible and produces some changes, but the improvement signal is limited. "
            "It should be retested with a larger subset, different max_shift values, or stronger objective components."
        )
    else:
        result = "MEANINGFUL"
        interpretation = (
            "The MVP produced measurable changes versus baseline in schedule timing, connection structure, or O-D rewards. "
            "On this small test, it shows a meaningful optimization effect."
        )
    lines = [
        "MVP MEANINGFULNESS RESULT",
        "-------------------------",
        f"Result: {result}",
        "",
        "Reasons:",
        f"- Objective improvement: {comparison['percentage_improvement']:.6f}%",
        f"- Changed flight percentage: {schedule['changed_flight_pct']:.6f}%",
        f"- New active connections: {connections['newly_activated_connections']}",
        f"- Deactivated connections: {connections['deactivated_connections']}",
        f"- Newly beaten competitors: {competitors['newly_beaten_competitors']}",
        f"- Improved O-D count: {od_summary['improved_od_count']}",
        f"- Jaccard similarity: {connections['connection_set_jaccard_similarity']:.6f}",
        f"- Weak/formality-like condition count: {weak_count}/7",
        "",
        "Interpretation:",
        interpretation,
    ]
    (run_dir / "mvp_meaningfulness_report.txt").write_text("\n".join(lines), encoding="utf-8")
    return {"meaningfulness_result": result, "weak_condition_count": weak_count}


def analyze_mvp_run(
    run_dir: Path,
    baseline_result: BaselineResult,
    current_metrics: dict[str, Any],
    max_shift: int | float | None = None,
) -> dict[str, Any]:
    """Run all MVP analysis outputs for a completed run."""

    comparison = compare_to_baseline(run_dir, baseline_result, current_metrics)
    schedule = analyze_schedule_changes(run_dir)
    shift_distribution = analyze_shift_distribution(run_dir, max_shift)
    connections = compare_connection_sets(run_dir, baseline_result.selected_connections)
    competitors = compare_competitor_results(run_dir, baseline_result.beaten_competitors)
    od_summary = compare_od_metrics(run_dir, baseline_result.od_metrics)
    meaning = write_meaningfulness_report(run_dir, comparison, schedule, connections, competitors, od_summary)
    return {**comparison, **schedule, **shift_distribution, **connections, **competitors, **od_summary, **meaning}


def write_sensitivity_summary(sensitivity_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write sensitivity summary and interpretation."""

    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(sensitivity_dir / "sensitivity_summary.csv", index=False)
    lines = ["SENSITIVITY INTERPRETATION", "--------------------------"]
    if df.empty:
        lines.append("No sensitivity runs were completed.")
    else:
        lines.append(f"Runs completed: {len(df)}")
        lines.append(f"Max objective: {df['objective'].max()}")
        lines.append(f"Min objective: {df['objective'].min()}")
        objective_varies = df["objective"].nunique(dropna=False) > 1
        changed_varies = df["changed_flight_pct"].nunique(dropna=False) > 1
        new_conn_varies = df["newly_activated_connections"].nunique(dropna=False) > 1
        if objective_varies or changed_varies or new_conn_varies:
            lines.append(
                "Interpretation: at least one main metric changes as max_shift changes; the MVP responds to time-shift flexibility."
            )
        else:
            lines.append(
                "Interpretation: the main metrics did not change as max_shift increased; on this subset, the model may not be showing timing optimization impact."
            )
    (sensitivity_dir / "sensitivity_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def write_scaling_summary(scaling_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write scaling summary CSV and Turkish interpretation."""

    scaling_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    ordered_columns = [
        "run_id",
        "max_od",
        "days",
        "max_shift",
        "baseline_objective",
        "objective",
        "absolute_improvement",
        "improvement_pct",
        "active_connections",
        "changed_flight_pct",
        "avg_abs_shift",
        "median_abs_shift",
        "pct_at_shift_bounds",
        "pct_small_shift_le_5min",
        "newly_activated_connections",
        "beaten_competitors",
        "newly_beaten_competitors",
        "improved_od_count",
        "worsened_od_count",
        "solver_status",
        "gap",
        "runtime_sec",
        "feasible",
        "meaningfulness_result",
    ]
    if not df.empty:
        df = df[[col for col in ordered_columns if col in df.columns]]
    df.to_csv(scaling_dir / "scaling_summary.csv", index=False)

    lines = ["SCALING TEST YORUMU", "--------------------"]
    if df.empty:
        lines.append("Hiç scaling deneyi tamamlanmadı.")
        (scaling_dir / "scaling_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return

    feasible_count = int(df["feasible"].fillna(False).sum()) if "feasible" in df else 0
    total_count = int(len(df))
    objective = pd.to_numeric(df.get("objective", pd.Series(dtype=float)), errors="coerce")
    improvement = pd.to_numeric(df.get("improvement_pct", pd.Series(dtype=float)), errors="coerce")
    runtime = pd.to_numeric(df.get("runtime_sec", pd.Series(dtype=float)), errors="coerce")
    gap = pd.to_numeric(df.get("gap", pd.Series(dtype=float)), errors="coerce")
    changed = pd.to_numeric(df.get("changed_flight_pct", pd.Series(dtype=float)), errors="coerce")
    new_connections = pd.to_numeric(df.get("newly_activated_connections", pd.Series(dtype=float)), errors="coerce")
    avg_abs_shift = pd.to_numeric(df.get("avg_abs_shift", pd.Series(dtype=float)), errors="coerce")
    median_abs_shift = pd.to_numeric(df.get("median_abs_shift", pd.Series(dtype=float)), errors="coerce")
    pct_at_bounds = pd.to_numeric(df.get("pct_at_shift_bounds", pd.Series(dtype=float)), errors="coerce")
    pct_small_shift = pd.to_numeric(df.get("pct_small_shift_le_5min", pd.Series(dtype=float)), errors="coerce")

    best_idx = improvement.idxmax() if improvement.notna().any() else objective.idxmax()
    best = df.loc[best_idx]
    largest_idx = pd.to_numeric(df.get("max_od", pd.Series(dtype=float)), errors="coerce").idxmax()
    largest = df.loc[largest_idx]
    meaningful_count = int((df.get("meaningfulness_result", pd.Series(dtype=str)) == "MEANINGFUL").sum())

    lines.extend(
        [
            f"Tamamlanan deney sayısı: {total_count}",
            f"Feasible deney sayısı: {feasible_count}/{total_count}",
            f"En yüksek objective: {float(objective.max()):.6f}",
            f"En yüksek improvement pct: {float(improvement.max()):.6f}%",
            f"En uzun runtime: {float(runtime.max()):.3f} sn",
            f"Meaningful deney sayısı: {meaningful_count}/{total_count}",
            f"Ortalama absolute shift aralığı: {float(avg_abs_shift.min()):.3f} - {float(avg_abs_shift.max()):.3f} dk",
            f"Median absolute shift aralığı: {float(median_abs_shift.min()):.3f} - {float(median_abs_shift.max()):.3f} dk",
            f"Shift sınırında uçuş oranı aralığı: {float(pct_at_bounds.min()):.3f}% - {float(pct_at_bounds.max()):.3f}%",
            f"<=5 dk küçük shift oranı aralığı: {float(pct_small_shift.min()):.3f}% - {float(pct_small_shift.max()):.3f}%",
            "",
            "En iyi iyileşme veren deney:",
            (
                f"- {best['run_id']} | max_od={best.get('max_od', '')}, days={best.get('days', '')}, "
                f"improvement={float(best.get('improvement_pct', 0.0)):.6f}%, "
                f"runtime={float(best.get('runtime_sec', 0.0)):.3f} sn"
            ),
            "",
            "En büyük O-D deneyinin durumu:",
            (
                f"- {largest['run_id']} | max_od={largest.get('max_od', '')}, days={largest.get('days', '')}, "
                f"feasible={largest.get('feasible', '')}, "
                f"improvement={float(largest.get('improvement_pct', 0.0)):.6f}%, "
                f"runtime={float(largest.get('runtime_sec', 0.0)):.3f} sn"
            ),
            "",
            "Yorum:",
        ]
    )

    if feasible_count == total_count:
        lines.append("- Feasibility bozulmadı; tüm scaling deneyleri checker'dan geçti.")
    else:
        lines.append("- Feasibility bazı deneylerde bozuldu; ilgili run klasörlerindeki feasibility_report.txt incelenmeli.")

    if improvement.notna().any() and float(improvement.min()) > 0:
        lines.append("- Objective iyileşmesi tüm deneylerde pozitif kaldı.")
    elif improvement.notna().any() and float(improvement.max()) > 0:
        lines.append("- Objective iyileşmesi bazı deneylerde korunuyor, ancak tüm deneylerde pozitif değil.")
    else:
        lines.append("- Objective iyileşmesi gözlenmedi; model bu ayarda baseline'a yakın kalıyor.")

    if runtime.notna().sum() >= 2 and runtime.max() > runtime.min() * 3:
        lines.append("- Runtime ölçek büyüdükçe belirgin artıyor; 50/100 O-D sonrası performans sıkılaştırma gerektirebilir.")
    else:
        lines.append("- Runtime bu deney setinde kontrollü görünüyor.")

    if gap.notna().any() and float(gap.fillna(0).max()) > 0.05:
        lines.append("- Gap tarafında kötüleşme işareti var; time_limit veya formülasyon sıkılığı kontrol edilmeli.")
    else:
        lines.append("- Gap alanında belirgin kötüleşme görünmüyor veya solver gap raporlamadı.")

    if meaningful_count >= max(1, total_count // 2):
        lines.append("- 50/100 O-D seviyesinde model hâlâ anlamlı sinyal üretiyor; full model geliştirme için yeterli zemin var.")
    elif changed.notna().any() and (changed > 0).any() and new_connections.notna().any() and (new_connections > 0).any():
        lines.append("- Model zaman ve bağlantı setini değiştirebiliyor, fakat meaningful sinyal sınırlı; objective/ranking bağı güçlendirilmeli.")
    else:
        lines.append("- Model büyüyünce anlamlılık sinyali zayıf; aday bağlantı üretimi ve ranking reward modellemesi öncelikli olmalı.")

    lines.append("")
    lines.append("Shift davranışı:")
    if median_abs_shift.notna().any() and float(median_abs_shift.max()) <= 5:
        lines.append("- Median shift genelde çok küçük; changed_flight_pct yüksek olsa bile değişimler operasyonel olarak zayıf olabilir.")
    elif median_abs_shift.notna().any() and float(median_abs_shift.min()) > 15:
        lines.append("- Median shift 15 dakikanın üzerinde; model gerçek ve görünür zaman kaydırmaları yapıyor.")
    else:
        lines.append("- Shiftler küçük mikro düzeltmelerden ibaret görünmüyor; bazı deneylerde orta büyüklükte hareket var.")

    if pct_small_shift.notna().any() and float(pct_small_shift.max()) >= 80:
        lines.append("- Uçuşların büyük kısmı 5 dakika veya altında kalıyor; zaman kaydırma etkisi sınırlı olabilir.")
    else:
        lines.append("- <=5 dakika küçük shift oranı baskın değil; 100% changed_flight_pct metriği tamamen küçük oynama kaynaklı değil.")

    if pct_at_bounds.notna().any() and float(pct_at_bounds.max()) >= 50:
        lines.append(
            "- Bir veya daha fazla deneyde uçuşların büyük kısmı max shift sınırına dayanıyor. Schedule disruption penalty veya daha güçlü operasyonel kısıtlar eklenmeli."
        )
    elif pct_at_bounds.notna().any() and float(pct_at_bounds.max()) > 0:
        lines.append("- Bazı uçuşlar shift sınırına dayanıyor, ancak çoğunluk sınırda değil.")
    else:
        lines.append("- Uçuşlar sınır değerlere yığılmıyor.")

    (scaling_dir / "scaling_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def write_rotation_interpretation(
    run_dir: Path,
    rotation_summary: dict[str, Any],
    reference_mvp: dict[str, Any] | None,
    shift_summary: dict[str, Any],
) -> None:
    """Write Turkish rotation-vs-MVP interpretation for a rotation run."""

    lines = ["ROTATION MODEL YORUMU", "---------------------"]
    if reference_mvp:
        objective_drop = float(reference_mvp["objective"]) - float(rotation_summary["objective"])
        runtime_delta = float(rotation_summary["runtime_sec"]) - float(reference_mvp["runtime_sec"])
        lines.extend(
            [
                f"MVP objective: {reference_mvp['objective']}",
                f"Rotation objective: {rotation_summary['objective']}",
                f"MVP improvement pct: {reference_mvp['improvement_pct']:.6f}%",
                f"Rotation improvement pct: {rotation_summary['improvement_pct']:.6f}%",
                f"Rotation eklenince objective dususu: {objective_drop}",
                f"MVP runtime: {reference_mvp['runtime_sec']:.3f} sn",
                f"Rotation runtime: {rotation_summary['runtime_sec']:.3f} sn",
                f"Runtime artisi: {runtime_delta:.3f} sn",
                f"MVP avg abs shift: {reference_mvp.get('avg_abs_shift', 0.0)}",
                f"Rotation avg abs shift: {shift_summary.get('avg_abs_shift', 0.0)}",
                f"MVP median abs shift: {reference_mvp.get('median_abs_shift', 0.0)}",
                f"Rotation median abs shift: {shift_summary.get('median_abs_shift', 0.0)}",
            ]
        )
    else:
        lines.append("MVP referans kosusu hesaplanamadi; yalnizca rotation run metrikleri yorumlandi.")

    lines.extend(
        [
            f"Feasible: {rotation_summary['feasible']}",
            f"Rotation candidates total: {rotation_summary['rotation_candidates_total']}",
            f"Rotation constraints active: {rotation_summary['rotation_constraints_active']}",
            f"Rotation constraints skipped: {rotation_summary['rotation_constraints_skipped']}",
            f"Rotation block time O&D-derived: {rotation_summary.get('rotation_block_time_od_derived', 0)}",
            f"Rotation block time schedule fallback: {rotation_summary.get('rotation_block_time_schedule_fallback', 0)}",
            f"Rotation violations: {rotation_summary['rotation_violations']}",
            f"Min rotation slack: {rotation_summary['min_rotation_slack']}",
            f"Avg rotation slack: {rotation_summary['avg_rotation_slack']}",
            "",
            "Yorum:",
        ]
    )
    if rotation_summary["rotation_constraints_active"] == 0:
        lines.append("- Flight Pairs ile modeldeki O&D flight eventleri yeterince eslesmedi; aktif rotation kisiti olusmadi.")
    else:
        lines.append("- Flight Pairs eslesmelerinden aktif rotation kisitlari uretildi.")
    if rotation_summary.get("rotation_block_time_od_derived", 0):
        lines.append("- Bazi rotation block time degerleri O&D baglanti tablosundan turetildi.")
    if rotation_summary.get("rotation_block_time_schedule_fallback", 0):
        lines.append("- Bazi aktif rotationlarda O&D flight/route eslesmesi bulunamadi; schedule elapsed fallback kullanildi.")
    if rotation_summary["rotation_violations"] == 0:
        lines.append("- Bagimsiz checker rotation violation bulmadi; feasibility korundu.")
    else:
        lines.append("- Bagimsiz checker rotation violation buldu; bu run feasible kabul edilmemeli.")
    if reference_mvp:
        if objective_drop > 1e-6:
            lines.append("- Rotation kisiti objective'i dusurdu; bu beklenen bir operasyonel gercekcilik maliyeti.")
        elif objective_drop < -1e-6:
            lines.append("- Rotation kisiti objective'i dusurmedi; solver farkli ve daha iyi bir cozum bulmus olabilir.")
        else:
            lines.append("- Rotation kisiti objective uzerinde belirgin fark yaratmadi.")
        if shift_summary.get("avg_abs_shift", 0.0) < reference_mvp.get("avg_abs_shift", 0.0):
            lines.append("- Shift dagilimi MVP'ye gore daha makul hale geldi.")
        else:
            lines.append("- Shift dagilimi MVP'ye gore belirgin yumusamadi; schedule disruption penalty hala gerekli olabilir.")
    if rotation_summary["rotation_constraints_skipped"] > rotation_summary["rotation_constraints_active"]:
        lines.append("- Skipped rotation sayisi aktif kisitlardan fazla; Flight Pairs ile model event kapsam eslesmesi iyilestirilmeli.")

    (run_dir / "rotation_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def write_capacity_interpretation(
    run_dir: Path,
    capacity_summary: dict[str, Any],
    reference_rotation: dict[str, Any] | None,
    shift_summary: dict[str, Any],
) -> None:
    """Write Turkish capacity-vs-rotation interpretation for a capacity run."""

    lines = ["CAPACITY MODEL YORUMU", "----------------------"]
    if reference_rotation:
        objective_drop = float(reference_rotation["objective"]) - float(capacity_summary["objective"])
        runtime_delta = float(capacity_summary["runtime_sec"]) - float(reference_rotation["runtime_sec"])
        lines.extend(
            [
                f"Rotation objective: {reference_rotation['objective']}",
                f"Capacity objective: {capacity_summary['objective']}",
                f"Rotation improvement pct: {reference_rotation['improvement_pct']:.6f}%",
                f"Capacity improvement pct: {capacity_summary['improvement_pct']:.6f}%",
                f"Capacity eklenince objective dususu: {objective_drop}",
                f"Rotation runtime: {reference_rotation['runtime_sec']:.3f} sn",
                f"Capacity runtime: {capacity_summary['runtime_sec']:.3f} sn",
                f"Runtime artisi: {runtime_delta:.3f} sn",
                f"Rotation avg abs shift: {reference_rotation.get('avg_abs_shift', 0.0)}",
                f"Capacity avg abs shift: {shift_summary.get('avg_abs_shift', 0.0)}",
                f"Rotation median abs shift: {reference_rotation.get('median_abs_shift', 0.0)}",
                f"Capacity median abs shift: {shift_summary.get('median_abs_shift', 0.0)}",
            ]
        )
    else:
        lines.append("Rotation referans kosusu hesaplanamadi; yalnizca capacity run metrikleri yorumlandi.")

    lines.extend(
        [
            f"Feasible: {capacity_summary['feasible']}",
            f"Baseline ustu improvement pct: {capacity_summary['improvement_pct']:.6f}%",
            f"Hub capacity violations: {capacity_summary['hub_capacity_violations']}",
            f"Max arrivals in bucket: {capacity_summary['max_arrivals_in_bucket']}",
            f"Max departures in bucket: {capacity_summary['max_departures_in_bucket']}",
            f"Avg arrivals per used bucket: {capacity_summary['avg_arrivals_per_used_bucket']}",
            f"Avg departures per used bucket: {capacity_summary['avg_departures_per_used_bucket']}",
            "",
            "Yorum:",
        ]
    )
    if capacity_summary["hub_capacity_violations"] == 0:
        lines.append("- Bagimsiz checker hub capacity violation bulmadi; feasibility bu acidan korundu.")
    else:
        lines.append("- Hub capacity violation var; checker sonucuna gore run feasible kabul edilmemeli.")
    if reference_rotation:
        if objective_drop > 1e-6:
            lines.append("- Capacity kisiti objective'i dusurdu; model uzerinde operasyonel baski olusturuyor.")
        else:
            lines.append("- Capacity kisiti objective uzerinde belirgin fark yaratmadi; mevcut kapasite limitleri bu subsette gevsek olabilir.")
        if shift_summary.get("avg_abs_shift", 0.0) < reference_rotation.get("avg_abs_shift", 0.0):
            lines.append("- Shift dagilimi rotation referansina gore daha makul hale geldi.")
        else:
            lines.append("- Shift dagilimi rotation referansina gore belirgin yumusamadi.")
    (run_dir / "capacity_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def write_regularity_interpretation(
    run_dir: Path,
    regularity_summary: dict[str, Any],
    reference_capacity: dict[str, Any] | None,
    shift_summary: dict[str, Any],
) -> None:
    """Write Turkish regularity-vs-capacity interpretation for a regularity run."""

    lines = ["REGULARITY MODEL YORUMU", "------------------------"]
    if reference_capacity:
        objective_drop = float(reference_capacity["objective"]) - float(regularity_summary["objective"])
        runtime_delta = float(regularity_summary["runtime_sec"]) - float(reference_capacity["runtime_sec"])
        lines.extend(
            [
                f"Capacity objective: {reference_capacity['objective']}",
                f"Regularity objective: {regularity_summary['objective']}",
                f"Capacity improvement pct: {reference_capacity['improvement_pct']:.6f}%",
                f"Regularity improvement pct: {regularity_summary['improvement_pct']:.6f}%",
                f"Regularity eklenince objective dususu: {objective_drop}",
                f"Capacity runtime: {reference_capacity['runtime_sec']:.3f} sn",
                f"Regularity runtime: {regularity_summary['runtime_sec']:.3f} sn",
                f"Runtime artisi: {runtime_delta:.3f} sn",
                f"Capacity avg abs shift: {reference_capacity.get('avg_abs_shift', 0.0)}",
                f"Regularity avg abs shift: {shift_summary.get('avg_abs_shift', 0.0)}",
                f"Capacity median abs shift: {reference_capacity.get('median_abs_shift', 0.0)}",
                f"Regularity median abs shift: {shift_summary.get('median_abs_shift', 0.0)}",
            ]
        )
    else:
        lines.append("Capacity referans kosusu hesaplanamadi; yalnizca regularity run metrikleri yorumlandi.")
    lines.extend(
        [
            f"Feasible: {regularity_summary['feasible']}",
            f"Regularity XDEV: {regularity_summary['regularity_xdev_minutes']}",
            f"Regularity groups total: {regularity_summary['regularity_groups_total']}",
            f"Regularity constraints active: {regularity_summary['regularity_constraints_active']}",
            f"Regularity groups skipped: {regularity_summary['regularity_groups_skipped']}",
            f"Regularity violations: {regularity_summary['regularity_violations']}",
            f"Max regularity spread: {regularity_summary['max_regularity_spread']}",
            f"Avg regularity spread: {regularity_summary['avg_regularity_spread']}",
            "",
            "Yorum:",
        ]
    )
    if regularity_summary["regularity_violations"] == 0:
        lines.append("- Bagimsiz checker regularity violation bulmadi; feasibility bu acidan korundu.")
    else:
        lines.append("- Regularity violation var; checker sonucuna gore run feasible kabul edilmemeli.")
    if reference_capacity:
        if objective_drop > 1e-6:
            lines.append("- Regularity kisiti objective'i dusurdu; model uzerinde operasyonel baski olusturuyor.")
        else:
            lines.append("- Regularity kisiti objective uzerinde belirgin fark yaratmadi.")
        if shift_summary.get("avg_abs_shift", 0.0) < reference_capacity.get("avg_abs_shift", 0.0):
            lines.append("- Shift dagilimi capacity referansina gore daha makul hale geldi.")
        else:
            lines.append("- Shift dagilimi capacity referansina gore belirgin yumusamadi.")
    if regularity_summary["regularity_constraints_active"] == 0:
        lines.append("- Bu subsette coklu gun regularity grubu bulunmadi; grouping kapsamı iyilestirilmeli veya daha fazla gun denenmeli.")
    (run_dir / "regularity_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def _connection_identity_set(df: pd.DataFrame) -> set[tuple[str, int, str, str]]:
    if df.empty:
        return set()
    active = df[df.get("active", True) == True].copy()
    identities = set()
    for row in active.itertuples(index=False):
        identities.add((str(row.od), int(row.day), str(row.inbound_flight), str(row.outbound_flight)))
    return identities


def _identity_to_row(identity: tuple[str, int, str, str]) -> dict[str, Any]:
    od, day, inbound, outbound = identity
    return {"od": od, "day": day, "inbound_flight": inbound, "outbound_flight": outbound}


def _beaten_identity_set(df: pd.DataFrame) -> set[tuple[str, int, str]]:
    if df.empty or "beaten" not in df.columns:
        return set()
    active = df[df["beaten"] == True].copy()
    return {(str(row.od), int(row.day), str(row.competitor_id)) for row in active.itertuples(index=False)}


def _competitor_identity_to_row(identity: tuple[str, int, str]) -> dict[str, Any]:
    od, day, competitor_id = identity
    return {"od": od, "day": day, "competitor_id": competitor_id}


def _od_reward_status_counts(baseline_od: pd.DataFrame, current_od: pd.DataFrame) -> dict[str, int]:
    base = baseline_od[["od", "day", "total_weighted_reward"]].rename(
        columns={"total_weighted_reward": "baseline_total_reward"}
    )
    current = current_od[["od", "day", "total_weighted_reward"]].rename(
        columns={"total_weighted_reward": "mvp_total_reward"}
    )
    merged = base.merge(current, on=["od", "day"], how="outer")
    delta = merged["mvp_total_reward"].fillna(0) - merged["baseline_total_reward"].fillna(0)
    return {
        "improved_od_count": int((delta > 1e-9).sum()),
        "worsened_od_count": int((delta < -1e-9).sum()),
        "unchanged_od_count": int((delta.abs() <= 1e-9).sum()),
    }


def _shift_bucket(abs_shift: float) -> str:
    if abs_shift == 0:
        return "zero"
    if abs_shift <= 1:
        return "0-1"
    if abs_shift <= 5:
        return "1-5"
    if abs_shift <= 15:
        return "5-15"
    if abs_shift <= 30:
        return "15-30"
    return "30+"
