"""CSV experiment tracking."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


EXPERIMENT_COLUMNS = [
    "run_id",
    "timestamp",
    "model_level",
    "solver",
    "time_limit",
    "num_od",
    "num_flights",
    "num_candidate_connections",
    "feasible",
    "objective",
    "baseline_objective",
    "shift_penalty_mode",
    "shift_penalty_weight",
    "shift_free_threshold",
    "mandatory_connection_selection",
    "ranking_objective_mode",
    "reported_lookup_ranking_reward",
    "model_proxy_ranking_reward",
    "ranking_reward_difference_lookup_vs_proxy",
    "ranking_objective_recalculation_difference",
    "total_candidate_connections",
    "connections_feasible_by_final_schedule",
    "feasible_but_inactive_connection_count",
    "infeasible_but_active_connection_count",
    "objective_before_penalty",
    "objective_after_penalty",
    "total_shift_penalty",
    "total_abs_shift",
    "avg_abs_shift",
    "median_abs_shift",
    "max_abs_shift",
    "pct_shift_le_5",
    "pct_shift_5_15",
    "pct_shift_15_30",
    "pct_at_shift_bounds",
    "improvement_pct",
    "connection_reward",
    "ranking_reward",
    "active_connections",
    "beaten_competitors",
    "improved_od_count",
    "worsened_od_count",
    "runtime_sec",
    "gap",
    "rotation_candidates_total",
    "rotation_constraints_active",
    "rotation_constraints_skipped",
    "rotation_coverage_pct",
    "rotation_skip_subset_excluded",
    "rotation_skip_format_mismatch",
    "rotation_skip_route_mismatch",
    "rotation_skip_missing_block_time",
    "rotation_block_exact_match_count",
    "rotation_block_fallback_count",
    "rotation_violations",
    "hub_capacity_violations",
    "max_arrivals_in_bucket",
    "max_departures_in_bucket",
    "regularity_constraints_active",
    "regularity_violations",
    "max_regularity_spread",
    "avg_regularity_spread",
    "directional_pairs_active",
    "directional_e1_mode",
    "directional_e1_formula",
    "directional_e1_abs_tolerance",
    "directional_e1_violations",
    "directional_e2_violations",
    "directional_e1_hard_violations",
    "directional_e1_soft_violations",
    "directional_e1_soft_violation_amount_total",
    "directional_e2_hard_violations",
    "directional_e2_soft_violations",
    "directional_e2_soft_violation_amount_total",
    "directional_total_violations",
    "max_connection_count_diff",
    "max_journey_time_diff",
    "notes",
]


def append_experiment(output_dir: Path, row: dict) -> None:
    """Append one row to outputs/experiments.csv."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiments.csv"
    payload = {col: row.get(col, "") for col in EXPERIMENT_COLUMNS}
    payload["timestamp"] = payload.get("timestamp") or datetime.now().isoformat(timespec="seconds")
    df = pd.DataFrame([payload], columns=EXPERIMENT_COLUMNS)
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    for col in EXPERIMENT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[EXPERIMENT_COLUMNS]
    df.to_csv(path, index=False)
