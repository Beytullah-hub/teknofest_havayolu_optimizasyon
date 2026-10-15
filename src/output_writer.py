"""Output table construction and file writing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .baseline import BaselineResult, lookup_ranking_reward
from .config import ModelConfig
from .preprocessing import PreparedData, connection_reward


def build_outputs_from_active_connections(
    data: PreparedData,
    active_connections: pd.DataFrame,
    config: ModelConfig,
    baseline_od_metrics: pd.DataFrame | None = None,
) -> BaselineResult:
    """Build standard output tables from a selected TK connection table."""

    selected = _selected_connections_table(active_connections)
    comp = data.competitor_connections.copy()
    od_rows = []
    beaten_rows = []
    all_keys = data.od_day[["od", "origin", "destination", "day", "importance_factor", "old_best_journey_time"]].copy()
    baseline_by_key = {}
    if baseline_od_metrics is not None and not baseline_od_metrics.empty:
        baseline_by_key = {
            (row.od, int(row.day)): row
            for row in baseline_od_metrics.itertuples(index=False)
        }
    for row in all_keys.itertuples(index=False):
        active_od = selected[(selected["od"] == row.od) & (selected["day"] == row.day) & (selected["active"] == True)]
        comp_od = comp[(comp["O&D"] == row.od) & (comp["day"] == row.day)]
        active_count = len(active_od)
        best_new = float(active_od["journey_time_min"].min()) if active_count else float("inf")
        baseline_row = baseline_by_key.get((row.od, int(row.day)))
        old_best_source = baseline_row.old_best_journey_time if baseline_row is not None else row.old_best_journey_time
        old_best = None if pd.isna(old_best_source) or old_best_source == float("inf") else float(old_best_source)
        competitor_count = len(comp_od)
        beaten_count = int((comp_od["journey_time_min"] >= best_new).sum()) if active_count else 0
        if baseline_row is not None:
            initial_rank = int(baseline_row.initial_rank)
        else:
            initial_beaten = 0
            if old_best is not None:
                initial_beaten = int((comp_od["journey_time_min"] >= old_best).sum())
            initial_rank = max(1, competitor_count - initial_beaten + 1)
        new_rank = max(1, competitor_count - beaten_count + 1)
        ranking_reward = lookup_ranking_reward(data.ranking_lookup, competitor_count, initial_rank, new_rank)
        conn_reward = connection_reward(active_count, config.max_connection_slots, config.connection_reward_decay)
        weighted = float(row.importance_factor) * (conn_reward + ranking_reward)
        if old_best is None or best_new == float("inf"):
            status = "unchanged"
        elif best_new < old_best:
            status = "improved"
        elif best_new > old_best:
            status = "worsened"
        else:
            status = "unchanged"
        od_rows.append(
            {
                "od": row.od,
                "origin": row.origin,
                "destination": row.destination,
                "day": int(row.day),
                "importance_factor": row.importance_factor,
                "active_connections": active_count,
                "old_best_journey_time": old_best,
                "new_best_journey_time": None if best_new == float("inf") else best_new,
                "competitor_count": competitor_count,
                "beaten_competitors": beaten_count,
                "initial_rank": initial_rank,
                "new_rank": new_rank,
                "connection_reward": conn_reward,
                "ranking_reward": ranking_reward,
                "total_weighted_reward": weighted,
                "status_improved_worsened_unchanged": status,
            }
        )
        for comp_row in comp_od.itertuples(index=False):
            beaten_rows.append(
                {
                    "od": row.od,
                    "day": int(row.day),
                    "competitor_id": comp_row.competitor_id,
                    "competitor_carrier": comp_row.carrier_code,
                    "competitor_time_min": comp_row.journey_time_min,
                    "best_tk_time_min": None if best_new == float("inf") else best_new,
                    "beaten": bool(active_count and best_new <= comp_row.journey_time_min),
                }
            )
    od_metrics = pd.DataFrame(od_rows)
    beaten = pd.DataFrame(beaten_rows)
    summary = {
        "total_objective": float(od_metrics["total_weighted_reward"].sum()) if not od_metrics.empty else 0.0,
        "connection_reward": float((od_metrics["importance_factor"] * od_metrics["connection_reward"]).sum()) if not od_metrics.empty else 0.0,
        "ranking_reward": float((od_metrics["importance_factor"] * od_metrics["ranking_reward"]).sum()) if not od_metrics.empty else 0.0,
        "active_connections": int(selected["active"].sum()) if not selected.empty else 0,
        "beaten_competitors": int(beaten["beaten"].sum()) if not beaten.empty else 0,
        "od_day_count": int(len(od_metrics)),
    }
    return BaselineResult(od_metrics, selected, beaten, summary)


def write_run_outputs(
    run_dir: Path,
    result: BaselineResult,
    data: PreparedData,
    metrics: dict,
    solver_status: str,
    gap: float | None,
    runtime_sec: float,
    feasibility_text: str | None = None,
    rotation_report: pd.DataFrame | None = None,
    rotation_candidates: pd.DataFrame | None = None,
    hub_capacity_report: pd.DataFrame | None = None,
    regularity_report: pd.DataFrame | None = None,
    directional_balance_report: pd.DataFrame | None = None,
    optimized_schedule: pd.DataFrame | None = None,
) -> None:
    """Write all requested run artifacts."""

    print("[INFO] Writing outputs...")
    run_dir.mkdir(parents=True, exist_ok=True)
    if optimized_schedule is None:
        optimized_schedule = _optimized_schedule_table(result.selected_connections, data)
    optimized_schedule.to_excel(run_dir / "optimized_schedule.xlsx", index=False)
    result.selected_connections.to_excel(run_dir / "selected_connections.xlsx", index=False)
    result.beaten_competitors.to_excel(run_dir / "beaten_competitors.xlsx", index=False)
    result.od_metrics.to_excel(run_dir / "od_metrics.xlsx", index=False)
    if rotation_candidates is None:
        rotation_candidates = pd.DataFrame(
            columns=[
                "rotation_id",
                "pair_id",
                "outbound_flight_no",
                "inbound_flight_no",
                "rotation_constraint_available",
                "skip_reason",
            ]
        )
    rotation_candidates.to_excel(run_dir / "rotation_candidates.xlsx", index=False)
    if rotation_report is None:
        rotation_report = pd.DataFrame(
            columns=[
                "rotation_id",
                "pair_id",
                "outbound_flight_no",
                "inbound_flight_no",
                "new_outbound_departure_time",
                "new_inbound_arrival_time",
                "required_delta_min",
                "actual_delta_min",
                "slack_min",
                "violation",
                "skip_reason",
            ]
        )
    rotation_report.to_excel(run_dir / "rotation_report.xlsx", index=False)
    if hub_capacity_report is None:
        hub_capacity_report = pd.DataFrame(
            columns=[
                "day",
                "bucket_id",
                "bucket_start_min",
                "bucket_end_min",
                "arrivals",
                "departures",
                "arrival_capacity",
                "departure_capacity",
                "arrival_slack",
                "departure_slack",
                "arrival_violation",
                "departure_violation",
                "total_violation",
            ]
        )
    hub_capacity_report.to_excel(run_dir / "hub_capacity_report.xlsx", index=False)
    if regularity_report is None:
        regularity_report = pd.DataFrame(
            columns=[
                "regularity_group_id",
                "normalized_flight_no",
                "origin",
                "destination",
                "direction_type",
                "days_in_group",
                "num_instances",
                "min_new_time",
                "max_new_time",
                "spread_min",
                "xdev_limit",
                "violation",
                "skip_reason",
            ]
        )
    regularity_report.to_excel(run_dir / "regularity_report.xlsx", index=False)
    if directional_balance_report is None:
        directional_balance_report = pd.DataFrame(
            columns=[
                "directional_pair_id",
                "day",
                "od_forward",
                "od_reverse",
                "forward_active_connections",
                "reverse_active_connections",
                "total_connections",
                "connection_diff",
                "alpha",
                "alpha_limit_value",
                "e1_violation",
                "forward_best_journey_time",
                "reverse_best_journey_time",
                "journey_time_diff",
                "gamma",
                "e2_applicable",
                "e2_violation",
                "e2_violation_amount",
                "directional_e2_mode",
                "allow_soft_directional_violations",
                "skip_reason",
                "violation",
            ]
        )
    directional_balance_report.to_excel(run_dir / "directional_balance_report.xlsx", index=False)
    objective_lines = [
        f"total objective: {metrics['total_objective']}",
        f"objective before penalty: {metrics.get('objective_before_penalty', metrics['total_objective'])}",
        f"objective after penalty: {metrics.get('objective_after_penalty', metrics['total_objective'])}",
        f"reward_before_violation_penalty: {metrics.get('reward_before_violation_penalty', metrics['total_objective'])}",
        f"violation_penalty_total: {metrics.get('violation_penalty_total', 0.0)}",
        f"net_solver_objective: {metrics.get('net_solver_objective', '')}",
        f"solution_quality: {metrics.get('solution_quality', '')}",
        f"total shift penalty: {metrics.get('total_shift_penalty', 0.0)}",
        f"connection reward: {metrics['connection_reward']}",
        f"ranking reward: {metrics['ranking_reward']}",
        f"solver status: {solver_status}",
        f"gap: {'' if gap is None else gap}",
        f"runtime: {runtime_sec:.3f}",
        f"feasibility result: {'UNKNOWN' if feasibility_text is None else ('FEASIBLE' if 'Result: FEASIBLE' in feasibility_text else 'INFEASIBLE')}",
    ]
    (run_dir / "objective_value.txt").write_text("\n".join(objective_lines), encoding="utf-8")
    summary_text = render_run_summary(metrics, solver_status, gap, runtime_sec)
    (run_dir / "run_summary.txt").write_text(summary_text, encoding="utf-8")
    if feasibility_text is not None:
        (run_dir / "feasibility_report.txt").write_text(feasibility_text, encoding="utf-8")


def build_comparison_to_baseline(baseline_metrics: dict, current_metrics: dict) -> dict:
    """Calculate current-vs-baseline comparison fields from real run metrics."""

    baseline_objective = float(baseline_metrics.get("total_objective", 0.0))
    current_objective = float(current_metrics.get("total_objective", 0.0))
    absolute_improvement = current_objective - baseline_objective
    if abs(baseline_objective) > 1e-12:
        percentage_improvement = 100.0 * absolute_improvement / abs(baseline_objective)
    else:
        percentage_improvement = 0.0
    return {
        "baseline_objective": baseline_objective,
        "current_objective": current_objective,
        "absolute_improvement": absolute_improvement,
        "percentage_improvement": percentage_improvement,
        "baseline_active_connections": int(baseline_metrics.get("active_connections", 0)),
        "current_active_connections": int(current_metrics.get("active_connections", 0)),
        "baseline_beaten_competitors": int(baseline_metrics.get("beaten_competitors", 0)),
        "current_beaten_competitors": int(current_metrics.get("beaten_competitors", 0)),
        "improved_od_count": int(current_metrics.get("improved_od_count", 0)),
        "worsened_od_count": int(current_metrics.get("worsened_od_count", 0)),
        "unchanged_od_count": int(current_metrics.get("unchanged_od_count", 0)),
    }


def build_directional_pairs_table(
    od_metrics: pd.DataFrame,
    keys_with_candidates: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    """Build O-D/reverse O-D day pairs, including missing reverse directions.

    keys_with_candidates verilirse, iki yonden biri hic TK aday baglantisi
    icermeyen ciftler E1/E2 kapsamindan cikarilir (model ile ayni kural).
    """

    columns = [
        "directional_pair_id",
        "day",
        "od_forward",
        "od_reverse",
        "forward_origin",
        "forward_destination",
        "reverse_origin",
        "reverse_destination",
        "forward_exists",
        "reverse_exists",
        "pair_active",
        "skip_reason",
    ]
    if od_metrics.empty:
        return pd.DataFrame(columns=columns)
    od_rows = {(str(row.od), int(row.day)): row for row in od_metrics.itertuples(index=False)}
    rows = []
    seen = set()
    idx = 0
    for (od, day), row in od_rows.items():
        reverse_od = _reverse_od(od)
        pair_key = tuple(sorted([od, reverse_od]) + [str(day)])
        if pair_key in seen:
            continue
        reverse = od_rows.get((reverse_od, day))
        seen.add(pair_key)
        forward_origin, forward_destination = _split_od(od)
        reverse_origin, reverse_destination = _split_od(reverse_od)
        pair_active = reverse is not None
        skip_reason = "" if reverse is not None else "missing_reverse_direction"
        if pair_active and keys_with_candidates is not None:
            forward_has = (str(od), int(day)) in keys_with_candidates
            reverse_has = (str(reverse_od), int(day)) in keys_with_candidates
            if not (forward_has and reverse_has):
                pair_active = False
                skip_reason = "one_direction_without_tk_candidates"
        rows.append(
            {
                "directional_pair_id": f"dir_{idx:05d}",
                "day": day,
                "od_forward": od,
                "od_reverse": reverse_od,
                "forward_origin": forward_origin,
                "forward_destination": forward_destination,
                "reverse_origin": reverse_origin,
                "reverse_destination": reverse_destination,
                "forward_exists": True,
                "reverse_exists": reverse is not None,
                "pair_active": pair_active,
                "skip_reason": skip_reason,
            }
        )
        idx += 1
    return pd.DataFrame(rows, columns=columns)


def build_directional_balance_report(
    od_metrics: pd.DataFrame,
    config: ModelConfig,
    directional_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build independent E1/E2 O-D/reverse O-D directional balance report."""

    columns = [
        "directional_pair_id",
        "day",
        "od_forward",
        "od_reverse",
        "forward_active_connections",
        "reverse_active_connections",
        "total_connections",
        "connection_diff",
        "alpha",
        "alpha_limit_value",
        "e1_formula",
        "e1_abs_tolerance",
        "e1_allowed_value",
        "e1_violation",
        "e1_violation_amount",
        "directional_e1_mode",
        "directional_e1_penalty_weight",
        "forward_best_journey_time",
        "reverse_best_journey_time",
        "journey_time_diff",
        "gamma",
        "e2_applicable",
        "e2_violation",
        "e2_violation_amount",
        "directional_e2_mode",
        "allow_soft_directional_violations",
        "skip_reason",
        "violation",
    ]
    if od_metrics.empty:
        return pd.DataFrame(columns=columns)
    if directional_pairs is None:
        directional_pairs = build_directional_pairs_table(od_metrics)
    metric_by_key = {(str(row.od), int(row.day)): row for row in od_metrics.itertuples(index=False)}
    rows = []
    for pair in directional_pairs.itertuples(index=False):
        skip_reason = "" if pd.isna(pair.skip_reason) else str(pair.skip_reason)
        forward = metric_by_key.get((str(pair.od_forward), int(pair.day)))
        reverse = metric_by_key.get((str(pair.od_reverse), int(pair.day)))
        if skip_reason or forward is None or reverse is None:
            rows.append(
                {
                    "directional_pair_id": pair.directional_pair_id,
                    "day": int(pair.day),
                    "od_forward": pair.od_forward,
                    "od_reverse": pair.od_reverse,
                    "forward_active_connections": 0 if forward is None else int(forward.active_connections),
                    "reverse_active_connections": 0 if reverse is None else int(reverse.active_connections),
                    "total_connections": 0,
                    "connection_diff": 0,
                    "alpha": config.directional_alpha,
                    "alpha_limit_value": 0.0,
                    "e1_formula": config.directional_e1_formula,
                    "e1_abs_tolerance": config.directional_e1_abs_tolerance,
                    "e1_allowed_value": 0.0,
                    "e1_violation": False,
                    "e1_violation_amount": 0.0,
                    "directional_e1_mode": config.directional_e1_mode,
                    "directional_e1_penalty_weight": config.directional_e1_penalty_weight,
                    "forward_best_journey_time": None if forward is None else forward.new_best_journey_time,
                    "reverse_best_journey_time": None if reverse is None else reverse.new_best_journey_time,
                    "journey_time_diff": None,
                    "gamma": config.directional_gamma_min,
                    "e2_applicable": False,
                    "e2_violation": False,
                    "e2_violation_amount": 0.0,
                    "directional_e2_mode": config.directional_e2_mode,
                    "allow_soft_directional_violations": config.allow_soft_directional_violations,
                    "skip_reason": skip_reason or "missing_reverse_direction",
                    "violation": False,
                }
            )
            continue
        count_forward = int(forward.active_connections)
        count_reverse = int(reverse.active_connections)
        total = count_forward + count_reverse
        diff = abs(count_forward - count_reverse)
        alpha_limit_value = float(config.directional_alpha) * total
        e1_allowed_value = alpha_limit_value
        if config.directional_e1_formula == "absolute_tolerance":
            e1_allowed_value += float(config.directional_e1_abs_tolerance)
        e1_violation_amount = max(0.0, float(diff) - float(e1_allowed_value))
        e1_violation = bool(e1_violation_amount > 1e-6)
        best_forward = forward.new_best_journey_time
        best_reverse = reverse.new_best_journey_time
        e2_applicable = count_forward > 0 and count_reverse > 0 and pd.notna(best_forward) and pd.notna(best_reverse)
        journey_diff = abs(float(best_forward) - float(best_reverse)) if e2_applicable else None
        e2_violation = bool(e2_applicable and journey_diff is not None and journey_diff > config.directional_gamma_min + 1e-6)
        e2_violation_amount = max(0.0, float(journey_diff) - float(config.directional_gamma_min)) if journey_diff is not None else 0.0
        rows.append(
            {
                "directional_pair_id": pair.directional_pair_id,
                "day": int(pair.day),
                "od_forward": pair.od_forward,
                "od_reverse": pair.od_reverse,
                "forward_active_connections": count_forward,
                "reverse_active_connections": count_reverse,
                "total_connections": total,
                "connection_diff": diff,
                "alpha": config.directional_alpha,
                "alpha_limit_value": alpha_limit_value,
                "e1_formula": config.directional_e1_formula,
                "e1_abs_tolerance": config.directional_e1_abs_tolerance,
                "e1_allowed_value": e1_allowed_value,
                "e1_violation": e1_violation,
                "e1_violation_amount": e1_violation_amount,
                "directional_e1_mode": config.directional_e1_mode,
                "directional_e1_penalty_weight": config.directional_e1_penalty_weight,
                "forward_best_journey_time": best_forward,
                "reverse_best_journey_time": best_reverse,
                "journey_time_diff": journey_diff,
                "gamma": config.directional_gamma_min,
                "e2_applicable": bool(e2_applicable),
                "e2_violation": e2_violation,
                "e2_violation_amount": e2_violation_amount,
                "directional_e2_mode": config.directional_e2_mode,
                "allow_soft_directional_violations": config.allow_soft_directional_violations,
                "skip_reason": "" if e2_applicable else ("one_direction_inactive" if total > 0 else ""),
                "violation": bool(e1_violation or e2_violation),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_comparison_to_baseline(run_dir: Path, comparison: dict) -> None:
    """Write outputs/<run_id>/comparison_to_baseline.txt."""

    lines = [
        "COMPARISON TO BASELINE",
        "----------------------",
        f"baseline objective: {comparison['baseline_objective']}",
        f"current objective: {comparison['current_objective']}",
        f"absolute improvement: {comparison['absolute_improvement']}",
        f"percentage improvement: {comparison['percentage_improvement']:.6f}%",
        f"baseline active connections: {comparison['baseline_active_connections']}",
        f"current active connections: {comparison['current_active_connections']}",
        f"baseline beaten competitors: {comparison['baseline_beaten_competitors']}",
        f"current beaten competitors: {comparison['current_beaten_competitors']}",
        f"improved O-D count: {comparison['improved_od_count']}",
        f"worsened O-D count: {comparison['worsened_od_count']}",
        f"unchanged O-D count: {comparison['unchanged_od_count']}",
    ]
    (run_dir / "comparison_to_baseline.txt").write_text("\n".join(lines), encoding="utf-8")


def render_run_summary(metrics: dict, solver_status: str, gap: float | None, runtime_sec: float) -> str:
    """Render key metrics for humans."""

    lines = ["RUN SUMMARY", "-----------"]
    for key, value in metrics.items():
        lines.append(f"{key}: {value}")
    lines.extend([f"runtime seconds: {runtime_sec:.3f}", f"solver status: {solver_status}", f"optimality gap: {gap}"])
    return "\n".join(lines)


def _selected_connections_table(active_connections: pd.DataFrame) -> pd.DataFrame:
    if active_connections.empty:
        return pd.DataFrame(
            columns=[
                "od",
                "origin",
                "destination",
                "day",
                "inbound_flight",
                "outbound_flight",
                "old_inbound_arrival",
                "new_inbound_arrival",
                "old_outbound_departure",
                "new_outbound_departure",
                "old_gap",
                "new_gap",
                "inbound_arr_time",
                "outbound_dep_time",
                "connection_gap_min",
                "journey_time_min",
                "active",
            ]
        )
    cols = [
        "connection_id",
        "O&D",
        "origin",
        "destination",
        "day",
        "inbound_flight",
        "outbound_flight",
        "arr_time",
        "dep_time",
        "connection_gap_min",
        "journey_time_min",
    ]
    optional_cols = [
        "old_inbound_arrival",
        "new_inbound_arrival",
        "old_outbound_departure",
        "new_outbound_departure",
        "old_gap",
        "new_gap",
        "old_journey_time_min",
        "new_journey_time_min",
        "inbound_shift_min",
        "outbound_shift_min",
        "beaten_best_competitor",
    ]
    cols += [col for col in optional_cols if col in active_connections.columns]
    selected = active_connections[cols].copy()
    selected = selected.rename(columns={"O&D": "od", "arr_time": "inbound_arr_time", "dep_time": "outbound_dep_time"})
    if "old_inbound_arrival" not in selected.columns:
        selected["old_inbound_arrival"] = selected["inbound_arr_time"]
    if "new_inbound_arrival" not in selected.columns:
        selected["new_inbound_arrival"] = selected["inbound_arr_time"]
    if "old_outbound_departure" not in selected.columns:
        selected["old_outbound_departure"] = selected["outbound_dep_time"]
    if "new_outbound_departure" not in selected.columns:
        selected["new_outbound_departure"] = selected["outbound_dep_time"]
    if "old_gap" not in selected.columns:
        selected["old_gap"] = selected["connection_gap_min"]
    if "new_gap" not in selected.columns:
        selected["new_gap"] = selected["connection_gap_min"]
    if "new_journey_time_min" in selected.columns:
        selected["journey_time_min"] = selected["new_journey_time_min"]
    selected["connection_gap_min"] = selected["new_gap"]
    selected["inbound_arr_time"] = selected["new_inbound_arrival"]
    selected["outbound_dep_time"] = selected["new_outbound_departure"]
    if "active" not in selected.columns:
        selected["active"] = True
    ordered = [
        "connection_id",
        "od",
        "origin",
        "destination",
        "day",
        "inbound_flight",
        "outbound_flight",
        "old_inbound_arrival",
        "new_inbound_arrival",
        "old_outbound_departure",
        "new_outbound_departure",
        "old_gap",
        "new_gap",
        "inbound_arr_time",
        "outbound_dep_time",
        "connection_gap_min",
        "journey_time_min",
        "active",
    ]
    ordered += [col for col in ["inbound_shift_min", "outbound_shift_min", "old_journey_time_min", "new_journey_time_min", "beaten_best_competitor"] if col in selected.columns]
    selected = selected[[col for col in ordered if col in selected.columns]]
    return selected


def _optimized_schedule_table(selected_connections: pd.DataFrame, data: PreparedData) -> pd.DataFrame:
    rows = []
    for row in selected_connections.itertuples(index=False):
        rows.append(
            {
                "flight_id": f"in_{row.connection_id}",
                "flight_no": row.inbound_flight,
                "day": row.day,
                "origin": row.origin,
                "destination": "IST",
                "old_ist_time": row.old_inbound_arrival if hasattr(row, "old_inbound_arrival") else row.inbound_arr_time,
                "new_ist_time": row.new_inbound_arrival if hasattr(row, "new_inbound_arrival") else row.inbound_arr_time,
                "time_change_min": row.inbound_shift_min if hasattr(row, "inbound_shift_min") else 0,
                "is_adjustable": True,
            }
        )
        rows.append(
            {
                "flight_id": f"out_{row.connection_id}",
                "flight_no": row.outbound_flight,
                "day": row.day,
                "origin": "IST",
                "destination": row.destination,
                "old_ist_time": row.old_outbound_departure if hasattr(row, "old_outbound_departure") else row.outbound_dep_time,
                "new_ist_time": row.new_outbound_departure if hasattr(row, "new_outbound_departure") else row.outbound_dep_time,
                "time_change_min": row.outbound_shift_min if hasattr(row, "outbound_shift_min") else 0,
                "is_adjustable": True,
            }
        )
    schedule = pd.DataFrame(rows).drop_duplicates(["flight_id"]) if rows else pd.DataFrame()
    return schedule


def _reverse_od(od: str) -> str:
    parts = str(od).split("-", 1)
    if len(parts) != 2:
        return str(od)
    return f"{parts[1]}-{parts[0]}"


def _split_od(od: str) -> tuple[str, str]:
    parts = str(od).split("-", 1)
    if len(parts) != 2:
        return str(od), ""
    return parts[0], parts[1]
