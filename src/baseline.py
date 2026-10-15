"""Baseline objective calculation for the current schedule."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ModelConfig
from .preprocessing import PreparedData, connection_reward


@dataclass
class BaselineResult:
    """Baseline metrics and objective components."""

    od_metrics: pd.DataFrame
    selected_connections: pd.DataFrame
    beaten_competitors: pd.DataFrame
    summary: dict


def compute_baseline(data: PreparedData, config: ModelConfig) -> BaselineResult:
    """Compute current-schedule objective and detailed O-D/day metrics."""

    print("[INFO] Building baseline...")
    tk = data.tk_connections.copy()
    tk = tk[
        (tk["connection_gap_min"] >= config.min_connection_min)
        & (tk["connection_gap_min"] <= config.max_connection_min)
    ].copy()
    comp = data.competitor_connections.copy()
    od_rows = []
    beaten_rows = []

    for row in data.od_day.itertuples(index=False):
        od = row.od
        day = int(row.day)
        tk_od = tk[(tk["O&D"] == od) & (tk["day"] == day)]
        comp_od = comp[(comp["O&D"] == od) & (comp["day"] == day)]
        active_connections = len(tk_od)
        best_tk = float(tk_od["journey_time_min"].min()) if active_connections else float("inf")
        competitor_count = len(comp_od)
        beaten_count = int((comp_od["journey_time_min"] >= best_tk).sum()) if active_connections else 0
        initial_rank = max(1, competitor_count - beaten_count + 1)
        new_rank = initial_rank
        ranking_reward = lookup_ranking_reward(data.ranking_lookup, competitor_count, initial_rank, new_rank)
        conn_reward = connection_reward(active_connections, config.max_connection_slots, config.connection_reward_decay)
        weighted = float(row.importance_factor) * (conn_reward + ranking_reward)
        od_rows.append(
            {
                "od": od,
                "origin": row.origin,
                "destination": row.destination,
                "day": day,
                "importance_factor": row.importance_factor,
                "active_connections": active_connections,
                "old_best_journey_time": None if best_tk == float("inf") else best_tk,
                "new_best_journey_time": None if best_tk == float("inf") else best_tk,
                "competitor_count": competitor_count,
                "beaten_competitors": beaten_count,
                "initial_rank": initial_rank,
                "new_rank": new_rank,
                "connection_reward": conn_reward,
                "ranking_reward": ranking_reward,
                "total_weighted_reward": weighted,
                "status_improved_worsened_unchanged": "unchanged",
            }
        )
        for comp_row in comp_od.itertuples(index=False):
            beaten_rows.append(
                {
                    "od": od,
                    "day": day,
                    "competitor_id": comp_row.competitor_id,
                    "competitor_carrier": comp_row.carrier_code,
                    "competitor_time_min": comp_row.journey_time_min,
                    "best_tk_time_min": None if best_tk == float("inf") else best_tk,
                    "beaten": bool(active_connections and best_tk <= comp_row.journey_time_min),
                }
            )

    od_metrics = pd.DataFrame(od_rows)
    selected = _baseline_selected_connections(tk)
    beaten = pd.DataFrame(beaten_rows)
    summary = summarize_baseline(od_metrics, selected, beaten)
    return BaselineResult(od_metrics, selected, beaten, summary)


def lookup_ranking_reward(lookup: dict[tuple[int, int, int], float], competitor_count: int, initial_rank: int, new_rank: int) -> float:
    """Lookup ranking reward with safe clipping for edge cases."""

    if competitor_count <= 0:
        return 0.0
    initial = min(max(1, int(initial_rank)), int(competitor_count))
    new = min(max(1, int(new_rank)), int(competitor_count))
    return float(lookup.get((int(competitor_count), initial, new), 0.0))


def summarize_baseline(od_metrics: pd.DataFrame, selected: pd.DataFrame, beaten: pd.DataFrame) -> dict:
    """Aggregate baseline metrics."""

    total_objective = float(od_metrics["total_weighted_reward"].sum()) if not od_metrics.empty else 0.0
    connection_reward = float((od_metrics["importance_factor"] * od_metrics["connection_reward"]).sum()) if not od_metrics.empty else 0.0
    ranking_reward = float((od_metrics["importance_factor"] * od_metrics["ranking_reward"]).sum()) if not od_metrics.empty else 0.0
    return {
        "total_objective": total_objective,
        "connection_reward": connection_reward,
        "ranking_reward": ranking_reward,
        "active_connections": int(selected["active"].sum()) if not selected.empty else 0,
        "beaten_competitors": int(beaten["beaten"].sum()) if not beaten.empty else 0,
        "od_day_count": int(len(od_metrics)),
    }


def _baseline_selected_connections(tk: pd.DataFrame) -> pd.DataFrame:
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
    selected = tk[cols].copy()
    selected = selected.rename(
        columns={
            "O&D": "od",
            "arr_time": "inbound_arr_time",
            "dep_time": "outbound_dep_time",
        }
    )
    selected["active"] = True
    return selected
