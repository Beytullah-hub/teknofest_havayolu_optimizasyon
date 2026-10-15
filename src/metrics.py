"""Run metrics for baseline and optimized solutions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_run_metrics(od_metrics: pd.DataFrame, selected_connections: pd.DataFrame, beaten_competitors: pd.DataFrame) -> dict:
    """Compute common metrics requested for experiment tracking."""

    gaps = selected_connections.loc[selected_connections["active"] == True, "connection_gap_min"] if not selected_connections.empty else pd.Series(dtype=float)
    beaten_count = int(beaten_competitors["beaten"].sum()) if "beaten" in beaten_competitors else 0
    competitor_count = len(beaten_competitors)
    rank_delta = od_metrics["initial_rank"] - od_metrics["new_rank"] if not od_metrics.empty else pd.Series(dtype=float)
    return {
        "total_objective": float(od_metrics["total_weighted_reward"].sum()) if not od_metrics.empty else 0.0,
        "connection_reward": float((od_metrics["importance_factor"] * od_metrics["connection_reward"]).sum()) if not od_metrics.empty else 0.0,
        "ranking_reward": float((od_metrics["importance_factor"] * od_metrics["ranking_reward"]).sum()) if not od_metrics.empty else 0.0,
        "active_connections": int(selected_connections["active"].sum()) if "active" in selected_connections else 0,
        "beaten_competitors": beaten_count,
        "beat_ratio": float(beaten_count / competitor_count) if competitor_count else 0.0,
        "improved_od_count": int((od_metrics["status_improved_worsened_unchanged"] == "improved").sum()) if not od_metrics.empty else 0,
        "worsened_od_count": int((od_metrics["status_improved_worsened_unchanged"] == "worsened").sum()) if not od_metrics.empty else 0,
        "unchanged_od_count": int((od_metrics["status_improved_worsened_unchanged"] == "unchanged").sum()) if not od_metrics.empty else 0,
        "average_rank_improvement": float(rank_delta.replace([np.inf, -np.inf], np.nan).dropna().mean()) if not rank_delta.empty else 0.0,
        "average_connection_gap": float(gaps.mean()) if len(gaps) else 0.0,
        "median_connection_gap": float(gaps.median()) if len(gaps) else 0.0,
        "connection_count_60_90": int(((gaps >= 60) & (gaps < 90)).sum()) if len(gaps) else 0,
        "connection_count_90_180": int(((gaps >= 90) & (gaps < 180)).sum()) if len(gaps) else 0,
        "connection_count_180_300": int(((gaps >= 180) & (gaps <= 300)).sum()) if len(gaps) else 0,
    }

