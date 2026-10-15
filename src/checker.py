"""Independent feasibility checks for produced outputs."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import ModelConfig
from .model import circular_spread, normalize_flight_no
from .output_writer import build_directional_balance_report


ROTATION_REPORT_COLUMNS = [
    "pair_id",
    "outbound_flight_no",
    "inbound_flight_no",
    "new_outbound_departure_time",
    "new_inbound_arrival_time",
    "outbound_block_time_min",
    "inbound_block_time_min",
    "min_ground_time_min",
    "required_delta_min",
    "actual_delta_min",
    "slack_min",
    "violation",
    "skip_reason",
]

HUB_CAPACITY_REPORT_COLUMNS = [
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

REGULARITY_GROUP_COLUMNS = [
    "regularity_group_id",
    "normalized_flight_no",
    "origin",
    "destination",
    "direction_type",
    "days_in_group",
    "num_instances",
    "old_time_min_list",
    "new_time_min_list",
    "min_new_time",
    "max_new_time",
    "spread_min",
    "regularity_constraint_active",
    "skip_reason",
]

REGULARITY_REPORT_COLUMNS = [
    "regularity_group_id",
    "normalized_flight_no",
    "origin",
    "destination",
    "direction_type",
    "days_in_group",
    "num_instances",
    "old_spread_min",
    "theoretical_min_spread",
    "base_xdev",
    "effective_xdev",
    "relax_applied",
    "relax_amount",
    "targeted_max_relax",
    "targeted_soft",
    "min_new_time",
    "max_new_time",
    "spread_min",
    "xdev_limit",
    "violation",
    "over_limit",
    "pre_existing_violation",
    "skip_reason",
]


@dataclass
class FeasibilityReport:
    """Checker result."""

    connection_window_violations: int = 0
    rotation_violations: int = 0
    hub_capacity_violations: int = 0
    regularity_violations: int = 0
    directional_balance_violations: int = 0
    directional_e1_violations: int = 0
    directional_e2_violations: int = 0
    directional_e1_hard_violations: int = 0
    directional_e1_soft_violations: int = 0
    directional_e2_hard_violations: int = 0
    directional_e2_soft_violations: int = 0
    directional_e1_soft_violation_amount_total: float = 0.0
    directional_e2_soft_violation_amount_total: float = 0.0
    feasible_but_inactive_connections: int = 0
    infeasible_but_active_connections: int = 0
    mandatory_connection_selection: bool = False
    rotation_enabled: bool = True
    capacity_enabled: bool = True
    regularity_enabled: bool = True
    regularity_relax_mode: str = "none"
    regularity_relax_applied_groups: int = 0
    regularity_soft_violations: int = 0
    regularity_soft_violation_amount_total: float = 0.0
    regularity_max_effective_xdev: float = 0.0
    regularity_pre_existing_violations: int = 0
    directional_enabled: bool = True
    objective_difference: float = 0.0
    missing_values: int = 0

    @property
    def feasible(self) -> bool:
        return (
            self.connection_window_violations == 0
            and (not self.rotation_enabled or self.rotation_violations == 0)
            and (not self.capacity_enabled or self.hub_capacity_violations == 0)
            and (not self.regularity_enabled or self.regularity_violations == 0)
            and (not self.directional_enabled or self.directional_balance_violations == 0)
            and (not self.directional_enabled or self.directional_e1_hard_violations == 0)
            and (not self.directional_enabled or self.directional_e2_hard_violations == 0)
            and (
                not self.mandatory_connection_selection
                or (
                    self.feasible_but_inactive_connections == 0
                    and self.infeasible_but_active_connections == 0
                )
            )
            and abs(self.objective_difference) <= 1e-4
            and self.missing_values == 0
        )

    @property
    def hard_violations(self) -> int:
        return (
            self.connection_window_violations
            + (self.rotation_violations if self.rotation_enabled else 0)
            + (self.hub_capacity_violations if self.capacity_enabled else 0)
            + (self.regularity_violations if self.regularity_enabled else 0)
            + (self.directional_e1_hard_violations if self.directional_enabled else 0)
            + (self.directional_e2_hard_violations if self.directional_enabled else 0)
            + (
                self.feasible_but_inactive_connections + self.infeasible_but_active_connections
                if self.mandatory_connection_selection
                else 0
            )
        )


def check_solution_tables(
    selected_connections: pd.DataFrame,
    od_metrics: pd.DataFrame,
    expected_objective: float,
    config: ModelConfig,
    rotation_report: pd.DataFrame | None = None,
    hub_capacity_report: pd.DataFrame | None = None,
    regularity_report: pd.DataFrame | None = None,
    directional_balance_report: pd.DataFrame | None = None,
    feasible_but_inactive_report: pd.DataFrame | None = None,
    rotation_enabled: bool = True,
    capacity_enabled: bool = True,
    regularity_enabled: bool = True,
    directional_enabled: bool = True,
) -> FeasibilityReport:
    """Check output tables independently from solver state."""

    report = FeasibilityReport()
    report.mandatory_connection_selection = bool(config.mandatory_connection_selection)
    report.rotation_enabled = bool(rotation_enabled)
    report.capacity_enabled = bool(capacity_enabled)
    report.regularity_enabled = bool(regularity_enabled)
    report.regularity_relax_mode = str(config.regularity_relax_mode)
    report.directional_enabled = bool(directional_enabled)
    active = selected_connections[selected_connections.get("active", False) == True].copy()
    if not active.empty:
        gap_col = "new_gap" if "new_gap" in active.columns else "connection_gap_min"
        report.connection_window_violations = int(
            (
                (active[gap_col] < config.min_connection_min - 1e-6)
                | (active[gap_col] > config.max_connection_min + 1e-6)
            ).sum()
        )
        required = ["od", "day", "inbound_flight", "outbound_flight", "connection_gap_min", "journey_time_min"]
        if "new_gap" in active.columns:
            required.append("new_gap")
        if "new_inbound_arrival" in active.columns:
            required.extend(["old_inbound_arrival", "new_inbound_arrival", "old_outbound_departure", "new_outbound_departure"])
        report.missing_values += int(active[required].isna().sum().sum())
    if not od_metrics.empty:
        recalculated = float(od_metrics["total_weighted_reward"].sum())
        report.objective_difference = recalculated - float(expected_objective)
        required_metrics = ["od", "day", "importance_factor", "active_connections", "total_weighted_reward"]
        report.missing_values += int(od_metrics[required_metrics].isna().sum().sum())
    if rotation_report is not None and not rotation_report.empty:
        if "skip_reason" in rotation_report.columns:
            constrained = rotation_report[rotation_report["skip_reason"].fillna("").astype(str) == ""].copy()
        else:
            constrained = rotation_report[rotation_report.get("status", "") == "constrained"].copy()
        if not constrained.empty:
            if "violation" in constrained.columns:
                report.rotation_violations = int(constrained["violation"].fillna(False).sum())
            elif "slack_min" in constrained.columns:
                report.rotation_violations = int((constrained["slack_min"] < -1e-6).sum())
            required_rotation = [
                "pair_id",
                "outbound_flight_no",
                "inbound_flight_no",
                "new_outbound_departure_time",
                "new_inbound_arrival_time",
                "required_delta_min",
                "actual_delta_min",
                "slack_min",
            ]
            present_required = [col for col in required_rotation if col in constrained.columns]
            report.missing_values += int(constrained[present_required].isna().sum().sum())
    if hub_capacity_report is not None and not hub_capacity_report.empty:
        if "total_violation" in hub_capacity_report.columns:
            report.hub_capacity_violations = int(hub_capacity_report["total_violation"].fillna(False).sum())
        elif "violation" in hub_capacity_report.columns:
            report.hub_capacity_violations = int(hub_capacity_report["violation"].fillna(False).sum())
        else:
            arr_bad = hub_capacity_report["arrivals"] > hub_capacity_report["arrival_capacity"]
            dep_bad = hub_capacity_report["departures"] > hub_capacity_report["departure_capacity"]
            report.hub_capacity_violations = int((arr_bad | dep_bad).sum())
        required_hub = [
            "day",
            "bucket_start_min" if "bucket_start_min" in hub_capacity_report.columns else "bucket_start",
            "bucket_end_min" if "bucket_end_min" in hub_capacity_report.columns else "bucket_end",
            "arrivals",
            "departures",
            "arrival_capacity",
            "departure_capacity",
        ]
        present_required = [col for col in required_hub if col in hub_capacity_report.columns]
        report.missing_values += int(hub_capacity_report[present_required].isna().sum().sum())
    if regularity_report is not None and not regularity_report.empty:
        if "skip_reason" in regularity_report.columns:
            constrained = regularity_report[regularity_report["skip_reason"].fillna("").astype(str) == ""].copy()
        else:
            constrained = regularity_report[regularity_report.get("status", "") == "constrained"].copy()
        if not constrained.empty:
            if "relax_applied" in constrained.columns:
                report.regularity_relax_applied_groups = int(constrained["relax_applied"].fillna(False).astype(bool).sum())
            if "xdev_limit" in constrained.columns:
                xdev_values = pd.to_numeric(constrained["xdev_limit"], errors="coerce").dropna()
                if len(xdev_values):
                    report.regularity_max_effective_xdev = float(xdev_values.max())
            if "spread_min" in constrained.columns and "xdev_limit" in constrained.columns:
                excess = (
                    pd.to_numeric(constrained["spread_min"], errors="coerce")
                    - pd.to_numeric(constrained["xdev_limit"], errors="coerce")
                ).clip(lower=0)
                if config.regularity_mode == "soft" or config.regularity_relax_mode == "targeted_soft":
                    soft_mask = pd.Series(True, index=constrained.index)
                    if config.regularity_relax_mode == "targeted_soft" and "targeted_soft" in constrained.columns:
                        soft_mask = constrained["targeted_soft"].fillna(False).astype(bool)
                    report.regularity_soft_violations = int((excess[soft_mask] > 1e-6).sum())
                    report.regularity_soft_violation_amount_total = float(excess[soft_mask].sum())
            if "pre_existing_violation" in constrained.columns:
                report.regularity_pre_existing_violations = int(
                    constrained["pre_existing_violation"].fillna(False).astype(bool).sum()
                )
            if "violation" in constrained.columns:
                violations = constrained["violation"].fillna(False).astype(bool)
                if config.regularity_mode == "soft":
                    report.regularity_violations = 0
                elif config.regularity_relax_mode == "targeted_soft" and "targeted_soft" in constrained.columns:
                    report.regularity_violations = int((violations & ~constrained["targeted_soft"].fillna(False).astype(bool)).sum())
                else:
                    report.regularity_violations = int(violations.sum())
            elif "spread_min" in constrained.columns:
                limits = pd.to_numeric(
                    constrained.get("xdev_limit", pd.Series(config.schedule_regular_dev_min, index=constrained.index)),
                    errors="coerce",
                ).fillna(config.schedule_regular_dev_min)
                report.regularity_violations = int((pd.to_numeric(constrained["spread_min"], errors="coerce") > limits + 1e-6).sum())
            elif "new_time_diff_min" in constrained.columns:
                limits = pd.to_numeric(
                    constrained.get("limit_min", pd.Series(config.schedule_regular_dev_min, index=constrained.index)),
                    errors="coerce",
                ).fillna(config.schedule_regular_dev_min)
                report.regularity_violations = int((pd.to_numeric(constrained["new_time_diff_min"], errors="coerce") > limits + 1e-6).sum())
            required_regularity = (
                ["regularity_group_id", "normalized_flight_no", "spread_min", "xdev_limit"]
                if "regularity_group_id" in constrained.columns
                else ["flight_no", "direction", "day_a", "day_b", "new_time_diff_min", "limit_min"]
            )
            present_required = [col for col in required_regularity if col in constrained.columns]
            report.missing_values += int(constrained[present_required].isna().sum().sum())
    if directional_balance_report is not None and not directional_balance_report.empty:
        if "e1_violation" in directional_balance_report.columns or "e2_violation" in directional_balance_report.columns:
            report.directional_e1_violations = int(
                directional_balance_report.get("e1_violation", pd.Series(dtype=bool)).fillna(False).sum()
            )
            report.directional_e2_violations = int(
                directional_balance_report.get("e2_violation", pd.Series(dtype=bool)).fillna(False).sum()
            )
            e1_amount = pd.to_numeric(
                directional_balance_report.get("e1_violation_amount", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0.0)
            e2_amount = pd.to_numeric(
                directional_balance_report.get("e2_violation_amount", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0.0)
            report.directional_e1_soft_violation_amount_total = float(e1_amount.sum())
            report.directional_e2_soft_violation_amount_total = float(e2_amount.sum())
            if config.directional_e1_mode == "soft" or config.directional_e1_formula == "soft_only":
                report.directional_e1_soft_violations = report.directional_e1_violations
            elif config.directional_e1_mode == "hard":
                report.directional_e1_hard_violations = report.directional_e1_violations
            if config.directional_e2_mode == "soft" and config.allow_soft_directional_violations:
                report.directional_e2_soft_violations = report.directional_e2_violations
            elif config.directional_e2_mode == "hard":
                report.directional_e2_hard_violations = report.directional_e2_violations
            report.directional_balance_violations = report.directional_e1_hard_violations + report.directional_e2_hard_violations
        elif "violation" in directional_balance_report.columns:
            report.directional_balance_violations = int(directional_balance_report["violation"].fillna(False).sum())
        else:
            count_bad = directional_balance_report.get("count_balance_violation", pd.Series(dtype=bool)).fillna(False)
            time_bad = directional_balance_report.get("best_time_balance_violation", pd.Series(dtype=bool)).fillna(False)
            report.directional_e1_violations = int(count_bad.sum())
            report.directional_e2_violations = int(time_bad.sum())
            report.directional_balance_violations = int((count_bad | time_bad).sum())
        required_directional = [
            "od_forward" if "od_forward" in directional_balance_report.columns else "od",
            "od_reverse" if "od_reverse" in directional_balance_report.columns else "reverse_od",
            "day",
            "forward_active_connections" if "forward_active_connections" in directional_balance_report.columns else "active_connections",
            "reverse_active_connections",
            "e1_violation" if "e1_violation" in directional_balance_report.columns else "count_balance_violation",
            "e2_violation" if "e2_violation" in directional_balance_report.columns else "best_time_balance_violation",
        ]
        present_required = [col for col in required_directional if col in directional_balance_report.columns]
        report.missing_values += int(directional_balance_report[present_required].isna().sum().sum())
    if feasible_but_inactive_report is not None and not feasible_but_inactive_report.empty:
        if "violation_type" in feasible_but_inactive_report.columns:
            report.feasible_but_inactive_connections = int(
                (feasible_but_inactive_report["violation_type"] == "feasible_but_inactive").sum()
            )
            report.infeasible_but_active_connections = int(
                (feasible_but_inactive_report["violation_type"] == "infeasible_but_active").sum()
            )
        else:
            report.feasible_but_inactive_connections = int(len(feasible_but_inactive_report))
    return report


def build_feasible_but_inactive_report(
    all_candidate_connections: pd.DataFrame | None,
    selected_connections: pd.DataFrame | None,
    optimized_schedule: pd.DataFrame | None,
    config: ModelConfig,
) -> pd.DataFrame:
    """Validate mandatory connection selection from candidate and schedule tables."""

    columns = [
        "connection_id",
        "od",
        "day",
        "inbound_flight_no",
        "outbound_flight_no",
        "old_gap",
        "new_gap",
        "reason",
        "violation_type",
    ]
    if all_candidate_connections is None or all_candidate_connections.empty:
        return pd.DataFrame(columns=columns)

    candidates = all_candidate_connections.copy()
    selected_ids: set[str] = set()
    if selected_connections is not None and not selected_connections.empty and "connection_id" in selected_connections.columns:
        active = selected_connections[selected_connections.get("active", True) == True]
        selected_ids = set(active["connection_id"].astype(str))

    shift_by_event: dict[str, float] = {}
    if optimized_schedule is not None and not optimized_schedule.empty and {"flight_id", "time_change_min"}.issubset(optimized_schedule.columns):
        shift_by_event = (
            optimized_schedule.assign(flight_id=lambda df: df["flight_id"].astype(str))
            .set_index("flight_id")["time_change_min"]
            .astype(float)
            .to_dict()
        )

    rows = []
    for row in candidates.itertuples(index=False):
        connection_id = str(getattr(row, "connection_id", ""))
        old_gap = float(getattr(row, "old_gap", getattr(row, "connection_gap_min", 0.0)) or 0.0)
        new_gap = getattr(row, "new_gap", None)
        inbound_event_id = getattr(row, "inbound_event_id", None)
        outbound_event_id = getattr(row, "outbound_event_id", None)
        if inbound_event_id is not None and outbound_event_id is not None and shift_by_event:
            inbound_shift = shift_by_event.get(str(inbound_event_id), 0.0)
            outbound_shift = shift_by_event.get(str(outbound_event_id), 0.0)
            new_gap = old_gap + outbound_shift - inbound_shift
        if pd.isna(new_gap):
            continue
        new_gap = float(new_gap)
        feasible = config.min_connection_min <= new_gap <= config.max_connection_min
        active = connection_id in selected_ids or bool(getattr(row, "active_x", False))
        if feasible and not active:
            rows.append(
                {
                    "connection_id": connection_id,
                    "od": getattr(row, "od", getattr(row, "O_D", "")),
                    "day": getattr(row, "day", ""),
                    "inbound_flight_no": getattr(row, "inbound_flight_no", getattr(row, "inbound_flight", "")),
                    "outbound_flight_no": getattr(row, "outbound_flight_no", getattr(row, "outbound_flight", "")),
                    "old_gap": old_gap,
                    "new_gap": new_gap,
                    "reason": "connection gap is inside window but connection is not active",
                    "violation_type": "feasible_but_inactive",
                }
            )
        elif active and not feasible:
            rows.append(
                {
                    "connection_id": connection_id,
                    "od": getattr(row, "od", getattr(row, "O_D", "")),
                    "day": getattr(row, "day", ""),
                    "inbound_flight_no": getattr(row, "inbound_flight_no", getattr(row, "inbound_flight", "")),
                    "outbound_flight_no": getattr(row, "outbound_flight_no", getattr(row, "outbound_flight", "")),
                    "old_gap": old_gap,
                    "new_gap": new_gap,
                    "reason": "connection is active but connection gap is outside window",
                    "violation_type": "infeasible_but_active",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_rotation_report_from_tables(
    optimized_schedule: pd.DataFrame,
    rotation_candidates: pd.DataFrame,
    config: ModelConfig,
) -> pd.DataFrame:
    """Validate rotation candidates from output tables, independently from solver state."""

    if rotation_candidates is None or rotation_candidates.empty:
        return pd.DataFrame(columns=ROTATION_REPORT_COLUMNS)
    if optimized_schedule is None or optimized_schedule.empty:
        rows = []
        for cand in rotation_candidates.itertuples(index=False):
            rows.append(_rotation_report_skip_row(cand, "optimized_schedule is empty"))
        return pd.DataFrame(rows, columns=ROTATION_REPORT_COLUMNS)

    schedule = optimized_schedule.copy()
    if "flight_id" not in schedule.columns or "new_ist_time" not in schedule.columns:
        rows = []
        for cand in rotation_candidates.itertuples(index=False):
            rows.append(_rotation_report_skip_row(cand, "optimized_schedule missing flight_id/new_ist_time"))
        return pd.DataFrame(rows, columns=ROTATION_REPORT_COLUMNS)

    schedule["new_ist_time"] = pd.to_datetime(schedule["new_ist_time"], errors="coerce")
    schedule_by_id = schedule.set_index("flight_id").to_dict("index")
    rows = []
    for cand in rotation_candidates.itertuples(index=False):
        raw_skip_reason = getattr(cand, "skip_reason", "")
        skip_reason = "" if pd.isna(raw_skip_reason) else str(raw_skip_reason)
        raw_available = getattr(cand, "rotation_constraint_available", False)
        if pd.isna(raw_available):
            available = False
        elif isinstance(raw_available, str):
            available = raw_available.strip().lower() in {"true", "1", "yes"}
        else:
            available = bool(raw_available)
        if not available or skip_reason:
            rows.append(_rotation_report_skip_row(cand, skip_reason or "rotation constraint unavailable"))
            continue
        outbound_event_id = getattr(cand, "outbound_event_id", None)
        inbound_event_id = getattr(cand, "inbound_event_id", None)
        out_schedule = schedule_by_id.get(outbound_event_id)
        in_schedule = schedule_by_id.get(inbound_event_id)
        if out_schedule is None or in_schedule is None:
            rows.append(_rotation_report_skip_row(cand, "outbound or inbound event missing from optimized_schedule"))
            continue
        new_out = out_schedule.get("new_ist_time")
        new_in = in_schedule.get("new_ist_time")
        if pd.isna(new_out) or pd.isna(new_in):
            rows.append(_rotation_report_skip_row(cand, "new outbound or inbound time is missing"))
            continue
        old_out_min = getattr(cand, "old_outbound_departure_min", None)
        old_in_min = getattr(cand, "old_inbound_arrival_min", None)
        out_shift = out_schedule.get("time_change_min", 0.0)
        in_shift = in_schedule.get("time_change_min", 0.0)
        if old_out_min is not None and old_in_min is not None and not pd.isna(old_out_min) and not pd.isna(old_in_min):
            actual_delta = (float(old_in_min) + float(in_shift or 0.0)) - (float(old_out_min) + float(out_shift or 0.0))
        else:
            actual_delta = (new_in - new_out).total_seconds() / 60.0
            if actual_delta <= 0:
                actual_delta += 7 * 24 * 60
        required_delta = float(getattr(cand, "required_delta_min", 0.0) or 0.0)
        slack = actual_delta - required_delta
        rows.append(
            {
                "pair_id": getattr(cand, "pair_id", None),
                "outbound_flight_no": getattr(cand, "outbound_flight_no", None),
                "inbound_flight_no": getattr(cand, "inbound_flight_no", None),
                "new_outbound_departure_time": new_out,
                "new_inbound_arrival_time": new_in,
                "outbound_block_time_min": getattr(cand, "outbound_block_time_min", None),
                "inbound_block_time_min": getattr(cand, "inbound_block_time_min", None),
                "min_ground_time_min": getattr(cand, "min_ground_time_min", config.min_ground_time_min),
                "required_delta_min": required_delta,
                "actual_delta_min": actual_delta,
                "slack_min": slack,
                "violation": bool(slack < -1e-6),
                "skip_reason": "",
            }
        )
    return pd.DataFrame(rows, columns=ROTATION_REPORT_COLUMNS)


def build_hub_buckets(config: ModelConfig) -> pd.DataFrame:
    """Build 0-1440 minute hub bucket definitions."""

    rows = []
    bucket_count = 1440 // config.bucket_size_min
    for bucket in range(bucket_count):
        rows.append(
            {
                "bucket_id": bucket,
                "bucket_start_min": bucket * config.bucket_size_min,
                "bucket_end_min": (bucket + 1) * config.bucket_size_min,
                "departure_capacity": config.departure_bucket_capacity,
                "arrival_capacity": config.arrival_bucket_capacity,
            }
        )
    return pd.DataFrame(rows)


def build_hub_capacity_report_from_schedule(optimized_schedule: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Validate hub capacity independently from optimized_schedule.xlsx."""

    if optimized_schedule is None or optimized_schedule.empty:
        return pd.DataFrame(columns=HUB_CAPACITY_REPORT_COLUMNS)
    schedule = optimized_schedule.copy()
    required = {"day", "origin", "destination", "new_ist_time"}
    if not required.issubset(schedule.columns):
        return pd.DataFrame(columns=HUB_CAPACITY_REPORT_COLUMNS)

    schedule["new_ist_time"] = pd.to_datetime(schedule["new_ist_time"], errors="coerce")
    schedule["minute_of_day"] = schedule["new_ist_time"].dt.hour * 60 + schedule["new_ist_time"].dt.minute
    schedule["bucket_id"] = (schedule["minute_of_day"] // config.bucket_size_min).clip(lower=0, upper=(1440 // config.bucket_size_min) - 1)
    schedule["direction"] = ""
    hub = config.hub
    schedule.loc[schedule["destination"].astype(str).str.upper().str.strip() == hub, "direction"] = "inbound"
    schedule.loc[schedule["origin"].astype(str).str.upper().str.strip() == hub, "direction"] = "outbound"
    schedule = schedule.dropna(subset=["day", "bucket_id"])
    if schedule.empty:
        return pd.DataFrame(columns=HUB_CAPACITY_REPORT_COLUMNS)

    days = sorted(schedule["day"].dropna().astype(int).unique().tolist())
    bucket_count = 1440 // config.bucket_size_min
    grouped = schedule.groupby(["day", "bucket_id", "direction"]).size().to_dict()
    rows = []
    for day in days:
        for bucket in range(bucket_count):
            arrivals = int(grouped.get((day, bucket, "inbound"), 0))
            departures = int(grouped.get((day, bucket, "outbound"), 0))
            arrival_violation = arrivals > config.arrival_bucket_capacity
            departure_violation = departures > config.departure_bucket_capacity
            arrival_slack = config.arrival_bucket_capacity - arrivals
            departure_slack = config.departure_bucket_capacity - departures
            rows.append(
                {
                    "day": day,
                    "bucket_id": bucket,
                    "bucket_start_min": bucket * config.bucket_size_min,
                    "bucket_end_min": (bucket + 1) * config.bucket_size_min,
                    "arrivals": arrivals,
                    "departures": departures,
                    "arrival_capacity": config.arrival_bucket_capacity,
                    "departure_capacity": config.departure_bucket_capacity,
                    "arrival_slack": arrival_slack,
                    "departure_slack": departure_slack,
                    "arrival_violation": bool(arrival_violation),
                    "departure_violation": bool(departure_violation),
                    "total_violation": bool(arrival_violation or departure_violation),
                }
            )
    return pd.DataFrame(rows, columns=HUB_CAPACITY_REPORT_COLUMNS)


def build_regularity_groups_from_schedule(optimized_schedule: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Build route-aware same-flight regularity groups from optimized_schedule.xlsx."""

    if optimized_schedule is None or optimized_schedule.empty:
        return pd.DataFrame(columns=REGULARITY_GROUP_COLUMNS)
    required = {"flight_no", "day", "origin", "destination", "old_ist_time", "new_ist_time"}
    if not required.issubset(optimized_schedule.columns):
        return pd.DataFrame(columns=REGULARITY_GROUP_COLUMNS)
    schedule = optimized_schedule.copy()
    schedule["normalized_flight_no"] = schedule["flight_no"].map(normalize_flight_no)
    schedule["origin"] = schedule["origin"].astype(str).str.strip().str.upper()
    schedule["destination"] = schedule["destination"].astype(str).str.strip().str.upper()
    hub = config.hub
    schedule["direction_type"] = ""
    schedule.loc[schedule["destination"] == hub, "direction_type"] = "inbound"
    schedule.loc[schedule["origin"] == hub, "direction_type"] = "outbound"
    schedule["old_ist_time"] = pd.to_datetime(schedule["old_ist_time"], errors="coerce")
    schedule["new_ist_time"] = pd.to_datetime(schedule["new_ist_time"], errors="coerce")
    schedule["old_minute_of_day"] = schedule["old_ist_time"].dt.hour * 60 + schedule["old_ist_time"].dt.minute
    schedule["new_minute_of_day"] = schedule["new_ist_time"].dt.hour * 60 + schedule["new_ist_time"].dt.minute
    schedule = schedule.dropna(subset=["normalized_flight_no", "day", "new_minute_of_day"])
    rows = []
    idx = 0
    group_cols = ["normalized_flight_no", "origin", "destination", "direction_type"]
    for key, group in schedule.groupby(group_cols, dropna=False):
        flight_no, origin, destination, direction = key
        group = group.sort_values(["day", "new_minute_of_day"]).copy()
        days = sorted(group["day"].dropna().astype(int).unique().tolist())
        old_list = [int(x) for x in group["old_minute_of_day"].dropna().round().astype(int).tolist()]
        new_list = [int(x) for x in group["new_minute_of_day"].dropna().round().astype(int).tolist()]
        active = len(days) >= 2 and len(group) >= 2
        min_new = float(group["new_minute_of_day"].min()) if len(group) else None
        max_new = float(group["new_minute_of_day"].max()) if len(group) else None
        # Dairesel yayilim: gece-yarisi sarmali gruplarda dogru (23:50 & 00:20 -> 30, 1410 degil).
        spread = circular_spread(new_list) if new_list else None
        rows.append(
            {
                "regularity_group_id": f"reggrp_{idx:05d}",
                "normalized_flight_no": flight_no,
                "origin": origin,
                "destination": destination,
                "direction_type": direction,
                "days_in_group": ",".join(str(day) for day in days),
                "num_instances": int(len(group)),
                "old_time_min_list": ",".join(str(x) for x in old_list),
                "new_time_min_list": ",".join(str(x) for x in new_list),
                "min_new_time": min_new,
                "max_new_time": max_new,
                "spread_min": spread,
                "regularity_constraint_active": bool(active),
                "skip_reason": "" if active else "single_instance_group",
            }
        )
        idx += 1
    return pd.DataFrame(rows, columns=REGULARITY_GROUP_COLUMNS)


def build_regularity_report_from_schedule(optimized_schedule: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Validate schedule regularity independently from optimized_schedule.xlsx."""

    groups = build_regularity_groups_from_schedule(optimized_schedule, config)
    if groups.empty:
        return pd.DataFrame(columns=REGULARITY_REPORT_COLUMNS)
    rows = []
    for row in groups.itertuples(index=False):
        skip_reason = getattr(row, "skip_reason", "")
        active = bool(getattr(row, "regularity_constraint_active", False)) and not skip_reason
        spread = getattr(row, "spread_min", None)
        old_list = _parse_minute_list(getattr(row, "old_time_min_list", ""))
        old_spread = circular_spread(old_list) if old_list else 0.0
        relax = _regularity_effective_xdev(old_spread, config)
        limit = relax["effective_xdev"]
        new_spread = float(spread) if pd.notna(spread) else None
        over_limit = bool(active and new_spread is not None and new_spread > limit + 1e-6)
        pre_existing = bool(active and old_spread > limit + 1e-6)
        # Hard ihlal YALNIZCA optimizasyonun kotulestirdigi durumdur: yeni yayilim
        # hem limiti hem de veride onceden var olan yayilimi asiyorsa. Veriden gelen
        # (dondurulmus/onarilmis ama hala limit ustu) yayilimlar ayri raporlanir.
        introduced_violation = bool(over_limit and (new_spread > old_spread + 1e-6 or not pre_existing))
        rows.append(
            {
                "regularity_group_id": row.regularity_group_id,
                "normalized_flight_no": row.normalized_flight_no,
                "origin": row.origin,
                "destination": row.destination,
                "direction_type": row.direction_type,
                "days_in_group": row.days_in_group,
                "num_instances": row.num_instances,
                "old_spread_min": old_spread,
                "theoretical_min_spread": relax["theoretical_min_spread"],
                "base_xdev": relax["base_xdev"],
                "effective_xdev": relax["effective_xdev"],
                "relax_applied": relax["relax_applied"],
                "relax_amount": relax["relax_amount"],
                "targeted_max_relax": relax["targeted_max_relax"],
                "targeted_soft": relax["targeted_soft"],
                "min_new_time": row.min_new_time,
                "max_new_time": row.max_new_time,
                "spread_min": row.spread_min,
                "xdev_limit": limit,
                "violation": introduced_violation,
                "over_limit": over_limit,
                "pre_existing_violation": pre_existing,
                "skip_reason": "" if active else skip_reason,
            }
        )
    return pd.DataFrame(rows, columns=REGULARITY_REPORT_COLUMNS)


def _parse_minute_list(value) -> list[float]:
    if pd.isna(value):
        return []
    parsed = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            parsed.append(float(part))
        except ValueError:
            continue
    return parsed


def _regularity_effective_xdev(old_spread: float, config: ModelConfig) -> dict:
    base = float(config.schedule_regular_dev_min)
    tolerance = float(config.regularity_abs_tolerance_min)
    targeted_max = float(config.regularity_targeted_max_relax_min)
    theoretical = max(0.0, float(old_spread) - 2.0 * float(config.max_shift_min))
    mode = str(config.regularity_relax_mode or "none")
    if mode == "global_tolerance":
        effective = base + tolerance
    elif mode == "targeted_theoretical":
        effective = min(max(base, theoretical), targeted_max)
    else:
        effective = base
    return {
        "base_xdev": base,
        "theoretical_min_spread": theoretical,
        "effective_xdev": effective,
        "relax_applied": bool(effective > base + 1e-6),
        "relax_amount": max(0.0, effective - base),
        "targeted_max_relax": targeted_max,
        "targeted_soft": bool(mode == "targeted_soft" and theoretical > base + 1e-6),
        "still_infeasible_by_theoretical_bound": bool(theoretical > effective + 1e-6),
    }


def _rotation_report_skip_row(candidate, reason: str) -> dict:
    return {
        "pair_id": getattr(candidate, "pair_id", None),
        "outbound_flight_no": getattr(candidate, "outbound_flight_no", None),
        "inbound_flight_no": getattr(candidate, "inbound_flight_no", None),
        "new_outbound_departure_time": pd.NaT,
        "new_inbound_arrival_time": pd.NaT,
        "outbound_block_time_min": getattr(candidate, "outbound_block_time_min", None),
        "inbound_block_time_min": getattr(candidate, "inbound_block_time_min", None),
        "min_ground_time_min": getattr(candidate, "min_ground_time_min", None),
        "required_delta_min": getattr(candidate, "required_delta_min", None),
        "actual_delta_min": None,
        "slack_min": None,
        "violation": False,
        "skip_reason": reason,
    }


def format_report(report: FeasibilityReport) -> str:
    """Render checker output."""

    lines = [
        "FEASIBILITY REPORT",
        "------------------",
        f"Connection window violations: {report.connection_window_violations}",
        f"Rotation violations: {report.rotation_violations}",
        f"Hub capacity violations: {report.hub_capacity_violations}",
        f"Schedule regularity violations: {report.regularity_violations}",
        f"Regularity pre-existing (data) violations, times unchanged/improved: {report.regularity_pre_existing_violations}",
        f"Regularity relax mode: {report.regularity_relax_mode}",
        f"Relax applied groups: {report.regularity_relax_applied_groups}",
        f"Regularity hard violations: {report.regularity_violations}",
        f"Regularity soft violations: {report.regularity_soft_violations}",
        f"Regularity soft violation amount total: {report.regularity_soft_violation_amount_total}",
        f"Max effective XDEV: {report.regularity_max_effective_xdev}",
        f"Directional balance violations: {report.directional_balance_violations}",
        f"Directional E1 violations: {report.directional_e1_violations}",
        f"Directional E2 violations: {report.directional_e2_violations}",
        f"Hard violations: {report.hard_violations}",
        f"Directional E1 hard violations: {report.directional_e1_hard_violations}",
        f"Directional E1 soft violations: {report.directional_e1_soft_violations}",
        f"Directional E1 soft violation amount total: {report.directional_e1_soft_violation_amount_total}",
        f"Directional E2 hard violations: {report.directional_e2_hard_violations}",
        f"Directional E2 soft violations: {report.directional_e2_soft_violations}",
        f"Directional E2 soft violation amount total: {report.directional_e2_soft_violation_amount_total}",
        f"Feasible but inactive connections: {report.feasible_but_inactive_connections}",
        f"Infeasible but active connections: {report.infeasible_but_active_connections}",
        f"Mandatory feasible-but-inactive: {report.feasible_but_inactive_connections}",
        f"Mandatory infeasible-but-active: {report.infeasible_but_active_connections}",
        f"Rotation enabled: {report.rotation_enabled}",
        f"Capacity enabled: {report.capacity_enabled}",
        f"Regularity hard-check enabled: {report.regularity_enabled}"
        + ("" if report.regularity_enabled else "  (regularity SOFT modda modellenmistir: ihlaller 'Regularity soft violations' altinda raporlanir, sert ihlal sayilmaz)"),
        f"Directional enabled: {report.directional_enabled}",
        f"Objective difference: {report.objective_difference:.4f}",
        f"Missing/NaN values: {report.missing_values}",
        "",
        f"Final feasible status: {report.feasible}",
        f"Result: {'FEASIBLE' if report.feasible else 'INFEASIBLE'}",
    ]
    return "\n".join(lines)


def check_run_output(run_dir: Path, config: ModelConfig) -> tuple[FeasibilityReport, str]:
    """Load output workbooks and check a run directory."""

    flags = _flags_from_run_summary(run_dir)
    config = _config_from_run_summary(config, run_dir)
    bucket_path = run_dir / "hub_buckets.xlsx"
    if bucket_path.exists():
        config = _config_from_hub_buckets(config, bucket_path)
    regularity_path = run_dir / "regularity_report.xlsx"
    regularity_enabled = (run_dir / "regularity_groups.xlsx").exists()
    if regularity_path.exists():
        regularity_enabled = regularity_enabled or _is_regularity_validation_report(regularity_path)
    if regularity_enabled and regularity_path.exists():
        config = _config_from_regularity_report(config, regularity_path)
    directional_path = run_dir / "directional_balance_report.xlsx"
    directional_pairs_path = run_dir / "directional_pairs.xlsx"
    directional_enabled = directional_pairs_path.exists()
    if directional_enabled and directional_path.exists():
        config = _config_from_directional_report(config, directional_path)

    selected = pd.read_excel(run_dir / "selected_connections.xlsx", engine="openpyxl")
    od_metrics = pd.read_excel(run_dir / "od_metrics.xlsx", engine="openpyxl")
    rotation_path = run_dir / "rotation_report.xlsx"
    schedule_path = run_dir / "optimized_schedule.xlsx"
    candidates_path = run_dir / "rotation_candidates.xlsx"
    if schedule_path.exists() and candidates_path.exists():
        schedule = pd.read_excel(schedule_path, engine="openpyxl")
        candidates = pd.read_excel(candidates_path, engine="openpyxl")
        rotation_report = build_rotation_report_from_tables(schedule, candidates, config)
        rotation_report.to_excel(rotation_path, index=False)
    else:
        rotation_report = pd.read_excel(rotation_path, engine="openpyxl") if rotation_path.exists() else None
    hub_path = run_dir / "hub_capacity_report.xlsx"
    if schedule_path.exists():
        schedule = pd.read_excel(schedule_path, engine="openpyxl")
        hub_capacity_report = build_hub_capacity_report_from_schedule(schedule, config)
        hub_capacity_report.to_excel(hub_path, index=False)
        build_hub_buckets(config).to_excel(run_dir / "hub_buckets.xlsx", index=False)
    else:
        hub_capacity_report = pd.read_excel(hub_path, engine="openpyxl") if hub_path.exists() else None
    if schedule_path.exists() and regularity_enabled:
        schedule = pd.read_excel(schedule_path, engine="openpyxl")
        regularity_groups = build_regularity_groups_from_schedule(schedule, config)
        regularity_groups.to_excel(run_dir / "regularity_groups.xlsx", index=False)
        regularity_report = build_regularity_report_from_schedule(schedule, config)
        regularity_report.to_excel(regularity_path, index=False)
    else:
        regularity_report = pd.read_excel(regularity_path, engine="openpyxl") if regularity_enabled and regularity_path.exists() else None
    if directional_enabled:
        directional_pairs = pd.read_excel(directional_pairs_path, engine="openpyxl")
        directional_balance_report = build_directional_balance_report(od_metrics, config, directional_pairs)
        directional_balance_report.to_excel(directional_path, index=False)
    else:
        directional_balance_report = None
    all_candidate_path = run_dir / "all_candidate_connections.xlsx"
    feasible_but_inactive_report = None
    if all_candidate_path.exists() and schedule_path.exists():
        all_candidate_connections = pd.read_excel(all_candidate_path, engine="openpyxl")
        if "mandatory_selection_enabled" in all_candidate_connections.columns:
            enabled = bool(all_candidate_connections["mandatory_selection_enabled"].fillna(False).astype(bool).any())
            config = replace(config, mandatory_connection_selection=enabled)
        schedule = pd.read_excel(schedule_path, engine="openpyxl")
        feasible_but_inactive_report = build_feasible_but_inactive_report(
            all_candidate_connections,
            selected,
            schedule,
            config,
        )
        feasible_but_inactive_report.to_excel(run_dir / "feasible_but_inactive_connections.xlsx", index=False)
    objective = float((run_dir / "objective_value.txt").read_text(encoding="utf-8").split("total objective:")[1].splitlines()[0].strip())
    report = check_solution_tables(
        selected,
        od_metrics,
        objective,
        config,
        rotation_report,
        hub_capacity_report,
        regularity_report,
        directional_balance_report,
        feasible_but_inactive_report,
        rotation_enabled=flags.get("rotation_enabled", True),
        capacity_enabled=flags.get("capacity_enabled", True),
        regularity_enabled=flags.get("regularity_enabled", True),
        directional_enabled=flags.get("directional_enabled", True),
    )
    text = format_report(report)
    (run_dir / "feasibility_report.txt").write_text(text, encoding="utf-8")
    _write_penalty_validation_report(run_dir)
    return report, text


def _flags_from_run_summary(run_dir: Path) -> dict[str, bool]:
    path = run_dir / "run_summary.txt"
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"rotation_enabled", "capacity_enabled", "regularity_enabled", "directional_enabled"}:
            values[key] = value.strip().lower() in {"true", "1", "yes"}
    return values


def _config_from_run_summary(config: ModelConfig, run_dir: Path) -> ModelConfig:
    path = run_dir / "run_summary.txt"
    if not path.exists():
        return config
    values = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    updates = {}
    int_fields = {
        "regularity_xdev_minutes": "schedule_regular_dev_min",
        "regularity_abs_tolerance": "regularity_abs_tolerance_min",
        "regularity_targeted_max_relax": "regularity_targeted_max_relax_min",
        "max_shift_min": "max_shift_min",
    }
    float_fields = {
        "regularity_penalty_weight": "regularity_penalty_weight",
        "directional_e1_penalty_weight": "directional_e1_penalty_weight",
        "directional_e2_penalty_weight": "directional_e2_penalty_weight",
        "directional_alpha": "directional_alpha",
    }
    str_fields = {
        "regularity_mode": "regularity_mode",
        "regularity_relax_mode": "regularity_relax_mode",
        "directional_e1_mode": "directional_e1_mode",
        "directional_e2_mode": "directional_e2_mode",
        "directional_e1_formula": "directional_e1_formula",
    }
    for key, field in int_fields.items():
        if key in values and values[key] != "":
            try:
                updates[field] = int(float(values[key]))
            except ValueError:
                pass
    for key, field in float_fields.items():
        if key in values and values[key] != "":
            try:
                updates[field] = float(values[key])
            except ValueError:
                pass
    for key, field in str_fields.items():
        if key in values and values[key] != "":
            updates[field] = values[key]
    return replace(config, **updates) if updates else config


def _write_penalty_validation_report(run_dir: Path) -> None:
    penalty_path = run_dir / "shift_penalty_report.xlsx"
    objective_path = run_dir / "objective_value.txt"
    if not penalty_path.exists() or not objective_path.exists():
        return
    try:
        penalty = pd.read_excel(penalty_path, engine="openpyxl")
    except Exception as exc:
        (run_dir / "penalty_validation_report.txt").write_text(
            "\n".join(
                [
                    "PENALTY VALIDATION REPORT",
                    "-------------------------",
                    f"warning: could not read shift_penalty_report.xlsx: {exc}",
                ]
            ),
            encoding="utf-8",
        )
        return
    objective_text = objective_path.read_text(encoding="utf-8")
    written_before = _objective_value_from_text(objective_text, "objective before penalty")
    written_after = _objective_value_from_text(objective_text, "objective after penalty")
    if written_before is None:
        written_before = _objective_value_from_text(objective_text, "total objective") or 0.0
    if penalty.empty:
        total_abs = 0.0
        total_excess = 0.0
        total_penalty = 0.0
    else:
        total_abs = float(pd.to_numeric(penalty.get("abs_shift_min", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        total_excess = float(pd.to_numeric(penalty.get("excess_shift_min", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        total_penalty = float(pd.to_numeric(penalty.get("penalty_contribution", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    recalculated_after = float(written_before) - total_penalty
    diff = recalculated_after - float(written_after if written_after is not None else recalculated_after)
    warning = "none" if abs(diff) <= 1e-4 else "objective_after_penalty mismatch"
    lines = [
        "PENALTY VALIDATION REPORT",
        "-------------------------",
        f"recalculated_total_abs_shift: {total_abs}",
        f"recalculated_total_excess_shift: {total_excess}",
        f"recalculated_total_shift_penalty: {total_penalty}",
        f"objective_before_penalty: {written_before}",
        f"objective_after_penalty: {written_after if written_after is not None else recalculated_after}",
        f"recalculated_objective_after_penalty: {recalculated_after}",
        f"objective_after_penalty_difference: {diff}",
        f"warning: {warning}",
    ]
    (run_dir / "penalty_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")


def _objective_value_from_text(text: str, key: str) -> float | None:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.strip().lower().startswith(prefix):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _config_from_hub_buckets(config: ModelConfig, bucket_path: Path) -> ModelConfig:
    """Reuse run-specific hub bucket/capacity settings when re-checking outputs."""

    try:
        buckets = pd.read_excel(bucket_path, engine="openpyxl")
    except Exception:
        return config
    if buckets.empty:
        return config
    bucket_size = config.bucket_size_min
    if {"bucket_start_min", "bucket_end_min"}.issubset(buckets.columns):
        sizes = (buckets["bucket_end_min"] - buckets["bucket_start_min"]).dropna()
        if len(sizes):
            bucket_size = int(sizes.iloc[0])
    arrival_capacity = config.arrival_bucket_capacity
    departure_capacity = config.departure_bucket_capacity
    if "arrival_capacity" in buckets.columns and buckets["arrival_capacity"].notna().any():
        arrival_capacity = int(buckets["arrival_capacity"].dropna().iloc[0])
    if "departure_capacity" in buckets.columns and buckets["departure_capacity"].notna().any():
        departure_capacity = int(buckets["departure_capacity"].dropna().iloc[0])
    return replace(
        config,
        bucket_size_min=bucket_size,
        arrival_bucket_capacity=arrival_capacity,
        departure_bucket_capacity=departure_capacity,
    )


def _config_from_regularity_report(config: ModelConfig, regularity_path: Path) -> ModelConfig:
    try:
        report = pd.read_excel(regularity_path, engine="openpyxl")
    except Exception:
        return config
    if report.empty or "base_xdev" not in report.columns or not report["base_xdev"].notna().any():
        return config
    return replace(config, schedule_regular_dev_min=int(report["base_xdev"].dropna().iloc[0]))


def _config_from_directional_report(config: ModelConfig, directional_path: Path) -> ModelConfig:
    try:
        report = pd.read_excel(directional_path, engine="openpyxl")
    except Exception:
        return config
    if report.empty:
        return config
    alpha = config.directional_alpha
    gamma = config.directional_gamma_min
    if "alpha" in report.columns and report["alpha"].notna().any():
        alpha = float(report["alpha"].dropna().iloc[0])
    if "gamma" in report.columns and report["gamma"].notna().any():
        gamma = int(report["gamma"].dropna().iloc[0])
    e2_mode = config.directional_e2_mode
    e1_mode = config.directional_e1_mode
    e1_formula = config.directional_e1_formula
    e1_abs_tolerance = config.directional_e1_abs_tolerance
    allow_soft = config.allow_soft_directional_violations
    if "directional_e1_mode" in report.columns and report["directional_e1_mode"].notna().any():
        e1_mode = str(report["directional_e1_mode"].dropna().iloc[0])
    if "e1_formula" in report.columns and report["e1_formula"].notna().any():
        e1_formula = str(report["e1_formula"].dropna().iloc[0])
    if "e1_abs_tolerance" in report.columns and report["e1_abs_tolerance"].notna().any():
        e1_abs_tolerance = int(report["e1_abs_tolerance"].dropna().iloc[0])
    if "directional_e2_mode" in report.columns and report["directional_e2_mode"].notna().any():
        e2_mode = str(report["directional_e2_mode"].dropna().iloc[0])
    if "allow_soft_directional_violations" in report.columns and report["allow_soft_directional_violations"].notna().any():
        value = report["allow_soft_directional_violations"].dropna().iloc[0]
        allow_soft = str(value).strip().lower() in {"true", "1", "yes"}
    return replace(
        config,
        directional_alpha=alpha,
        directional_gamma_min=gamma,
        directional_e1_mode=e1_mode,
        directional_e1_formula=e1_formula,
        directional_e1_abs_tolerance=e1_abs_tolerance,
        directional_e2_mode=e2_mode,
        allow_soft_directional_violations=allow_soft,
    )


def _is_regularity_validation_report(regularity_path: Path) -> bool:
    try:
        report = pd.read_excel(regularity_path, engine="openpyxl", nrows=1)
    except Exception:
        return False
    return "regularity_group_id" in report.columns and not report.empty
