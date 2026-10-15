"""Pyomo model builders for baseline, MVP, and full levels."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd

from .config import ModelConfig
from .preprocessing import PreparedData


def circular_spread(minutes) -> float:
    """Minimum enclosing arc of a set of minute-of-day points on a 1440-min circle.

    Gece-yarisi sarmasi icin dogru olcum: 23:50 ve 00:20 arasindaki gercek
    yayilim 30 dk'dir, duz (max-min) hesapta 1410 cikar. Sirali noktalar
    arasindaki en buyuk boslugun (wrap dahil) tumleyeni minimal yayilimi verir.
    Tarife duzenliligi (G) icin ikili dairesel mesafelerin hepsinin bu yayilimla
    sinirli oldugu garanti edilir (yayilim <= 720 iken).
    """

    pts = sorted({int(round(float(m))) % 1440 for m in minutes if pd.notna(m)})
    if len(pts) <= 1:
        return 0.0
    gaps = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
    gaps.append(pts[0] + 1440 - pts[-1])
    return float(1440 - max(gaps))


@dataclass
class ModelArtifacts:
    """Pyomo model plus mapping tables needed to decode a solution."""

    model: Any
    candidate_connections: pd.DataFrame
    flight_events: pd.DataFrame
    rotation_candidates: pd.DataFrame
    rotation_report: pd.DataFrame
    bucket_table: pd.DataFrame
    event_bucket_pairs: list[tuple[str, int]]
    regularity_report: pd.DataFrame
    directional_pairs: pd.DataFrame
    od_day_keys: list[str]
    connection_slots: dict[str, list[int]]


def build_model(
    data: PreparedData,
    config: ModelConfig,
    model_level: str = "mvp",
    fixed_event_shifts: dict[str, float] | None = None,
    anchor_references: dict[tuple[str, str, str], list[float]] | None = None,
) -> ModelArtifacts:
    """Build the requested model level."""

    if model_level == "baseline":
        raise ValueError("Baseline does not build a Pyomo model; use baseline.compute_baseline.")
    if model_level == "rotation":
        print("[INFO] Rotation model requested; using MVP core with rotation constraints enabled.")
    if model_level == "capacity":
        print("[INFO] Capacity model requested; using MVP core with rotation and hub capacity constraints enabled.")
    if model_level == "regularity":
        print("[INFO] Regularity model requested; using MVP core with rotation, hub capacity, and schedule regularity constraints enabled.")
    if model_level == "directional":
        print("[INFO] Directional model requested; using MVP core with rotation, hub capacity, schedule regularity, and directional balance constraints enabled.")
    if model_level == "full":
        print("[INFO] Full model requested; using MVP core with full-model extension hooks enabled.")
    if model_level == "final":
        print("[INFO] Final model requested; using MVP core with rotation, hub capacity, schedule regularity, directional balance, and shift penalty enabled.")
    return build_mvp_model(
        data,
        config,
        enable_rotation_constraints=model_level in {"rotation", "capacity", "regularity", "directional", "full", "final"},
        enable_hub_capacity_constraints=model_level in {"capacity", "regularity", "directional", "full", "final"},
        enable_regularity_constraints=model_level in {"regularity", "directional", "full", "final"},
        enable_directional_constraints=model_level in {"directional", "full", "final"},
        fixed_event_shifts=fixed_event_shifts,
        anchor_references=anchor_references,
    )


def build_mvp_model(
    data: PreparedData,
    config: ModelConfig,
    enable_rotation_constraints: bool = False,
    enable_hub_capacity_constraints: bool = False,
    enable_regularity_constraints: bool = False,
    enable_directional_constraints: bool = False,
    fixed_event_shifts: dict[str, float] | None = None,
    anchor_references: dict[tuple[str, str, str], list[float]] | None = None,
) -> ModelArtifacts:
    """Build a compact MIP over existing TK candidate connections.

    MVP assumptions:
    - Candidate TK connections are existing rows from the connection table.
    - Each unique inbound/outbound IST event receives an integer minute shift.
    - Journey time is updated as old journey time plus the change in IST connection gap.
    - Flight-pair rotation constraints are applied for the rotation/full model levels when pair legs can be inferred.
    - IST hub capacity is enforced with 10-minute bucket assignment variables.
    - Schedule regularity is enforced for same flight_no/direction across days.
    - Directional balance is enforced for O-D/reverse O-D pairs in the selected subset.
    - Objective includes weighted diminishing connection reward plus a simple competitor beating reward.
    """

    print("[INFO] Building Pyomo MVP model...")
    try:
        import pyomo.environ as pyo
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pyomo is not installed. Run `pip install -r requirements.txt` before MVP/full runs.") from exc
    candidates = data.tk_connections.copy().reset_index(drop=True)
    candidates["old_gap_min"] = candidates["connection_gap_min"].astype(float)
    candidates["min_possible_gap"] = candidates["old_gap_min"] - 2 * config.max_shift_min
    candidates["max_possible_gap"] = candidates["old_gap_min"] + 2 * config.max_shift_min
    candidates = candidates[
        (candidates["min_possible_gap"] <= config.max_connection_min)
        & (candidates["max_possible_gap"] >= config.min_connection_min)
    ].copy()
    candidates["candidate_id"] = [f"cand_{i:05d}" for i in range(len(candidates))]
    candidates["od_day_key"] = candidates["O&D"] + "|" + candidates["day"].astype(str)
    candidates["inbound_flight_norm"] = candidates["inbound_flight"].map(_normalize_flight_no)
    candidates["outbound_flight_norm"] = candidates["outbound_flight"].map(_normalize_flight_no)
    # Event gunu Gün kolonundan degil gercek minute-of-week'ten turetilir; hafta
    # sarmali baglantilarda ayni fiziksel ucus tek bir event/shift degiskeni alir.
    candidates["inbound_event_day"] = (candidates["arr_min"].astype(int) // 1440).clip(lower=1, upper=7)
    candidates["outbound_event_day"] = (candidates["dep_min"].astype(int) // 1440).clip(lower=1, upper=7)
    candidates["inbound_event_id"] = (
        "in|"
        + candidates["inbound_event_day"].astype(str)
        + "|"
        + candidates["inbound_flight_norm"].astype(str)
        + "|"
        + candidates["origin"].astype(str)
        + "|"
        + candidates["arr_min"].astype(str)
    )
    candidates["outbound_event_id"] = (
        "out|"
        + candidates["outbound_event_day"].astype(str)
        + "|"
        + candidates["outbound_flight_norm"].astype(str)
        + "|"
        + candidates["destination"].astype(str)
        + "|"
        + candidates["dep_min"].astype(str)
    )

    od_day = data.od_day.copy()
    od_day["od_day_key"] = od_day["od"] + "|" + od_day["day"].astype(str)
    od_day_keys = od_day["od_day_key"].tolist()
    if candidates.empty:
        raise ValueError("No TK candidate connections remain after preprocessing filters.")

    flight_events = _build_flight_events(candidates)
    flight_events, shift_key_of, frozen_shift_keys, shift_granularity_stats = _assign_shift_keys(flight_events, config)
    rotation_candidates = _build_rotation_candidates(flight_events, data.flight_pairs, config, data.raw_connections)
    rotation_report = rotation_candidates.copy()
    fixed_tk_events = _build_fixed_tk_events(data, flight_events)
    hub_postcheck = enable_hub_capacity_constraints and config.hub_capacity_mode == "postcheck"
    if hub_postcheck:
        # Kova-atama makinesi (binary + big-M kisitlar) cozum sirasinda kurulmaz;
        # kapasite bagimsiz checker ile dogrulanir. Model ~70k binary + ~140k
        # big-M kisittan kurtulur -> LP gevsemesi cok daha siki, cozum tractable.
        bucket_table = pd.DataFrame(columns=["day", "bucket", "direction", "capacity"])
        event_bucket_pairs = []
        feasible_buckets_by_event = {}
        print(
            "[INFO] Hub capacity mode=postcheck: bucket-assignment machinery skipped during solve; "
            "capacity verified independently by the checker (residual capacity rarely binds)."
        )
    else:
        bucket_table, event_bucket_pairs, feasible_buckets_by_event = _build_bucket_data(
            flight_events, config, fixed_tk_events if enable_hub_capacity_constraints else None
        )
    anchor_bounds = _build_regularity_anchor_bounds(
        flight_events,
        data,
        config,
        enabled=enable_regularity_constraints and config.regularity_mode == "hard",
        explicit_references=anchor_references,
    )
    regularity_report = _build_regularity_report(flight_events, config)
    if config.shift_granularity == "flight_week" and not regularity_report.empty:
        # flight_week modunda ayni ucusun tum gunleri tek shift aldigi icin gunler-arasi
        # sapma sabit kalir: duzenli gruplar yapisal olarak saglanir, event-bazli cozulen
        # duzensiz gruplarin satirlari kisit olarak kalir, duzeltilemeyecek gruplar
        # dondurulur ve pre-existing veri ihlali olarak etiketlenir.
        event_level_ids = set(flight_events.loc[flight_events["shift_key"] == flight_events["event_id"], "event_id"])
        frozen_event_ids = set(
            flight_events.loc[flight_events["shift_key"].isin(frozen_shift_keys), "event_id"]
        )
        both_event_level = regularity_report["event_id_a"].isin(event_level_ids) & regularity_report["event_id_b"].isin(event_level_ids)
        any_frozen = regularity_report["event_id_a"].isin(frozen_event_ids) | regularity_report["event_id_b"].isin(frozen_event_ids)
        regularity_report.loc[~both_event_level, "status"] = "auto_satisfied_flight_week_shift"
        regularity_report.loc[any_frozen, "status"] = "pre_existing_data_violation_frozen"
        regularity_report.loc[any_frozen, "warning"] = (
            "input schedule already exceeds weekly XDEV beyond repair range; events frozen at original times"
        )
    directional_pairs = _build_directional_pairs(od_day_keys)
    comp_best = data.competitor_connections.groupby(["O&D", "day"])["journey_time_min"].min().to_dict()
    best_comp_by_candidate = {}
    for _, row in candidates.iterrows():
        key = (row["O&D"], int(row["day"]))
        best_comp_by_candidate[row["candidate_id"]] = float(comp_best.get(key, float("inf")))

    candidate_by_key = {
        key: group["candidate_id"].tolist()
        for key, group in candidates.groupby("od_day_key")
    }
    # Sikilastirilmis big-M: her O-D/gun icin ulasilabilir en buyuk journey time.
    candidates["max_new_journey_min"] = (
        candidates["journey_time_min"].astype(float)
        + candidates["max_possible_gap"].astype(float)
        - candidates["old_gap_min"].astype(float)
    )
    key_max_journey = candidates.groupby("od_day_key")["max_new_journey_min"].max().to_dict()
    key_max_journey = {key: max(1.0, float(key_max_journey.get(key, 0.0))) for key in od_day_keys}
    # Onisleme icin guvenli alt sinir: aktif bir baglantinin journey suresi en az
    # (bacak sablon suresi + L) olur (gap >= L). Bir pazarin ulasabilecegi en kucuk
    # journey = min_c(leg_sum_c) + L. Bu sinirdan hizli olan rakip ASLA yenilemez.
    candidates["leg_sum_min"] = candidates["journey_time_min"].astype(float) - candidates["old_gap_min"].astype(float)
    key_min_journey = candidates.groupby("od_day_key")["leg_sum_min"].min().to_dict()
    key_min_journey = {key: float(key_min_journey.get(key, float("-inf"))) + config.min_connection_min for key in od_day_keys}
    slots = {
        key: list(range(1, min(config.max_connection_slots, len(candidate_by_key.get(key, []))) + 1))
        for key in od_day_keys
    }
    importance = od_day.set_index("od_day_key")["importance_factor"].astype(float).to_dict()
    initial_rank_by_key = od_day.set_index("od_day_key")["initial_rank"].astype(int).to_dict()
    competitor_count_by_key = od_day.set_index("od_day_key")["competitor_count"].astype(int).to_dict()
    # Onisleme: rakip yenme binary'si YALNIZCA (a) TK adayi olan pazarlar ve
    # (b) yenilebilir olan rakipler icin kurulur. Adaysiz pazarda key_active=0
    # oldugundan tum beaten binary'leri zaten 0'dir; asla-yenilemez rakipler de
    # sabit 0'dir. Ikisini de atmak model boyutunu ciddi kuculttur (model dogrulugu
    # degismez: bu binary'ler zaten kararsiz degildi).
    keys_with_candidates = {key for key, ids in candidate_by_key.items() if ids}
    lookup_skipped_no_candidate = 0
    lookup_skipped_never_beatable = 0
    lookup_competitors = []
    for idx, row in data.competitor_connections.iterrows():
        key = f"{row['O&D']}|{int(row['day'])}"
        if key not in od_day_keys:
            continue
        if key not in keys_with_candidates:
            lookup_skipped_no_candidate += 1
            continue
        comp_time = float(row["journey_time_min"])
        if comp_time < key_min_journey.get(key, float("-inf")) - 1e-9:
            lookup_skipped_never_beatable += 1
            continue
        lookup_competitors.append(
            {
                "lookup_competitor_id": f"lcomp_{idx:06d}",
                "od_day_key": key,
                "journey_time_min": comp_time,
            }
        )
    if lookup_skipped_no_candidate or lookup_skipped_never_beatable:
        print(
            "[INFO] Lookup-competitor preprocessing: dropped "
            f"{lookup_skipped_no_candidate} (no TK candidate market) + "
            f"{lookup_skipped_never_beatable} (never beatable) determinable binaries; "
            f"{len(lookup_competitors)} genuine competitor binaries remain."
        )
    lookup_competitors_by_key: dict[str, list[str]] = {}
    lookup_competitor_time = {}
    for row in lookup_competitors:
        lookup_competitors_by_key.setdefault(row["od_day_key"], []).append(row["lookup_competitor_id"])
        lookup_competitor_time[row["lookup_competitor_id"]] = row["journey_time_min"]
    lookup_competitor_key = {row["lookup_competitor_id"]: row["od_day_key"] for row in lookup_competitors}
    lookup_levels = []
    lookup_reward_by_level = {}
    for key in od_day_keys:
        competitor_count = int(competitor_count_by_key.get(key, 0))
        initial_rank = int(initial_rank_by_key.get(key, 1))
        for beaten_count in range(competitor_count + 1):
            new_rank = max(1, competitor_count - beaten_count + 1)
            lookup_levels.append((key, beaten_count))
            lookup_reward_by_level[(key, beaten_count)] = float(
                data.ranking_lookup.get(
                    (
                        competitor_count,
                        min(max(1, initial_rank), max(1, competitor_count)),
                        min(max(1, new_rank), max(1, competitor_count)),
                    ),
                    0.0,
                )
            )

    m = pyo.ConcreteModel("teknofest_airline_mvp")
    m.C = pyo.Set(initialize=candidates["candidate_id"].tolist(), ordered=True)
    m.R = pyo.Set(initialize=flight_events["event_id"].tolist(), ordered=True)
    m.K = pyo.Set(initialize=od_day_keys, ordered=True)
    m.x = pyo.Var(m.C, domain=pyo.Binary)
    if config.mandatory_connection_selection:
        m.low = pyo.Var(m.C, domain=pyo.Binary)
        m.high = pyo.Var(m.C, domain=pyo.Binary)
    m.beat = pyo.Var(m.C, domain=pyo.Binary)
    m.key_active = pyo.Var(m.K, domain=pyo.Binary)
    m.best_conn = pyo.Var(m.C, domain=pyo.Binary)
    m.best_time = pyo.Var(m.K, domain=pyo.NonNegativeReals, bounds=lambda _, key: (0, key_max_journey[key]))
    m.LCOMP = pyo.Set(initialize=[row["lookup_competitor_id"] for row in lookup_competitors], ordered=True)
    m.LR = pyo.Set(initialize=lookup_levels, dimen=2, ordered=True)
    m.lookup_competitor_beaten = pyo.Var(m.LCOMP, domain=pyo.Binary)
    m.lookup_level = pyo.Var(m.LR, domain=pyo.Binary)
    shift_keys = sorted(set(shift_key_of.values()))
    m.SH = pyo.Set(initialize=shift_keys, ordered=True)
    m.shift = pyo.Var(m.SH, domain=pyo.Integers, bounds=(-config.max_shift_min, config.max_shift_min))
    m.event_shift = pyo.Expression(m.R, rule=lambda model, event_id: model.shift[shift_key_of[event_id]])
    for frozen_key in frozen_shift_keys:
        if frozen_key in shift_key_of.values():
            m.shift[frozen_key].setlb(0)
            m.shift[frozen_key].setub(0)
    if anchor_bounds:
        for event_id, (anchor_lb, anchor_ub) in anchor_bounds.items():
            var = m.shift[shift_key_of[event_id]]
            var.setlb(max(var.lb if var.lb is not None else -config.max_shift_min, anchor_lb))
            var.setub(min(var.ub if var.ub is not None else config.max_shift_min, anchor_ub))
        print(
            "[INFO] Applied out-of-scope regularity anchor bounds to "
            f"{len(anchor_bounds)} adjustable events (weekly XDEV guarantee)."
        )
    if fixed_event_shifts:
        frozen = 0
        event_id_set = set(flight_events["event_id"].values)
        for event_id, value in fixed_event_shifts.items():
            if event_id in event_id_set:
                var = m.shift[shift_key_of[event_id]]
                var.setlb(value)
                var.setub(value)
                frozen += 1
        if frozen:
            print(f"[INFO] Froze {frozen} event shifts from previously solved days (sequential mode).")
    bound_conflicts = 0
    for sh in shift_keys:
        var = m.shift[sh]
        if var.lb is not None and var.ub is not None and var.lb > var.ub:
            var.setlb(0)
            var.setub(0)
            bound_conflicts += 1
    if bound_conflicts:
        print(
            f"[WARN] {bound_conflicts} shift variables had conflicting anchor bounds; "
            "frozen at original times and reported."
        )
    m.abs_shift = pyo.Var(m.SH, domain=pyo.NonNegativeReals, bounds=(0, config.max_shift_min))
    m.excess_shift = pyo.Var(m.SH, domain=pyo.NonNegativeReals, bounds=(0, config.max_shift_min))
    m.shift_seg_0_5 = pyo.Var(m.SH, domain=pyo.NonNegativeReals, bounds=(0, min(5, config.max_shift_min)))
    m.shift_seg_5_15 = pyo.Var(m.SH, domain=pyo.NonNegativeReals, bounds=(0, max(0, min(15, config.max_shift_min) - 5)))
    m.shift_seg_15_plus = pyo.Var(m.SH, domain=pyo.NonNegativeReals, bounds=(0, max(0, config.max_shift_min - 15)))
    gap_bounds = {
        row.candidate_id: (float(row.min_possible_gap), float(row.max_possible_gap))
        for row in candidates.itertuples(index=False)
    }
    m.gap = pyo.Var(m.C, domain=pyo.Reals, bounds=lambda _, c: gap_bounds[c])
    active_rotations = rotation_candidates[
        (rotation_candidates["rotation_constraint_available"] == True)
        & (rotation_candidates["status"] == "constrained")
    ].copy() if enable_rotation_constraints else pd.DataFrame(columns=rotation_candidates.columns)
    rotation_rows = active_rotations.set_index("rotation_id").to_dict("index") if not active_rotations.empty else {}
    m.Q = pyo.Set(initialize=active_rotations["rotation_id"].tolist(), ordered=True)
    m.EB = pyo.Set(initialize=event_bucket_pairs, dimen=2, ordered=True)
    m.bucket_assign = pyo.Var(m.EB, domain=pyo.Binary)
    slot_index = [(key, slot) for key, slot_list in slots.items() for slot in slot_list]
    m.S = pyo.Set(initialize=slot_index, dimen=2, ordered=True)
    m.z = pyo.Var(m.S, domain=pyo.Binary)

    candidate_rows = candidates.set_index("candidate_id").to_dict("index")

    def _abs_shift_pos_rule(model, sh):
        return model.abs_shift[sh] >= model.shift[sh]

    def _abs_shift_neg_rule(model, sh):
        return model.abs_shift[sh] >= -model.shift[sh]

    def _excess_shift_rule(model, sh):
        return model.excess_shift[sh] >= model.abs_shift[sh] - config.shift_free_threshold_min

    def _shift_segment_sum_rule(model, sh):
        return model.abs_shift[sh] == model.shift_seg_0_5[sh] + model.shift_seg_5_15[sh] + model.shift_seg_15_plus[sh]

    m.abs_shift_pos = pyo.Constraint(m.SH, rule=_abs_shift_pos_rule)
    m.abs_shift_neg = pyo.Constraint(m.SH, rule=_abs_shift_neg_rule)
    m.excess_shift_floor = pyo.Constraint(m.SH, rule=_excess_shift_rule)
    m.shift_segment_sum = pyo.Constraint(m.SH, rule=_shift_segment_sum_rule)

    def _gap_definition_rule(model, c):
        row = candidate_rows[c]
        return model.gap[c] == float(row["old_gap_min"]) + model.event_shift[row["outbound_event_id"]] - model.event_shift[row["inbound_event_id"]]

    m.gap_definition = pyo.Constraint(m.C, rule=_gap_definition_rule)

    def _active_gap_lower_rule(model, c):
        row = candidate_rows[c]
        big_m = max(0.0, config.min_connection_min - float(row["min_possible_gap"]))
        return model.gap[c] >= config.min_connection_min - big_m * (1 - model.x[c])

    def _active_gap_upper_rule(model, c):
        row = candidate_rows[c]
        big_m = max(0.0, float(row["max_possible_gap"]) - config.max_connection_min)
        return model.gap[c] <= config.max_connection_min + big_m * (1 - model.x[c])

    m.active_gap_lower = pyo.Constraint(m.C, rule=_active_gap_lower_rule)
    m.active_gap_upper = pyo.Constraint(m.C, rule=_active_gap_upper_rule)

    if config.mandatory_connection_selection:
        epsilon = 1.0

        def _connection_state_rule(model, c):
            return model.x[c] + model.low[c] + model.high[c] == 1

        def _low_gap_upper_rule(model, c):
            row = candidate_rows[c]
            big_m = max(0.0, float(row["max_possible_gap"]) - (config.min_connection_min - epsilon))
            return model.gap[c] <= (config.min_connection_min - epsilon) + big_m * (1 - model.low[c])

        def _high_gap_lower_rule(model, c):
            row = candidate_rows[c]
            big_m = max(0.0, (config.max_connection_min + epsilon) - float(row["min_possible_gap"]))
            return model.gap[c] >= (config.max_connection_min + epsilon) - big_m * (1 - model.high[c])

        m.connection_state = pyo.Constraint(m.C, rule=_connection_state_rule)
        m.low_gap_upper = pyo.Constraint(m.C, rule=_low_gap_upper_rule)
        m.high_gap_lower = pyo.Constraint(m.C, rule=_high_gap_lower_rule)
        print("[INFO] Mandatory connection selection enabled: x iff connection gap is inside the window.")

    def _rotation_rule(model, q):
        row = rotation_rows[q]
        return (
            float(row["old_inbound_arrival_min"]) + model.event_shift[row["inbound_event_id"]]
            >= float(row["old_outbound_departure_min"]) + model.event_shift[row["outbound_event_id"]] + float(row["required_delta_min"])
        )

    m.rotation_constraints = pyo.Constraint(m.Q, rule=_rotation_rule)
    skipped_count = int((rotation_candidates["rotation_constraint_available"] != True).sum()) if not rotation_candidates.empty else 0
    if enable_rotation_constraints and len(active_rotations):
        print(f"[INFO] Added {len(active_rotations)} rotation constraints from Flight Pairs.")
    elif enable_rotation_constraints:
        print("[WARN] No active rotation constraints could be added from Flight Pairs.")
    if enable_rotation_constraints and not active_rotations.empty and "derivation_method" in active_rotations.columns:
        derivation = active_rotations["derivation_method"].fillna("").astype(str)
        fallback_count = int(derivation.str.startswith("approximated from schedule elapsed").sum())
        od_count = int(derivation.str.startswith("O&D connection block total").sum())
        if od_count:
            print(f"[INFO] Derived block times from O&D connection rows for {od_count} rotation constraints.")
        if fallback_count:
            print(
                "[WARN] O&D block-time match missing for "
                f"{fallback_count} active rotation constraints; used schedule elapsed approximation."
            )
    if skipped_count:
        print(f"[WARN] Skipped {skipped_count} rotation candidates; see rotation_candidates.xlsx.")

    event_rows = flight_events.set_index("event_id").to_dict("index")

    def _bucket_assign_once_rule(model, event_id):
        # Yalnizca kapasite-ilgili event'ler bucket_assign alir (onisleme);
        # digerleri icin kova atamasi gereksizdir (kapasite baglayamaz).
        buckets = feasible_buckets_by_event.get(event_id)
        if not buckets:
            return pyo.Constraint.Skip
        return sum(model.bucket_assign[event_id, b] for b in buckets) == 1

    m.bucket_assign_once = pyo.Constraint(m.R, rule=_bucket_assign_once_rule)

    def _bucket_lower_rule(model, event_id, bucket):
        event = event_rows[event_id]
        start = bucket * config.bucket_size_min
        big_m = 2 * config.max_shift_min + config.bucket_size_min + 1
        new_minute = float(event["minute_of_day"]) + model.event_shift[event_id]
        return new_minute >= start - big_m * (1 - model.bucket_assign[event_id, bucket])

    def _bucket_upper_rule(model, event_id, bucket):
        event = event_rows[event_id]
        end = (bucket + 1) * config.bucket_size_min
        big_m = 2 * config.max_shift_min + config.bucket_size_min + 1
        new_minute = float(event["minute_of_day"]) + model.event_shift[event_id]
        return new_minute <= (end - 1) + big_m * (1 - model.bucket_assign[event_id, bucket])

    m.bucket_lower = pyo.Constraint(m.EB, rule=_bucket_lower_rule)
    m.bucket_upper = pyo.Constraint(m.EB, rule=_bucket_upper_rule)

    capacity_index = [
        (int(row.day), int(row.bucket), row.direction)
        for row in bucket_table.itertuples(index=False)
    ] if enable_hub_capacity_constraints else []
    m.CAP = pyo.Set(initialize=capacity_index, dimen=3, ordered=True)
    active_regularity = (
        regularity_report[regularity_report["status"] == "constrained"].copy()
        if enable_regularity_constraints and config.regularity_mode in {"hard", "soft"}
        else pd.DataFrame(columns=regularity_report.columns)
    )
    regularity_rows = active_regularity.set_index("regularity_id").to_dict("index")
    m.REG = pyo.Set(initialize=active_regularity["regularity_id"].tolist(), ordered=True)
    m.regularity_slack = pyo.Var(m.REG, domain=pyo.NonNegativeReals)
    # E1/E2 yalnizca her iki yonun de en az bir aday baglantisi olan ciftlere
    # uygulanir. Ters yonu yapisal olarak hic aday iceremeyen pazarlarda denge
    # kisiti tanimsizdir (mandatory selection ile birlesince model tum agi
    # infeasible yapar); haric tutulan ciftler sebep koduyla raporlanir.
    keys_with_candidates = {key for key, cands in candidate_by_key.items() if cands}
    if not directional_pairs.empty:
        directional_pairs["both_directions_have_candidates"] = directional_pairs["od_key"].isin(
            keys_with_candidates
        ) & directional_pairs["reverse_key"].isin(keys_with_candidates)
    else:
        directional_pairs["both_directions_have_candidates"] = pd.Series(dtype=bool)
    if enable_directional_constraints and not directional_pairs.empty:
        active_directional = directional_pairs[directional_pairs["both_directions_have_candidates"]].copy()
        excluded_pairs = int((~directional_pairs["both_directions_have_candidates"]).sum())
        if excluded_pairs:
            print(
                f"[INFO] Excluded {excluded_pairs} directional pairs from E1/E2: "
                "one direction has no TK candidate connections (structurally one-sided market)."
            )
    else:
        active_directional = pd.DataFrame(columns=directional_pairs.columns)
    m.DP = pyo.Set(initialize=active_directional["directional_pair_id"].tolist(), ordered=True)
    directional_rows = active_directional.set_index("directional_pair_id").to_dict("index")
    m.directional_diff = pyo.Var(m.DP, domain=pyo.NonNegativeReals)
    m.directional_e1_slack = pyo.Var(m.DP, domain=pyo.NonNegativeReals)
    m.directional_e2_slack = pyo.Var(m.DP, domain=pyo.NonNegativeReals)
    events_by_day_bucket_direction: dict[tuple[int, int, str], list[str]] = {}
    for event_id, bucket in event_bucket_pairs:
        event = event_rows[event_id]
        key = (int(event["day"]), int(bucket), event["direction"])
        events_by_day_bucket_direction.setdefault(key, []).append(event_id)
    capacity_lookup = {
        (int(row.day), int(row.bucket), row.direction): int(row.capacity)
        for row in bucket_table.itertuples(index=False)
    }

    def _hub_capacity_rule(model, day, bucket, direction):
        assigned = [
            model.bucket_assign[event_id, bucket]
            for event_id in events_by_day_bucket_direction.get((int(day), int(bucket), direction), [])
        ]
        if not assigned:
            return pyo.Constraint.Skip
        return sum(assigned) <= capacity_lookup[(int(day), int(bucket), direction)]

    m.hub_capacity = pyo.Constraint(m.CAP, rule=_hub_capacity_rule)
    if enable_hub_capacity_constraints:
        print(f"[INFO] Added hub capacity constraints for {len(flight_events)} adjustable events.")
    else:
        print(f"[INFO] Hub bucket assignment prepared for {len(flight_events)} adjustable events; capacity constraints disabled.")

    def _regularity_upper_rule(model, reg_id):
        row = regularity_rows[reg_id]
        new_a = float(row["old_minute_of_day_a"]) + model.event_shift[row["event_id_a"]]
        new_b = float(row["old_minute_of_day_b"]) + model.event_shift[row["event_id_b"]]
        slack = model.regularity_slack[reg_id] if _regularity_row_is_soft(row, config) else 0
        return new_a - new_b <= float(row.get("effective_xdev_min", config.schedule_regular_dev_min)) + slack

    def _regularity_lower_rule(model, reg_id):
        row = regularity_rows[reg_id]
        new_a = float(row["old_minute_of_day_a"]) + model.event_shift[row["event_id_a"]]
        new_b = float(row["old_minute_of_day_b"]) + model.event_shift[row["event_id_b"]]
        slack = model.regularity_slack[reg_id] if _regularity_row_is_soft(row, config) else 0
        return new_b - new_a <= float(row.get("effective_xdev_min", config.schedule_regular_dev_min)) + slack

    m.regularity_upper = pyo.Constraint(m.REG, rule=_regularity_upper_rule)
    m.regularity_lower = pyo.Constraint(m.REG, rule=_regularity_lower_rule)
    if enable_regularity_constraints and len(active_regularity):
        if config.regularity_mode == "soft" or config.regularity_relax_mode == "targeted_soft":
            print(
                "[INFO] Added "
                f"{len(active_regularity)} soft schedule regularity constraints "
                f"with weight={config.regularity_penalty_weight}."
            )
        else:
            print(f"[INFO] Added {len(active_regularity)} schedule regularity constraints.")
    elif enable_regularity_constraints:
        if config.regularity_mode == "report_only":
            print("[INFO] Regularity report_only mode; no model-side regularity constraint added.")
        elif config.shift_granularity == "flight_week":
            print(
                "[INFO] Schedule regularity structurally guaranteed by flight_week shift granularity; "
                "no explicit cross-day constraints required."
            )
        else:
            print("[WARN] No active schedule regularity constraints could be added.")

    def _slot_fill_rule(model, key):
        if not candidate_by_key.get(key, []) and not slots[key]:
            return pyo.Constraint.Feasible
        return sum(model.x[c] for c in candidate_by_key.get(key, [])) == sum(model.z[key, s] for s in slots[key])

    m.slot_fill = pyo.Constraint(m.K, rule=_slot_fill_rule)

    def _slot_monotonic_rule(model, key, slot):
        if slot <= 1:
            return pyo.Constraint.Skip
        return model.z[key, slot] <= model.z[key, slot - 1]

    m.slot_monotonic = pyo.Constraint(m.S, rule=_slot_monotonic_rule)

    candidate_key = candidates.set_index("candidate_id")["od_day_key"].to_dict()
    max_candidates_by_key = {key: max(1, len(candidate_by_key.get(key, []))) for key in od_day_keys}

    def _key_active_upper_rule(model, key):
        return sum(model.x[c] for c in candidate_by_key.get(key, [])) <= max_candidates_by_key[key] * model.key_active[key]

    def _key_active_lower_rule(model, key):
        return sum(model.x[c] for c in candidate_by_key.get(key, [])) >= model.key_active[key]

    m.key_active_upper = pyo.Constraint(m.K, rule=_key_active_upper_rule)
    m.key_active_lower = pyo.Constraint(m.K, rule=_key_active_lower_rule)

    def _best_select_rule(model, key):
        return sum(model.best_conn[c] for c in candidate_by_key.get(key, [])) == model.key_active[key]

    m.best_select = pyo.Constraint(m.K, rule=_best_select_rule)

    def _best_conn_active_rule(model, c):
        return model.best_conn[c] <= model.x[c]

    m.best_conn_active = pyo.Constraint(m.C, rule=_best_conn_active_rule)

    def _journey_expr(model, c):
        row = candidate_rows[c]
        return float(row["journey_time_min"]) + model.gap[c] - float(row["old_gap_min"])

    def _best_time_le_active_rule(model, c):
        key = candidate_key[c]
        return model.best_time[key] <= _journey_expr(model, c) + key_max_journey[key] * (1 - model.x[c])

    def _best_time_ge_selected_rule(model, c):
        key = candidate_key[c]
        return model.best_time[key] >= _journey_expr(model, c) - key_max_journey[key] * (1 - model.best_conn[c])

    def _best_time_le_selected_rule(model, c):
        key = candidate_key[c]
        return model.best_time[key] <= _journey_expr(model, c) + key_max_journey[key] * (1 - model.best_conn[c])

    m.best_time_le_active = pyo.Constraint(m.C, rule=_best_time_le_active_rule)
    m.best_time_ge_selected = pyo.Constraint(m.C, rule=_best_time_ge_selected_rule)
    m.best_time_le_selected = pyo.Constraint(m.C, rule=_best_time_le_selected_rule)

    lookup_epsilon = 0.001

    def _lookup_comp_beaten_active_rule(model, comp_id):
        if config.ranking_objective_mode != "lookup_linear":
            return pyo.Constraint.Skip
        key = lookup_competitor_key[comp_id]
        return model.lookup_competitor_beaten[comp_id] <= model.key_active[key]

    def _lookup_comp_beaten_upper_rule(model, comp_id):
        if config.ranking_objective_mode != "lookup_linear":
            return pyo.Constraint.Skip
        key = lookup_competitor_key[comp_id]
        big_m = max(0.0, key_max_journey[key] - lookup_competitor_time[comp_id])
        return model.best_time[key] <= lookup_competitor_time[comp_id] + big_m * (1 - model.lookup_competitor_beaten[comp_id])

    def _lookup_comp_beaten_lower_rule(model, comp_id):
        if config.ranking_objective_mode != "lookup_linear":
            return pyo.Constraint.Skip
        key = lookup_competitor_key[comp_id]
        inactive = 1 - model.key_active[key]
        big_m = lookup_competitor_time[comp_id] + lookup_epsilon
        return model.best_time[key] >= lookup_competitor_time[comp_id] + lookup_epsilon - big_m * model.lookup_competitor_beaten[comp_id] - big_m * inactive

    m.lookup_comp_beaten_active = pyo.Constraint(m.LCOMP, rule=_lookup_comp_beaten_active_rule)
    m.lookup_comp_beaten_upper = pyo.Constraint(m.LCOMP, rule=_lookup_comp_beaten_upper_rule)
    m.lookup_comp_beaten_lower = pyo.Constraint(m.LCOMP, rule=_lookup_comp_beaten_lower_rule)

    def _lookup_level_select_rule(model, key):
        if config.ranking_objective_mode != "lookup_linear":
            return pyo.Constraint.Skip
        levels = range(int(competitor_count_by_key.get(key, 0)) + 1)
        return sum(model.lookup_level[key, level] for level in levels) == 1

    def _lookup_level_count_rule(model, key):
        if config.ranking_objective_mode != "lookup_linear":
            return pyo.Constraint.Skip
        levels = range(int(competitor_count_by_key.get(key, 0)) + 1)
        return sum(model.lookup_competitor_beaten[comp_id] for comp_id in lookup_competitors_by_key.get(key, [])) == sum(
            level * model.lookup_level[key, level] for level in levels
        )

    m.lookup_level_select = pyo.Constraint(m.K, rule=_lookup_level_select_rule)
    m.lookup_level_count = pyo.Constraint(m.K, rule=_lookup_level_count_rule)

    def _active_count_expr(model, key):
        return sum(model.x[c] for c in candidate_by_key.get(key, []))

    def _directional_diff_forward_rule(model, pair_id):
        row = directional_rows[pair_id]
        return model.directional_diff[pair_id] >= _active_count_expr(model, row["od_key"]) - _active_count_expr(model, row["reverse_key"])

    def _directional_diff_reverse_rule(model, pair_id):
        row = directional_rows[pair_id]
        return model.directional_diff[pair_id] >= _active_count_expr(model, row["reverse_key"]) - _active_count_expr(model, row["od_key"])

    def _directional_count_balance_rule(model, pair_id):
        if config.directional_e1_mode not in {"hard", "soft"} and config.directional_e1_formula != "soft_only":
            return pyo.Constraint.Skip
        row = directional_rows[pair_id]
        total = _active_count_expr(model, row["od_key"]) + _active_count_expr(model, row["reverse_key"])
        allowed = config.directional_alpha * total
        if config.directional_e1_formula == "absolute_tolerance":
            allowed += config.directional_e1_abs_tolerance
        if config.directional_e1_mode == "soft" or config.directional_e1_formula == "soft_only":
            allowed += model.directional_e1_slack[pair_id]
        return model.directional_diff[pair_id] <= allowed

    m.directional_diff_forward = pyo.Constraint(m.DP, rule=_directional_diff_forward_rule)
    m.directional_diff_reverse = pyo.Constraint(m.DP, rule=_directional_diff_reverse_rule)
    m.directional_count_balance = pyo.Constraint(m.DP, rule=_directional_count_balance_rule)

    def _directional_pair_big_m(row):
        return max(key_max_journey.get(row["od_key"], 1.0), key_max_journey.get(row["reverse_key"], 1.0))

    def _directional_time_upper_rule(model, pair_id):
        if config.directional_e2_mode not in {"hard", "soft"}:
            return pyo.Constraint.Skip
        row = directional_rows[pair_id]
        inactive = 2 - model.key_active[row["od_key"]] - model.key_active[row["reverse_key"]]
        pair_big_m = _directional_pair_big_m(row)
        if config.directional_e2_mode == "soft":
            return model.best_time[row["od_key"]] - model.best_time[row["reverse_key"]] <= config.directional_gamma_min + model.directional_e2_slack[pair_id] + pair_big_m * inactive
        return model.best_time[row["od_key"]] - model.best_time[row["reverse_key"]] <= config.directional_gamma_min + pair_big_m * inactive

    def _directional_time_lower_rule(model, pair_id):
        if config.directional_e2_mode not in {"hard", "soft"}:
            return pyo.Constraint.Skip
        row = directional_rows[pair_id]
        inactive = 2 - model.key_active[row["od_key"]] - model.key_active[row["reverse_key"]]
        pair_big_m = _directional_pair_big_m(row)
        if config.directional_e2_mode == "soft":
            return model.best_time[row["reverse_key"]] - model.best_time[row["od_key"]] <= config.directional_gamma_min + model.directional_e2_slack[pair_id] + pair_big_m * inactive
        return model.best_time[row["reverse_key"]] - model.best_time[row["od_key"]] <= config.directional_gamma_min + pair_big_m * inactive

    m.directional_time_upper = pyo.Constraint(m.DP, rule=_directional_time_upper_rule)
    m.directional_time_lower = pyo.Constraint(m.DP, rule=_directional_time_lower_rule)
    if enable_directional_constraints and len(active_directional):
        if config.directional_e1_mode == "hard" and config.directional_e1_formula != "soft_only":
            print(
                "[INFO] Added "
                f"{len(active_directional)} directional E1 hard count-balance constraints "
                f"with formula={config.directional_e1_formula}."
            )
        elif config.directional_e1_mode == "soft" or config.directional_e1_formula == "soft_only":
            print(
                "[INFO] Added directional E1 soft penalty constraints "
                f"with formula={config.directional_e1_formula}, weight={config.directional_e1_penalty_weight}."
            )
        else:
            print("[INFO] Directional E1 report_only mode; no model-side E1 constraint added.")
        if config.directional_e2_mode == "hard":
            print(f"[INFO] Added directional E2 hard best-time constraints with gamma={config.directional_gamma_min}.")
        elif config.directional_e2_mode == "soft":
            print(
                "[INFO] Added directional E2 soft penalty constraints "
                f"with gamma={config.directional_gamma_min}, weight={config.directional_e2_penalty_weight}."
            )
    elif enable_directional_constraints:
        print("[WARN] No active directional pairs could be added.")

    def _beat_active_rule(model, c):
        return model.beat[c] <= model.x[c]

    def _beat_possible_rule(model, c):
        best_comp = best_comp_by_candidate[c]
        if best_comp == float("inf"):
            return model.beat[c] == 0
        row = candidate_rows[c]
        max_new_journey = float(row["journey_time_min"]) + float(row["max_possible_gap"]) - float(row["old_gap_min"])
        big_m = max(0.0, max_new_journey - best_comp)
        return _journey_expr(model, c) <= best_comp + big_m * (1 - model.beat[c])

    m.beat_active = pyo.Constraint(m.C, rule=_beat_active_rule)
    m.beat_possible = pyo.Constraint(m.C, rule=_beat_possible_rule)

    def _reward_rule(model):
        conn_reward = sum(
            importance[key] * (config.connection_reward_decay ** (slot - 1)) * model.z[key, slot]
            for key, slot in model.S
        )
        if config.ranking_objective_mode == "lookup_linear":
            beat_reward = sum(
                importance[key] * lookup_reward_by_level[(key, level)] * model.lookup_level[key, level]
                for key, level in model.LR
            )
        else:
            beat_reward = sum(
                importance[candidate_key[c]] * config.competitor_beating_reward * model.beat[c]
                for c in model.C
            )
        return conn_reward + beat_reward

    def _penalty_rule(model):
        e2_penalty = (
            config.directional_e2_penalty_weight * sum(model.directional_e2_slack[pair_id] for pair_id in model.DP)
            if config.directional_e2_mode == "soft"
            else 0
        )
        e1_penalty = (
            config.directional_e1_penalty_weight * sum(model.directional_e1_slack[pair_id] for pair_id in model.DP)
            if config.directional_e1_mode == "soft" or config.directional_e1_formula == "soft_only"
            else 0
        )
        regularity_penalty = (
            config.regularity_penalty_weight * sum(model.regularity_slack[reg_id] for reg_id in model.REG)
            if config.regularity_mode == "soft" or config.regularity_relax_mode == "targeted_soft"
            else 0
        )
        shift_penalty = 0
        if config.shift_penalty_mode == "linear":
            shift_penalty = config.shift_penalty_weight * sum(model.abs_shift[sh] for sh in model.SH)
        elif config.shift_penalty_mode == "threshold":
            shift_penalty = config.shift_penalty_weight * sum(model.excess_shift[sh] for sh in model.SH)
        elif config.shift_penalty_mode == "quadratic_proxy":
            shift_penalty = config.shift_penalty_weight * sum(
                0.25 * model.shift_seg_0_5[sh]
                + 1.0 * model.shift_seg_5_15[sh]
                + 2.0 * model.shift_seg_15_plus[sh]
                for sh in model.SH
            )
        return e1_penalty + e2_penalty + regularity_penalty + shift_penalty

    # Odul ve ceza ayri Expression olarak saklanir; iki asamali (leksikografik)
    # cozum bunlari kullanarak once cezayi minimize eder, sonra cezayi capleyip
    # odulu maksimize eder (solver.solve_two_phase).
    m.reward_expr = pyo.Expression(rule=_reward_rule)
    m.penalty_expr = pyo.Expression(rule=_penalty_rule)
    m.objective = pyo.Objective(expr=m.reward_expr - m.penalty_expr, sense=pyo.maximize)
    return ModelArtifacts(
        m,
        candidates,
        flight_events,
        rotation_candidates,
        rotation_report,
        bucket_table,
        event_bucket_pairs,
        regularity_report,
        directional_pairs,
        od_day_keys,
        slots,
    )


def extract_solution(artifacts: ModelArtifacts) -> pd.DataFrame:
    """Decode candidate connection x values from a solved model."""

    import pyomo.environ as pyo

    model = artifacts.model
    rows = []
    for _, row in artifacts.candidate_connections.iterrows():
        value = pyo.value(model.x[row["candidate_id"]], exception=False)
        inbound_shift = pyo.value(model.event_shift[row["inbound_event_id"]], exception=False) or 0.0
        outbound_shift = pyo.value(model.event_shift[row["outbound_event_id"]], exception=False) or 0.0
        new_gap = pyo.value(model.gap[row["candidate_id"]], exception=False)
        if new_gap is None:
            new_gap = row["old_gap_min"] + outbound_shift - inbound_shift
        new_journey = float(row["journey_time_min"]) + float(new_gap) - float(row["old_gap_min"])
        beat_value = pyo.value(model.beat[row["candidate_id"]], exception=False)
        low_value = pyo.value(model.low[row["candidate_id"]], exception=False) if hasattr(model, "low") else None
        high_value = pyo.value(model.high[row["candidate_id"]], exception=False) if hasattr(model, "high") else None
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "connection_id": row["connection_id"],
                "inbound_event_id": row["inbound_event_id"],
                "outbound_event_id": row["outbound_event_id"],
                "active": bool(value and value > 0.5),
                "low_state": bool(low_value and low_value > 0.5) if low_value is not None else None,
                "high_state": bool(high_value and high_value > 0.5) if high_value is not None else None,
                "inbound_shift_min": int(round(inbound_shift)),
                "outbound_shift_min": int(round(outbound_shift)),
                "old_inbound_arrival": row["arr_time"],
                "new_inbound_arrival": row["arr_time"] + pd.to_timedelta(inbound_shift, unit="m"),
                "old_outbound_departure": row["dep_time"],
                "new_outbound_departure": row["dep_time"] + pd.to_timedelta(outbound_shift, unit="m"),
                "old_gap": float(row["old_gap_min"]),
                "new_gap": float(new_gap),
                "old_journey_time_min": float(row["journey_time_min"]),
                "new_journey_time_min": float(new_journey),
                "beaten_best_competitor": bool(beat_value and beat_value > 0.5),
            }
        )
    return pd.DataFrame(rows)


def extract_optimized_schedule(artifacts: ModelArtifacts) -> pd.DataFrame:
    """Decode all adjustable modeled flight events and their new IST times."""

    import pyomo.environ as pyo

    model = artifacts.model
    rows = []
    for row in artifacts.flight_events.itertuples(index=False):
        shift = pyo.value(model.event_shift[row.event_id], exception=False) or 0.0
        old_time = row.old_ist_time
        new_time = old_time + pd.to_timedelta(shift, unit="m")
        if row.direction == "inbound":
            origin = row.station
            destination = "IST"
        else:
            origin = "IST"
            destination = row.station
        rows.append(
            {
                "flight_id": row.event_id,
                "flight_no": row.flight_no,
                "day": int(row.day),
                "origin": origin,
                "destination": destination,
                "old_ist_time": old_time,
                "new_ist_time": new_time,
                "time_change_min": int(round(shift)),
                "is_adjustable": True,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["flight_id"]).reset_index(drop=True) if rows else pd.DataFrame()


def extract_rotation_report(artifacts: ModelArtifacts) -> pd.DataFrame:
    """Decode rotation constraints and violations from a solved model."""

    import pyomo.environ as pyo

    model = artifacts.model
    report = artifacts.rotation_candidates.copy()
    if report.empty:
        return report
    new_out = []
    new_in = []
    slack = []
    violation = []
    outbound_shift = []
    inbound_shift = []
    for row in report.itertuples(index=False):
        if row.status != "constrained":
            new_out.append(pd.NaT)
            new_in.append(pd.NaT)
            slack.append(None)
            violation.append(False)
            outbound_shift.append(None)
            inbound_shift.append(None)
            continue
        out_shift = pyo.value(model.event_shift[row.outbound_event_id], exception=False) or 0.0
        in_shift = pyo.value(model.event_shift[row.inbound_event_id], exception=False) or 0.0
        new_out_min = float(row.old_outbound_departure_min) + float(out_shift)
        new_in_min = float(row.old_inbound_arrival_min) + float(in_shift)
        row_slack = new_in_min - new_out_min - float(row.required_delta_min)
        outbound_shift.append(int(round(out_shift)))
        inbound_shift.append(int(round(in_shift)))
        new_out.append(row.old_outbound_departure_time + pd.to_timedelta(out_shift, unit="m"))
        new_in.append(row.old_inbound_arrival_time + pd.to_timedelta(in_shift, unit="m"))
        slack.append(row_slack)
        violation.append(row_slack < -1e-6)
    report["outbound_shift_min"] = outbound_shift
    report["inbound_shift_min"] = inbound_shift
    report["new_outbound_departure_time"] = new_out
    report["new_inbound_arrival_time"] = new_in
    report["actual_delta_min"] = [
        None if s is None or pd.isna(s) else float(r.required_delta_min) + float(s)
        for s, r in zip(slack, report.itertuples(index=False))
    ]
    report["slack_min"] = slack
    report["violation"] = violation
    return report


def extract_hub_capacity_report(artifacts: ModelArtifacts, config: ModelConfig) -> pd.DataFrame:
    """Decode hub bucket assignments and capacity usage from a solved model."""

    import pyomo.environ as pyo

    if artifacts.bucket_table.empty:
        return pd.DataFrame(
            columns=[
                "day",
                "bucket_start",
                "bucket_end",
                "arrivals",
                "departures",
                "arrival_capacity",
                "departure_capacity",
                "violation",
            ]
        )

    model = artifacts.model
    events = artifacts.flight_events.set_index("event_id").to_dict("index")
    assignments: dict[tuple[int, int, str], int] = {}
    for event_id, bucket in artifacts.event_bucket_pairs:
        value = pyo.value(model.bucket_assign[event_id, bucket], exception=False)
        if value and value > 0.5:
            event = events[event_id]
            key = (int(event["day"]), int(bucket), event["direction"])
            assignments[key] = assignments.get(key, 0) + 1

    rows = []
    days = sorted(artifacts.bucket_table["day"].dropna().astype(int).unique().tolist())
    bucket_count = 1440 // config.bucket_size_min
    for day in days:
        for bucket in range(bucket_count):
            arrivals = assignments.get((day, bucket, "inbound"), 0)
            departures = assignments.get((day, bucket, "outbound"), 0)
            rows.append(
                {
                    "day": day,
                    "bucket_start": bucket * config.bucket_size_min,
                    "bucket_end": (bucket + 1) * config.bucket_size_min,
                    "arrivals": arrivals,
                    "departures": departures,
                    "arrival_capacity": config.arrival_bucket_capacity,
                    "departure_capacity": config.departure_bucket_capacity,
                    "violation": arrivals > config.arrival_bucket_capacity or departures > config.departure_bucket_capacity,
                }
            )
    return pd.DataFrame(rows)


def extract_regularity_report(artifacts: ModelArtifacts, config: ModelConfig) -> pd.DataFrame:
    """Decode schedule regularity pair checks from a solved model."""

    import pyomo.environ as pyo

    report = artifacts.regularity_report.copy()
    if report.empty:
        return report
    model = artifacts.model
    new_a = []
    new_b = []
    diff = []
    violation = []
    shift_a = []
    shift_b = []
    for row in report.itertuples(index=False):
        if row.status != "constrained":
            new_a.append(pd.NaT)
            new_b.append(pd.NaT)
            diff.append(None)
            violation.append(False)
            shift_a.append(None)
            shift_b.append(None)
            continue
        s_a = pyo.value(model.event_shift[row.event_id_a], exception=False) or 0.0
        s_b = pyo.value(model.event_shift[row.event_id_b], exception=False) or 0.0
        minute_a = float(row.old_minute_of_day_a) + float(s_a)
        minute_b = float(row.old_minute_of_day_b) + float(s_b)
        current_diff = abs(minute_a - minute_b)
        shift_a.append(int(round(s_a)))
        shift_b.append(int(round(s_b)))
        new_a.append(row.old_time_a + pd.to_timedelta(s_a, unit="m"))
        new_b.append(row.old_time_b + pd.to_timedelta(s_b, unit="m"))
        diff.append(current_diff)
        violation.append(current_diff > config.schedule_regular_dev_min + 1e-6)
    report["shift_a_min"] = shift_a
    report["shift_b_min"] = shift_b
    report["new_time_a"] = new_a
    report["new_time_b"] = new_b
    report["new_time_diff_min"] = diff
    report["violation"] = violation
    return report


def _build_flight_events(candidates: pd.DataFrame) -> pd.DataFrame:
    inbound = candidates[
        ["inbound_event_id", "inbound_flight", "inbound_flight_norm", "inbound_event_day", "origin", "arr_time", "arr_min"]
    ].rename(
        columns={
            "inbound_event_id": "event_id",
            "inbound_flight": "flight_no",
            "inbound_flight_norm": "flight_no_norm",
            "inbound_event_day": "day",
            "origin": "station",
            "arr_time": "old_ist_time",
            "arr_min": "old_min",
        }
    )
    inbound["direction"] = "inbound"
    inbound["origin"] = inbound["station"]
    inbound["destination"] = "IST"
    inbound["minute_of_day"] = inbound["old_min"].astype(float) % 1440
    inbound["station_norm"] = inbound["station"].map(_normalize_airport_code)
    outbound = candidates[
        ["outbound_event_id", "outbound_flight", "outbound_flight_norm", "outbound_event_day", "destination", "dep_time", "dep_min"]
    ].rename(
        columns={
            "outbound_event_id": "event_id",
            "outbound_flight": "flight_no",
            "outbound_flight_norm": "flight_no_norm",
            "outbound_event_day": "day",
            "destination": "station",
            "dep_time": "old_ist_time",
            "dep_min": "old_min",
        }
    )
    outbound["direction"] = "outbound"
    outbound["origin"] = "IST"
    outbound["destination"] = outbound["station"]
    outbound["minute_of_day"] = outbound["old_min"].astype(float) % 1440
    outbound["station_norm"] = outbound["station"].map(_normalize_airport_code)
    return pd.concat([inbound, outbound], ignore_index=True).drop_duplicates("event_id").reset_index(drop=True)


def _assign_shift_keys(
    flight_events: pd.DataFrame,
    config: ModelConfig,
) -> tuple[pd.DataFrame, dict[str, str], set[str], dict]:
    """Assign each event a shift variable key according to shift granularity.

    flight_week modunda ayni (yon, ucus no, istasyon) grubunun tum gunleri tek
    shift degiskenini paylasir; boylece tarife duzenliligi (G) yapisal olarak
    saglanir. Haftalik paterni Xdev'den fazla dagilan gruplar:
      - 2*max_shift ile Xdev bandina cekilebiliyorsa event bazinda cozulur,
      - cekilemiyorsa dondurulur (shift=0) ve pre-existing ihlal olarak raporlanir.
    """

    fe = flight_events.copy()
    fe["group_key"] = (
        "grp|" + fe["direction"].astype(str) + "|" + fe["flight_no_norm"].astype(str) + "|" + fe["station"].astype(str)
    )
    stats = {
        "shift_granularity": config.shift_granularity,
        "shift_groups": int(fe["group_key"].nunique()),
        "regular_groups": 0,
        "irregular_fixable_groups": 0,
        "frozen_groups": 0,
    }
    if config.shift_granularity != "flight_week":
        fe["shift_key"] = fe["event_id"]
        return fe, dict(zip(fe["event_id"], fe["shift_key"])), set(), stats

    xdev = float(config.schedule_regular_dev_min)
    spread = fe.groupby("group_key")["minute_of_day"].agg(circular_spread)
    irregular = spread[spread > xdev + 1e-6]
    fixable_groups = set(irregular[irregular - 2 * config.max_shift_min <= xdev + 1e-6].index)
    frozen_groups = set(irregular.index) - fixable_groups
    stats["regular_groups"] = int(len(spread) - len(irregular))
    stats["irregular_fixable_groups"] = int(len(fixable_groups))
    stats["frozen_groups"] = int(len(frozen_groups))

    fe["shift_key"] = fe["group_key"]
    fe.loc[fe["group_key"].isin(fixable_groups), "shift_key"] = fe.loc[
        fe["group_key"].isin(fixable_groups), "event_id"
    ]
    print(
        "[INFO] Shift granularity=flight_week: "
        f"{stats['regular_groups']} regular groups share one weekly shift, "
        f"{stats['irregular_fixable_groups']} irregular groups solved per-event, "
        f"{stats['frozen_groups']} unfixable groups frozen at original times."
    )
    return fe, dict(zip(fe["event_id"], fe["shift_key"])), frozen_groups, stats


def _build_fixed_tk_events(data: PreparedData, adjustable_events: pd.DataFrame) -> pd.DataFrame:
    """Return TK IST events from the full network that are NOT adjustable in this model.

    Bu etkinlikler tarifesi sabit kabul edilir ve hub kova kapasitesinden
    dusulur (rezidüel kapasite, Brief 4.6/F geregi).
    """

    columns = ["event_id", "day", "direction", "minute_of_day"]
    full = getattr(data, "full_connections", None)
    if full is None or full.empty or adjustable_events.empty:
        return pd.DataFrame(columns=columns)
    tk = full[full["is_tk"]].copy()
    if tk.empty:
        return pd.DataFrame(columns=columns)
    tk["inbound_flight_norm"] = tk["inbound_flight"].map(_normalize_flight_no)
    tk["outbound_flight_norm"] = tk["outbound_flight"].map(_normalize_flight_no)
    inbound_event_day = (tk["arr_min"].astype(int) // 1440).clip(lower=1, upper=7)
    outbound_event_day = (tk["dep_min"].astype(int) // 1440).clip(lower=1, upper=7)
    inbound = pd.DataFrame(
        {
            "event_id": (
                "in|" + inbound_event_day.astype(str) + "|" + tk["inbound_flight_norm"].astype(str)
                + "|" + tk["origin"].astype(str) + "|" + tk["arr_min"].astype(str)
            ),
            "day": inbound_event_day.astype(int),
            "direction": "inbound",
            "minute_of_day": tk["arr_min"].astype(float) % 1440,
        }
    )
    outbound = pd.DataFrame(
        {
            "event_id": (
                "out|" + outbound_event_day.astype(str) + "|" + tk["outbound_flight_norm"].astype(str)
                + "|" + tk["destination"].astype(str) + "|" + tk["dep_min"].astype(str)
            ),
            "day": outbound_event_day.astype(int),
            "direction": "outbound",
            "minute_of_day": tk["dep_min"].astype(float) % 1440,
        }
    )
    events = pd.concat([inbound, outbound], ignore_index=True).drop_duplicates("event_id")
    adjustable_ids = set(adjustable_events["event_id"].tolist())
    fixed = events[~events["event_id"].isin(adjustable_ids)].reset_index(drop=True)
    model_days = set(adjustable_events["day"].dropna().astype(int).tolist())
    return fixed[fixed["day"].isin(model_days)].reset_index(drop=True)


def _build_regularity_anchor_bounds(
    flight_events: pd.DataFrame,
    data: PreparedData,
    config: ModelConfig,
    enabled: bool,
    explicit_references: dict[tuple[str, str, str], list[float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Bound event shifts so weekly XDEV holds against out-of-scope fixed days.

    Kismi gun kosularinda ayni ucus numarasinin kapsam disi gunlerdeki sabit
    saatleri referans alinir; kapsam ici yeni saat bu referanslarin +-Xdev
    bandinda kalacak sekilde shift degiskeni sinirlanir. Bant orijinal saat
    icin bile bos ise (veri zaten ihlalde) shift 0'a sabitlenir ve raporlanir.
    """

    if flight_events.empty or not enabled:
        return {}
    if explicit_references is not None:
        reference = explicit_references
        if not reference:
            return {}
    else:
        if config.regularity_anchor_mode != "auto":
            return {}
        # flight_week modunda kapsam-ici haftalik duzenlilik zaten paylasilan shift
        # ile yapisal olarak saglanir; kapsam-disi gunlere per-event capa uygulamak
        # paylasilan degiskende celiskiye ve yapay infeasibility'ye yol acar. Tam
        # kapsam teslim kosusunda zaten kapsam-disi gun yoktur (capa moot). Bu yuzden
        # auto-capa bu modda devre disi birakilir.
        if config.shift_granularity == "flight_week":
            print(
                "[INFO] Regularity anchor skipped under flight_week granularity "
                "(in-scope weekly regularity already structurally guaranteed)."
            )
            return {}
        full = getattr(data, "full_connections", None)
        if full is None or full.empty:
            return {}
        model_days = set(flight_events["day"].dropna().astype(int).tolist())
        tk = full[full["is_tk"]].copy()
        all_days = set(tk["day"].dropna().astype(int).tolist())
        out_of_scope_days = all_days - model_days
        if not out_of_scope_days:
            return {}
        tk["inbound_flight_norm"] = tk["inbound_flight"].map(_normalize_flight_no)
        tk["outbound_flight_norm"] = tk["outbound_flight"].map(_normalize_flight_no)
        reference = {}
        for row in tk.itertuples(index=False):
            in_day = int(row.arr_min) // 1440
            out_day = int(row.dep_min) // 1440
            if in_day in out_of_scope_days:
                reference.setdefault(("inbound", str(row.inbound_flight_norm), str(row.origin)), []).append(float(row.arr_min) % 1440)
            if out_day in out_of_scope_days:
                reference.setdefault(("outbound", str(row.outbound_flight_norm), str(row.destination)), []).append(float(row.dep_min) % 1440)
    xdev = float(config.schedule_regular_dev_min)
    bounds: dict[str, tuple[float, float]] = {}
    infeasible_count = 0
    for event in flight_events.itertuples(index=False):
        key = (str(event.direction), str(event.flight_no_norm), str(event.station))
        minutes = reference.get(key)
        if not minutes:
            continue
        band_lo = max(minutes) - xdev
        band_hi = min(minutes) + xdev
        old_minute = float(event.minute_of_day)
        shift_lo = band_lo - old_minute
        shift_hi = band_hi - old_minute
        if shift_lo > shift_hi or shift_hi < -config.max_shift_min or shift_lo > config.max_shift_min:
            # Orijinal veri bu ucus icin zaten Xdev bandi disinda; mevcut saati koru.
            bounds[event.event_id] = (0.0, 0.0)
            infeasible_count += 1
            continue
        bounds[event.event_id] = (shift_lo, shift_hi)
    if infeasible_count:
        print(
            "[WARN] Out-of-scope regularity anchor found "
            f"{infeasible_count} events with pre-existing weekly XDEV conflicts; their times were frozen."
        )
    return bounds


def _build_bucket_data(
    flight_events: pd.DataFrame,
    config: ModelConfig,
    fixed_events: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[tuple[str, int]], dict[str, list[int]]]:
    """Create feasible 10-minute bucket assignments for each adjustable event.

    fixed_events verilirse, her kovadaki kapasiteden sabit (ayarlanamayan) TK
    etkinliklerinin kullandigi slotlar dusulur (rezidüel kapasite).
    """

    event_bucket_pairs: list[tuple[str, int]] = []
    feasible_buckets_by_event: dict[str, list[int]] = {}
    if flight_events.empty:
        return pd.DataFrame(columns=["day", "bucket", "direction", "capacity"]), event_bucket_pairs, feasible_buckets_by_event

    # 1) Her event icin ulasilabilir kova araligi.
    event_feasible: dict[str, tuple[int, int, str]] = {}
    potential_count: dict[tuple[int, int, str], int] = {}
    for event in flight_events.itertuples(index=False):
        min_minute = max(0, int(event.minute_of_day) - config.max_shift_min)
        max_minute = min(1439, int(event.minute_of_day) + config.max_shift_min)
        first_bucket = min_minute // config.bucket_size_min
        last_bucket = max_minute // config.bucket_size_min
        day = int(event.day)
        direction = str(event.direction)
        event_feasible[event.event_id] = (first_bucket, last_bucket, direction, day)
        for bucket in range(first_bucket, last_bucket + 1):
            potential_count[(day, bucket, direction)] = potential_count.get((day, bucket, direction), 0) + 1

    # 2) Rezidüel kapasite (sabit TK etkinlikleri dusulur).
    fixed_usage: dict[tuple[int, int, str], int] = {}
    if fixed_events is not None and not fixed_events.empty:
        for row in fixed_events.itertuples(index=False):
            bucket = int(row.minute_of_day) // config.bucket_size_min
            usage_key = (int(row.day), bucket, str(row.direction))
            fixed_usage[usage_key] = fixed_usage.get(usage_key, 0) + 1

    def _residual(day: int, bucket: int, direction: str) -> int:
        nominal = config.arrival_bucket_capacity if direction == "inbound" else config.departure_bucket_capacity
        return nominal - fixed_usage.get((day, bucket, direction), 0)

    # 3) Onisleme: bir kova, oraya dusebilecek event sayisi rezidüel kapasiteyi
    #    asamiyorsa kapasite kisiti hicbir zaman baglayamaz -> atlanir. Yalnizca
    #    baglama-ihtimali olan kovalar tutulur; yalnizca bu kovalara degen event'ler
    #    bucket_assign binary'si alir. (Model dogrulugu korunur: elenen kovalar
    #    kanitlanabilir sekilde gevsek.)
    binding_buckets: set[tuple[int, int, str]] = set()
    for key, count in potential_count.items():
        day, bucket, direction = key
        if count > _residual(day, bucket, direction):
            binding_buckets.add(key)

    capacity_relevant_events = set()
    for event_id, (first_bucket, last_bucket, direction, day) in event_feasible.items():
        if any((day, bucket, direction) in binding_buckets for bucket in range(first_bucket, last_bucket + 1)):
            capacity_relevant_events.add(event_id)

    for event_id, (first_bucket, last_bucket, direction, day) in event_feasible.items():
        if event_id not in capacity_relevant_events:
            continue
        buckets = list(range(first_bucket, last_bucket + 1))
        feasible_buckets_by_event[event_id] = buckets
        event_bucket_pairs.extend((event_id, bucket) for bucket in buckets)

    # 4) Kapasite tablosu: yalnizca baglama-ihtimali olan kovalar.
    rows = []
    clipped = 0
    for (day, bucket, direction) in sorted(binding_buckets):
        residual = _residual(day, bucket, direction)
        if residual < 0:
            clipped += 1
        rows.append({"day": day, "bucket": bucket, "direction": direction, "capacity": max(0, residual)})

    total_buckets = 2 * (1440 // config.bucket_size_min) * len(set(d for _, _, _, d in event_feasible.values()))
    print(
        "[INFO] Hub capacity preprocessing: "
        f"{len(binding_buckets)} potentially-binding buckets kept (of ~{total_buckets}); "
        f"{len(capacity_relevant_events)}/{len(event_feasible)} events need bucket-assignment binaries."
    )
    if fixed_usage:
        print(f"[INFO] Residual hub capacity: subtracted {sum(fixed_usage.values())} fixed TK event slots.")
        if clipped:
            print(f"[WARN] {clipped} buckets had fixed usage above nominal capacity; residual clamped to 0.")
    return pd.DataFrame(rows, columns=["day", "bucket", "direction", "capacity"]), event_bucket_pairs, feasible_buckets_by_event


def _build_regularity_report(flight_events: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """Build same-flight cross-day regularity checks."""

    columns = [
        "regularity_id",
        "flight_no",
        "origin",
        "destination",
        "direction",
        "event_id_a",
        "event_id_b",
        "day_a",
        "day_b",
        "old_time_a",
        "old_time_b",
        "old_minute_of_day_a",
        "old_minute_of_day_b",
        "old_time_diff_min",
        "limit_min",
        "base_xdev_min",
        "theoretical_min_spread",
        "effective_xdev_min",
        "relax_applied",
        "relax_amount",
        "targeted_max_relax",
        "targeted_soft",
        "still_infeasible_by_theoretical_bound",
        "status",
        "warning",
    ]
    if flight_events.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    reg_idx = 0
    group_cols = ["flight_no_norm", "origin", "destination", "direction"]
    for _, group in flight_events.groupby(group_cols):
        group = group.sort_values(["day", "old_min"]).reset_index(drop=True)
        if group["day"].nunique() < 2:
            continue
        records = group.to_dict("records")
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a = records[i]
                b = records[j]
                if int(a["day"]) == int(b["day"]):
                    continue
                old_diff = abs(float(a["minute_of_day"]) - float(b["minute_of_day"]))
                relax = _regularity_effective_xdev(old_diff, config)
                rows.append(
                    {
                        "regularity_id": f"reg_{reg_idx:05d}",
                        "flight_no": a["flight_no_norm"],
                        "origin": a["origin"],
                        "destination": a["destination"],
                        "direction": a["direction"],
                        "event_id_a": a["event_id"],
                        "event_id_b": b["event_id"],
                        "day_a": int(a["day"]),
                        "day_b": int(b["day"]),
                        "old_time_a": a["old_ist_time"],
                        "old_time_b": b["old_ist_time"],
                        "old_minute_of_day_a": float(a["minute_of_day"]),
                        "old_minute_of_day_b": float(b["minute_of_day"]),
                        "old_time_diff_min": old_diff,
                        "limit_min": relax["effective_xdev_min"],
                        "base_xdev_min": config.schedule_regular_dev_min,
                        "theoretical_min_spread": relax["theoretical_min_spread"],
                        "effective_xdev_min": relax["effective_xdev_min"],
                        "relax_applied": relax["relax_applied"],
                        "relax_amount": relax["relax_amount"],
                        "targeted_max_relax": relax["targeted_max_relax"],
                        "targeted_soft": relax["targeted_soft"],
                        "still_infeasible_by_theoretical_bound": relax["still_infeasible_by_theoretical_bound"],
                        "status": "constrained",
                        "warning": relax["warning"],
                    }
                )
                reg_idx += 1
    return pd.DataFrame(rows, columns=columns)


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
    targeted_soft = bool(mode == "targeted_soft" and theoretical > base + 1e-6)
    relax_amount = max(0.0, effective - base)
    still_infeasible = bool(theoretical > effective + 1e-6)
    warning = ""
    if mode == "targeted_theoretical" and still_infeasible:
        warning = "theoretical_min_spread_exceeds_targeted_max_relax"
    elif targeted_soft:
        warning = "targeted_soft_group_uses_slack_penalty"
    return {
        "base_xdev_min": base,
        "theoretical_min_spread": theoretical,
        "effective_xdev_min": effective,
        "relax_applied": bool(relax_amount > 1e-6),
        "relax_amount": relax_amount,
        "targeted_max_relax": targeted_max,
        "targeted_soft": targeted_soft,
        "still_infeasible_by_theoretical_bound": still_infeasible,
        "warning": warning,
    }


def _regularity_row_is_soft(row: dict, config: ModelConfig) -> bool:
    return bool(config.regularity_mode == "soft" or (config.regularity_relax_mode == "targeted_soft" and row.get("targeted_soft", False)))


def _build_directional_pairs(od_day_keys: list[str]) -> pd.DataFrame:
    """Build unique O-D/reverse O-D day pairs present in the current model."""

    key_parts = {}
    for key in od_day_keys:
        od, day_text = key.rsplit("|", 1)
        key_parts[key] = (od, int(day_text))
    key_by_od_day = {(od, day): key for key, (od, day) in key_parts.items()}
    rows = []
    seen = set()
    idx = 0
    for key, (od, day) in key_parts.items():
        reverse = _reverse_od(od)
        reverse_key = key_by_od_day.get((reverse, day))
        if reverse_key is None:
            continue
        pair_key = tuple(sorted([key, reverse_key]))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        first_key, second_key = pair_key
        first_od, _ = key_parts[first_key]
        second_od, _ = key_parts[second_key]
        rows.append(
            {
                "directional_pair_id": f"dir_{idx:05d}",
                "od": first_od,
                "reverse_od": second_od,
                "day": day,
                "od_key": first_key,
                "reverse_key": second_key,
                "status": "constrained",
            }
        )
        idx += 1
    return pd.DataFrame(rows, columns=["directional_pair_id", "od", "reverse_od", "day", "od_key", "reverse_key", "status"])


def _reverse_od(od: str) -> str:
    parts = str(od).split("-", 1)
    if len(parts) != 2:
        return str(od)
    return f"{parts[1]}-{parts[0]}"


def _build_rotation_candidates(
    flight_events: pd.DataFrame,
    flight_pairs: pd.DataFrame,
    config: ModelConfig,
    connections: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build Flight Pairs based rotation candidates for the modeled event subset."""

    columns = [
        "rotation_id",
        "pair_id",
        "station",
        "day",
        "outbound_flight_no",
        "outbound_origin",
        "outbound_destination",
        "inbound_flight_no",
        "inbound_origin",
        "inbound_destination",
        "outbound_event_id",
        "inbound_event_id",
        "old_outbound_departure_time",
        "old_inbound_arrival_time",
        "old_outbound_departure_min",
        "old_inbound_arrival_min",
        "day_offset_days",
        "outbound_block_time_min",
        "inbound_block_time_min",
        "min_ground_time_min",
        "required_delta_min",
        "total_required_rotation_time_min",
        "rotation_constraint_available",
        "block_time_source",
        "derivation_method",
        "status",
        "skip_reason",
    ]
    if flight_pairs.empty:
        return pd.DataFrame(columns=columns)

    od_block_lookup = _build_od_connection_block_lookup(connections, config)
    hub_norm = _normalize_airport_code(config.hub)
    fp = flight_pairs.copy()
    fp["flight_no_norm"] = fp["flight_no"].map(_normalize_flight_no)
    fp["origin_norm"] = fp["origin"].map(_normalize_airport_code)
    fp["destination_norm"] = fp["destination"].map(_normalize_airport_code)
    outbound_pairs = fp[(fp["origin_norm"] == hub_norm) & (fp["destination_norm"] != hub_norm)].copy()
    inbound_pairs = fp[(fp["origin_norm"] != hub_norm) & (fp["destination_norm"] == hub_norm)].copy()
    outbound_pairs = outbound_pairs.rename(
        columns={
            "flight_no": "outbound_flight_no",
            "flight_no_norm": "outbound_flight_norm",
            "origin": "outbound_origin",
            "destination": "outbound_destination",
            "origin_norm": "outbound_origin_norm",
            "destination_norm": "outbound_destination_norm",
        }
    )
    inbound_pairs = inbound_pairs.rename(
        columns={
            "flight_no": "inbound_flight_no",
            "flight_no_norm": "inbound_flight_norm",
            "origin": "inbound_origin",
            "destination": "inbound_destination",
            "origin_norm": "inbound_origin_norm",
            "destination_norm": "inbound_destination_norm",
        }
    )
    pair_legs = outbound_pairs[
        [
            "pair",
            "outbound_flight_no",
            "outbound_flight_norm",
            "outbound_origin",
            "outbound_origin_norm",
            "outbound_destination",
            "outbound_destination_norm",
        ]
    ].merge(
        inbound_pairs[
            [
                "pair",
                "inbound_flight_no",
                "inbound_flight_norm",
                "inbound_origin",
                "inbound_origin_norm",
                "inbound_destination",
                "inbound_destination_norm",
            ]
        ],
        left_on=["pair", "outbound_destination_norm"],
        right_on=["pair", "inbound_origin_norm"],
        how="inner",
    )

    rows = []
    rotation_idx = 0
    matched_pairs = set(pair_legs["pair"].astype(str).tolist()) if not pair_legs.empty else set()
    all_pairs = set(fp["pair"].astype(str).tolist())
    for pair_id in sorted(all_pairs - matched_pairs):
        rows.append(
            _rotation_candidate_skip_row(
                rotation_idx,
                pair_id,
                "pair has no compatible IST-outbound and IST-inbound leg",
                min_ground_time_min=config.min_ground_time_min,
            )
        )
        rotation_idx += 1

    if pair_legs.empty or flight_events.empty:
        for leg in pair_legs.itertuples(index=False):
            rows.append(
                _rotation_candidate_skip_row(
                    rotation_idx,
                    leg.pair,
                    "missing modeled outbound/inbound event",
                    leg,
                    min_ground_time_min=config.min_ground_time_min,
                )
            )
            rotation_idx += 1
        return pd.DataFrame(rows, columns=columns)

    out_events = flight_events[flight_events["direction"] == "outbound"].copy()
    in_events = flight_events[flight_events["direction"] == "inbound"].copy()
    for leg in pair_legs.itertuples(index=False):
        matching_out = out_events[
            (out_events["flight_no_norm"] == leg.outbound_flight_norm)
            & (out_events["station_norm"] == leg.outbound_destination_norm)
        ]
        matching_in = in_events[
            (in_events["flight_no_norm"] == leg.inbound_flight_norm)
            & (in_events["station_norm"] == leg.inbound_origin_norm)
        ]
        if matching_out.empty or matching_in.empty:
            rows.append(
                _rotation_candidate_skip_row(
                    rotation_idx,
                    leg.pair,
                    "missing modeled outbound/inbound event",
                    leg,
                    min_ground_time_min=config.min_ground_time_min,
                )
            )
            rotation_idx += 1
            continue

        any_same_day = False
        for out_row in matching_out.itertuples(index=False):
            day_in_events = matching_in[matching_in["day"] == out_row.day]
            if day_in_events.empty:
                continue
            any_same_day = True
            for in_row in day_in_events.itertuples(index=False):
                elapsed = float(in_row.old_min) - float(out_row.old_min)
                adjusted_in_min = float(in_row.old_min)
                if elapsed <= 0:
                    adjusted_in_min += 7 * 24 * 60
                    elapsed += 7 * 24 * 60
                day_offset_days = 0
                block_info = _lookup_rotation_block_times(od_block_lookup, leg, int(out_row.day))
                if block_info is None:
                    if elapsed < config.min_ground_time_min:
                        rows.append(
                            _rotation_candidate_skip_row(
                                rotation_idx,
                                leg.pair,
                                f"derived elapsed {elapsed:.1f} < min ground {config.min_ground_time_min}",
                                leg,
                                day=int(out_row.day),
                                min_ground_time_min=config.min_ground_time_min,
                            )
                        )
                        rotation_idx += 1
                        continue
                    remaining_block = max(0.0, elapsed - float(config.min_ground_time_min))
                    outbound_block = remaining_block / 2.0
                    inbound_block = remaining_block / 2.0
                    required_delta = elapsed
                    block_time_source = "schedule_elapsed_fallback"
                    derivation_method = (
                        "approximated from schedule elapsed; O&D connection block-time match missing; "
                        "block split evenly after minimum ground time"
                    )
                else:
                    outbound_block, inbound_block, derivation_method, block_time_source = block_info
                    required_delta = float(outbound_block) + float(inbound_block) + float(config.min_ground_time_min)
                    max_reachable_delta = elapsed + 2 * float(config.max_shift_min)
                    while required_delta > max_reachable_delta + 1e-6 and day_offset_days < 3:
                        day_offset_days += 1
                        adjusted_in_min += 24 * 60
                        elapsed += 24 * 60
                        max_reachable_delta = elapsed + 2 * float(config.max_shift_min)
                    if day_offset_days:
                        derivation_method = f"{derivation_method}; inbound day_offset_days={day_offset_days} applied for rotation rollover"
                    if required_delta > max_reachable_delta + 1e-6:
                        rows.append(
                            _rotation_candidate_skip_row(
                                rotation_idx,
                                leg.pair,
                                (
                                    f"rotation required delta {required_delta:.1f} exceeds max reachable "
                                    f"{max_reachable_delta:.1f} under max_shift {config.max_shift_min}"
                                ),
                                leg,
                                day=int(out_row.day),
                                min_ground_time_min=config.min_ground_time_min,
                            )
                        )
                        rows[-1]["block_time_source"] = block_time_source
                        rows[-1]["derivation_method"] = derivation_method
                        rows[-1]["outbound_block_time_min"] = outbound_block
                        rows[-1]["inbound_block_time_min"] = inbound_block
                        rows[-1]["required_delta_min"] = required_delta
                        rows[-1]["total_required_rotation_time_min"] = required_delta
                        rows[-1]["old_outbound_departure_min"] = float(out_row.old_min)
                        rows[-1]["old_inbound_arrival_min"] = adjusted_in_min
                        rows[-1]["day_offset_days"] = day_offset_days
                        rotation_idx += 1
                        continue
                rows.append(
                    {
                        "rotation_id": f"rot_{rotation_idx:05d}",
                        "pair_id": leg.pair,
                        "station": leg.outbound_destination,
                        "day": int(out_row.day),
                        "outbound_flight_no": leg.outbound_flight_no,
                        "outbound_origin": leg.outbound_origin,
                        "outbound_destination": leg.outbound_destination,
                        "inbound_flight_no": leg.inbound_flight_no,
                        "inbound_origin": leg.inbound_origin,
                        "inbound_destination": leg.inbound_destination,
                        "outbound_event_id": out_row.event_id,
                        "inbound_event_id": in_row.event_id,
                        "old_outbound_departure_time": out_row.old_ist_time,
                        "old_inbound_arrival_time": in_row.old_ist_time,
                        "old_outbound_departure_min": float(out_row.old_min),
                        "old_inbound_arrival_min": adjusted_in_min,
                        "day_offset_days": day_offset_days,
                        "outbound_block_time_min": outbound_block,
                        "inbound_block_time_min": inbound_block,
                        "min_ground_time_min": config.min_ground_time_min,
                        "required_delta_min": required_delta,
                        "total_required_rotation_time_min": required_delta,
                        "rotation_constraint_available": True,
                        "block_time_source": block_time_source,
                        "derivation_method": derivation_method,
                        "status": "constrained",
                        "skip_reason": "",
                    }
                )
                rotation_idx += 1
        if not any_same_day:
            rows.append(
                _rotation_candidate_skip_row(
                    rotation_idx,
                    leg.pair,
                    "no inbound event for modeled outbound day",
                    leg,
                    min_ground_time_min=config.min_ground_time_min,
                )
            )
            rotation_idx += 1
    return pd.DataFrame(rows, columns=columns)


def _build_od_connection_block_lookup(connections: pd.DataFrame | None, config: ModelConfig) -> dict:
    """Infer rotation block times from TK O&D rows when the exact flight/route pair is present."""

    if connections is None or connections.empty:
        return {}
    required = {
        "inbound_flight",
        "outbound_flight",
        "Dep1",
        "Arr1",
        "Dep2",
        "Arr2",
        "connection_gap_min",
        "day",
    }
    if not required.issubset(connections.columns):
        return {}

    pair_rows = []
    leg_rows = []
    conn = connections.copy()
    if "is_tk" in conn.columns:
        conn = conn[conn["is_tk"] == True].copy()
    hub_norm = _normalize_airport_code(config.hub)
    for row in conn.itertuples(index=False):
        inbound_origin = _normalize_airport_code(getattr(row, "Dep1", ""))
        inbound_destination = _normalize_airport_code(getattr(row, "Arr1", ""))
        outbound_origin = _normalize_airport_code(getattr(row, "Dep2", ""))
        outbound_destination = _normalize_airport_code(getattr(row, "Arr2", ""))
        inbound_flight_norm = _normalize_flight_no(getattr(row, "inbound_flight"))
        outbound_flight_norm = _normalize_flight_no(getattr(row, "outbound_flight"))
        gap = getattr(row, "connection_gap_min", None)
        elapsed1 = getattr(row, "elapsed_time_1_min", np.nan)
        elapsed2 = getattr(row, "elapsed_time_2_min", np.nan)
        journey = getattr(row, "gate_to_gate_journey_time_min", getattr(row, "journey_time_min", None))
        day = int(getattr(row, "day"))
        if inbound_destination == hub_norm and not pd.isna(elapsed1) and float(elapsed1) > 0:
            leg_rows.append(
                {
                    "day": day,
                    "direction": "inbound",
                    "flight_norm": inbound_flight_norm,
                    "origin": inbound_origin,
                    "destination": inbound_destination,
                    "block_time_min": float(elapsed1),
                    "block_time_source": "elapsedtime_exact_match",
                }
            )
        if outbound_origin == hub_norm and not pd.isna(elapsed2) and float(elapsed2) > 0:
            leg_rows.append(
                {
                    "day": day,
                    "direction": "outbound",
                    "flight_norm": outbound_flight_norm,
                    "origin": outbound_origin,
                    "destination": outbound_destination,
                    "block_time_min": float(elapsed2),
                    "block_time_source": "elapsedtime_exact_match",
                }
            )
        if inbound_destination != hub_norm or outbound_origin != hub_norm:
            continue
        has_elapsed = not pd.isna(elapsed1) and not pd.isna(elapsed2) and float(elapsed1) > 0 and float(elapsed2) > 0
        has_legacy = not pd.isna(journey) and not pd.isna(gap) and (float(journey) - float(gap)) > 0
        if has_elapsed:
            inbound_block = float(elapsed1)
            outbound_block = float(elapsed2)
            source = "elapsedtime_exact_match"
        elif has_legacy:
            block_total = float(journey) - float(gap)
            inbound_block = block_total / 2.0
            outbound_block = block_total / 2.0
            source = "od_exact_match_legacy"
        else:
            continue
        pair_rows.append(
            {
                "day": day,
                "inbound_flight_norm": inbound_flight_norm,
                "outbound_flight_norm": outbound_flight_norm,
                "inbound_origin": inbound_origin,
                "inbound_destination": inbound_destination,
                "outbound_origin": outbound_origin,
                "outbound_destination": outbound_destination,
                "outbound_block_time_min": outbound_block,
                "inbound_block_time_min": inbound_block,
                "block_time_source": source,
            }
        )

    def _median_record(frame: pd.DataFrame) -> dict:
        if "elapsedtime_exact_match" in set(frame["block_time_source"].fillna("").astype(str)):
            frame = frame[frame["block_time_source"].fillna("").astype(str) == "elapsedtime_exact_match"]
        source_counts = frame["block_time_source"].fillna("").astype(str).value_counts()
        source = str(source_counts.index[0]) if len(source_counts) else ""
        return {
            "outbound_block_time_min": float(frame["outbound_block_time_min"].median()),
            "inbound_block_time_min": float(frame["inbound_block_time_min"].median()),
            "block_time_source": source,
        }

    def _median_leg_record(frame: pd.DataFrame) -> dict:
        return {
            "block_time_min": float(frame["block_time_min"].median()),
            "block_time_source": "elapsedtime_exact_match",
        }

    pair_day_lookup = {}
    pair_any_day_lookup = {}
    if pair_rows:
        block_df = pd.DataFrame(pair_rows)
        group_cols = [
            "day",
            "outbound_flight_norm",
            "outbound_origin",
            "outbound_destination",
            "inbound_flight_norm",
            "inbound_origin",
            "inbound_destination",
        ]
        pair_day_lookup = {key: _median_record(group) for key, group in block_df.groupby(group_cols, dropna=False)}
        any_day_cols = group_cols[1:]
        pair_any_day_lookup = {key: _median_record(group) for key, group in block_df.groupby(any_day_cols, dropna=False)}

    leg_day_lookup = {}
    leg_any_day_lookup = {}
    if leg_rows:
        leg_df = pd.DataFrame(leg_rows)
        leg_cols = ["day", "direction", "flight_norm", "origin", "destination"]
        leg_day_lookup = {key: _median_leg_record(group) for key, group in leg_df.groupby(leg_cols, dropna=False)}
        leg_any_cols = leg_cols[1:]
        leg_any_day_lookup = {key: _median_leg_record(group) for key, group in leg_df.groupby(leg_any_cols, dropna=False)}
    return {
        "day": pair_day_lookup,
        "any_day": pair_any_day_lookup,
        "leg_day": leg_day_lookup,
        "leg_any_day": leg_any_day_lookup,
    }


def _lookup_rotation_block_times(od_block_lookup: dict, leg, day: int) -> tuple[float, float, str, str] | None:
    if not od_block_lookup:
        return None
    day_key = (
        int(day),
        leg.outbound_flight_norm,
        leg.outbound_origin_norm,
        leg.outbound_destination_norm,
        leg.inbound_flight_norm,
        leg.inbound_origin_norm,
        leg.inbound_destination_norm,
    )
    any_day_key = day_key[1:]
    record = od_block_lookup.get("day", {}).get(day_key)
    method_suffix = "same day"
    if record is None:
        record = od_block_lookup.get("any_day", {}).get(any_day_key)
        method_suffix = "any day"
    if record is None or str(record.get("block_time_source", "")) != "elapsedtime_exact_match":
        out_day_key = (
            int(day),
            "outbound",
            leg.outbound_flight_norm,
            leg.outbound_origin_norm,
            leg.outbound_destination_norm,
        )
        in_day_key = (
            int(day),
            "inbound",
            leg.inbound_flight_norm,
            leg.inbound_origin_norm,
            leg.inbound_destination_norm,
        )
        out_any_key = out_day_key[1:]
        in_any_key = in_day_key[1:]
        out_record = od_block_lookup.get("leg_day", {}).get(out_day_key)
        in_record = od_block_lookup.get("leg_day", {}).get(in_day_key)
        leg_method_suffix = "same day"
        if out_record is None or in_record is None:
            out_record = od_block_lookup.get("leg_any_day", {}).get(out_any_key)
            in_record = od_block_lookup.get("leg_any_day", {}).get(in_any_key)
            leg_method_suffix = "any day"
        if out_record is not None and in_record is not None:
            outbound_block = out_record.get("block_time_min")
            inbound_block = in_record.get("block_time_min")
            if (
                outbound_block is not None
                and inbound_block is not None
                and not pd.isna(outbound_block)
                and not pd.isna(inbound_block)
                and float(outbound_block) > 0
                and float(inbound_block) > 0
            ):
                return (
                    float(outbound_block),
                    float(inbound_block),
                    f"ElapsedTime1/ElapsedTime2 exact route-leg match from O&D rows ({leg_method_suffix})",
                    "elapsedtime_exact_match",
                )
    if record is None:
        return None
    outbound_block = record.get("outbound_block_time_min")
    inbound_block = record.get("inbound_block_time_min")
    source = str(record.get("block_time_source", "od_exact_match_legacy"))
    if (
        outbound_block is None
        or inbound_block is None
        or pd.isna(outbound_block)
        or pd.isna(inbound_block)
        or float(outbound_block) <= 0
        or float(inbound_block) <= 0
    ):
        return None
    if source == "elapsedtime_exact_match":
        derivation = f"ElapsedTime1/ElapsedTime2 exact match from O&D rows ({method_suffix})"
    else:
        derivation = f"O&D connection block total inferred from journey_time - connection_gap ({method_suffix}); block split evenly"
    return (
        float(outbound_block),
        float(inbound_block),
        derivation,
        source,
    )


def _rotation_candidate_skip_row(
    rotation_idx: int,
    pair_id,
    reason: str,
    leg=None,
    day=None,
    min_ground_time_min=None,
) -> dict:
    return {
        "rotation_id": f"rot_{rotation_idx:05d}",
        "pair_id": pair_id,
        "station": getattr(leg, "outbound_destination", None) if leg is not None else None,
        "day": day,
        "outbound_flight_no": getattr(leg, "outbound_flight_no", None) if leg is not None else None,
        "outbound_origin": getattr(leg, "outbound_origin", None) if leg is not None else None,
        "outbound_destination": getattr(leg, "outbound_destination", None) if leg is not None else None,
        "inbound_flight_no": getattr(leg, "inbound_flight_no", None) if leg is not None else None,
        "inbound_origin": getattr(leg, "inbound_origin", None) if leg is not None else None,
        "inbound_destination": getattr(leg, "inbound_destination", None) if leg is not None else None,
        "outbound_event_id": None,
        "inbound_event_id": None,
        "old_outbound_departure_time": pd.NaT,
        "old_inbound_arrival_time": pd.NaT,
        "old_outbound_departure_min": None,
        "old_inbound_arrival_min": None,
        "day_offset_days": None,
        "outbound_block_time_min": None,
        "inbound_block_time_min": None,
        "min_ground_time_min": min_ground_time_min,
        "required_delta_min": None,
        "total_required_rotation_time_min": None,
        "rotation_constraint_available": False,
        "block_time_source": "missing",
        "derivation_method": "",
        "status": "skipped",
        "skip_reason": reason,
    }


def _normalize_flight_no(value) -> str:
    return normalize_flight_no(value)


def normalize_flight_no(value) -> str:
    """Canonical flight number: remove TK prefix/spaces and leading zeroes."""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.upper().replace(" ", "")
    if text.startswith("TK"):
        text = text[2:]
    digits = "".join(re.findall(r"\d+", text))
    if digits:
        return str(int(digits)) if int(digits) != 0 else "0"
    return text.lstrip("0") or text


def _normalize_airport_code(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()
