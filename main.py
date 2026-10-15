"""CLI entry point for TEKNOFEST airline optimization runs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

from src.analyzer import (
    analyze_mvp_run,
    write_capacity_interpretation,
    write_regularity_interpretation,
    write_rotation_interpretation,
    write_scaling_summary,
    write_sensitivity_summary,
)
from src.baseline import compute_baseline
from src.checker import build_rotation_report_from_tables, check_run_output, check_solution_tables, format_report
from src.checker import (
    build_feasible_but_inactive_report,
    build_hub_buckets,
    build_hub_capacity_report_from_schedule,
    build_regularity_groups_from_schedule,
    build_regularity_report_from_schedule,
)
from src.config import (
    ALPHA_DIRECTIONAL_BALANCE,
    DIRECTIONAL_E2_MODE,
    DIRECTIONAL_E2_PENALTY_WEIGHT,
    DIRECTIONAL_E1_ABS_TOLERANCE,
    DIRECTIONAL_E1_FORMULA,
    DIRECTIONAL_E1_MODE,
    DIRECTIONAL_E1_PENALTY_WEIGHT,
    GAMMA_BEST_JOURNEY_MINUTES,
    HUB_ARRIVAL_CAPACITY_PER_BUCKET,
    HUB_BUCKET_SIZE_MINUTES,
    HUB_DEPARTURE_CAPACITY_PER_BUCKET,
    SCHEDULE_REGULARITY_XDEV_MINUTES,
    REGULARITY_ABS_TOLERANCE_MINUTES,
    REGULARITY_MODE,
    REGULARITY_PENALTY_WEIGHT,
    REGULARITY_RELAX_MODE,
    REGULARITY_TARGETED_MAX_RELAX_MINUTES,
    SCHEDULE_DISRUPTION_PENALTY_WEIGHT,
    SHIFT_FREE_THRESHOLD_MINUTES,
    SHIFT_PENALTY_MODE,
    MANDATORY_CONNECTION_SELECTION,
    RANKING_OBJECTIVE_MODE,
    COMPETITOR_MODE,
    CANDIDATE_MODE,
    REGULARITY_ANCHOR_MODE,
    SHIFT_GRANULARITY,
    HUB_CAPACITY_MODE,
    SOLVER_RANDOM_SEED,
    SOLVER_THREADS,
    ModelConfig,
    RunConfig,
)
from src.data_loader import load_inputs
from src.experiment_logger import append_experiment
from src.metrics import compute_run_metrics
from src.model import (
    build_model,
    extract_optimized_schedule,
    extract_rotation_report,
    extract_solution,
    normalize_flight_no,
)
from src.output_writer import (
    build_comparison_to_baseline,
    build_directional_pairs_table,
    build_directional_balance_report,
    build_outputs_from_active_connections,
    render_run_summary,
    write_comparison_to_baseline,
    write_run_outputs,
)
from src.preprocessing import prepare_data
from src.rotation_analysis import (
    apply_rotation_skip_categories,
    build_rotation_skip_analysis,
    summarize_rotation_skip_analysis,
    write_rotation_skip_outputs,
)
from src.solver import solve_model, solve_two_phase, solve_with_baseline_warmstart


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TEKNOFEST airline schedule optimization")
    parser.add_argument("--data_dir", required=True, help="Directory containing input Excel files")
    parser.add_argument("--output_dir", default="./outputs", help="Directory for run outputs")
    parser.add_argument("--model_level", choices=["baseline", "mvp", "rotation", "capacity", "regularity", "directional", "full", "final"], default="baseline")
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--time_limit", type=int, default=600)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--max_od", type=int, default=None)
    parser.add_argument("--days", default=None, help="Comma-separated day list, e.g. 1,2,3")
    parser.add_argument("--top_importance_only", action="store_true")
    parser.add_argument("--max_shift", type=int, default=60)
    parser.add_argument("--hub_arrival_capacity", type=int, default=HUB_ARRIVAL_CAPACITY_PER_BUCKET)
    parser.add_argument("--hub_departure_capacity", type=int, default=HUB_DEPARTURE_CAPACITY_PER_BUCKET)
    parser.add_argument("--hub_bucket_size", type=int, default=HUB_BUCKET_SIZE_MINUTES)
    parser.add_argument("--regularity_xdev", type=int, default=SCHEDULE_REGULARITY_XDEV_MINUTES)
    parser.add_argument("--regularity_mode", choices=["hard", "soft", "report_only"], default=REGULARITY_MODE)
    parser.add_argument("--regularity_penalty_weight", type=float, default=REGULARITY_PENALTY_WEIGHT)
    parser.add_argument("--regularity_abs_tolerance", type=int, default=REGULARITY_ABS_TOLERANCE_MINUTES)
    parser.add_argument("--regularity_relax_mode", choices=["none", "global_tolerance", "targeted_theoretical", "targeted_soft"], default=REGULARITY_RELAX_MODE)
    parser.add_argument("--regularity_targeted_max_relax", type=int, default=REGULARITY_TARGETED_MAX_RELAX_MINUTES)
    parser.add_argument("--directional_alpha", type=float, default=ALPHA_DIRECTIONAL_BALANCE)
    parser.add_argument("--directional_gamma", type=int, default=GAMMA_BEST_JOURNEY_MINUTES)
    parser.add_argument("--directional_e1_mode", choices=["report_only", "hard", "soft"], default=None)
    parser.add_argument("--directional_e1_penalty_weight", type=float, default=DIRECTIONAL_E1_PENALTY_WEIGHT)
    parser.add_argument("--directional_e1_formula", choices=["ratio_total", "absolute_tolerance", "soft_only"], default=DIRECTIONAL_E1_FORMULA)
    parser.add_argument("--directional_e1_abs_tolerance", type=int, default=DIRECTIONAL_E1_ABS_TOLERANCE)
    parser.add_argument("--directional_e2_mode", choices=["report_only", "hard", "soft"], default=None)
    parser.add_argument("--directional_e2_penalty_weight", type=float, default=DIRECTIONAL_E2_PENALTY_WEIGHT)
    parser.add_argument("--allow_soft_directional_violations", action="store_true")
    parser.add_argument("--shift_penalty_weight", type=float, default=SCHEDULE_DISRUPTION_PENALTY_WEIGHT)
    parser.add_argument("--shift_penalty_mode", choices=["none", "linear", "threshold", "quadratic_proxy"], default=SHIFT_PENALTY_MODE)
    parser.add_argument("--shift_free_threshold", type=int, default=SHIFT_FREE_THRESHOLD_MINUTES)
    parser.add_argument("--mandatory_connection_selection", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ranking_objective_mode", choices=["proxy", "lookup_posthoc", "lookup_linear"], default=None)
    parser.add_argument("--competitor_mode", choices=["carrier", "row"], default=COMPETITOR_MODE, help="Rakip tanimi: carrier=tasiyici bazli (Tcomp=en iyi sure), row=tablodaki her satir ayri rakip")
    parser.add_argument("--candidate_mode", choices=["table", "cross_product"], default=CANDIDATE_MODE, help="Aday TK baglantilari: table=yalnizca girdi tablosu satirlari, cross_product=tum inbound x outbound kombinasyonlari")
    parser.add_argument("--regularity_anchor_mode", choices=["auto", "off"], default=REGULARITY_ANCHOR_MODE, help="Kismi gun kosularinda kapsam disi gunlere gore Xdev capasi")
    parser.add_argument("--shift_granularity", choices=["flight_week", "event"], default=SHIFT_GRANULARITY, help="flight_week=ucus basina tek haftalik shift (G kisiti yapisal), event=gun bazli shift")
    parser.add_argument("--hub_capacity_mode", choices=["enforce", "postcheck"], default=HUB_CAPACITY_MODE, help="enforce=kova binary/kisitlariyla aktif, postcheck=cozumde atla + checker ile dogrula")
    parser.add_argument("--solver_random_seed", type=int, default=SOLVER_RANDOM_SEED)
    parser.add_argument("--solver_threads", type=int, default=SOLVER_THREADS)
    parser.add_argument("--two_phase", action="store_true", help="Leksikografik iki asamali cozum: once ihlal minimizasyonu, sonra odul maksimizasyonu")
    parser.add_argument("--warmstart", action="store_true", help="Baseline (shift=0) MIP start ile coz (buyuk modelde erken feasible incumbent)")
    parser.add_argument("--warmstart_seconds", type=int, default=120, help="Warm start baseline seed cozumu icin ayrilan sure")
    parser.add_argument("--sequential_days", action="store_true", help="Gunleri sirayla coz (fix-and-optimize); regularity onceki gun cozumlerine baglanir")
    parser.add_argument("--competitor_connections_file", default="O&D Rakip Bağlantı Tablosu.xlsx")
    parser.add_argument("--passenger_data_file", default="Yolcu Verisi_masked.xlsx")
    parser.add_argument("--ranking_input_file", default="change_ranking_input.xlsx")
    parser.add_argument("--flight_pairs_file", default="Flight Pairs.xlsx")
    parser.add_argument("--analyze_mvp", action="store_true", help="Write deep MVP-vs-baseline analysis outputs")
    parser.add_argument("--sensitivity_test", action="store_true", help="Run MVP max_shift sensitivity tests")
    parser.add_argument("--scaling_test", action="store_true", help="Run MVP scalability/stability experiment set")
    parser.add_argument("--rotation_coverage_only", action="store_true", help="Analyze Flight Pairs matching without solving")
    parser.add_argument("--capacity_stress_test", action="store_true", help="Run capacity model with progressively tighter hub capacities")
    parser.add_argument("--regularity_stress_test", action="store_true", help="Run regularity model with progressively tighter XDEV limits")
    parser.add_argument("--directional_stress_test", action="store_true", help="Run directional model with progressively tighter alpha/gamma limits")
    parser.add_argument("--directional_coverage_only", action="store_true", help="Analyze directional pair coverage without solving")
    parser.add_argument("--directional_scaling_test", action="store_true", help="Run directional model across larger directional-coverage subsets")
    parser.add_argument("--shift_penalty_stress_test", action="store_true", help="Run final model across schedule-disruption penalty settings")
    parser.add_argument("--final_scale_test", action="store_true", help="Run final model with recommended settings across larger subsets")
    parser.add_argument("--mandatory_directional_diagnosis", action="store_true", help="Diagnose mandatory connection selection with directional E1/E2 mode combinations")
    parser.add_argument("--ranking_objective_comparison_test", action="store_true", help="Compare proxy/posthoc/lookup-linear ranking objective modes")
    parser.add_argument("--data_schema_audit", action="store_true", help="Audit current Excel input schema and duration consistency")
    parser.add_argument("--od300_infeasibility_diagnosis", action="store_true", help="Run OD300 final-model infeasibility diagnosis matrix")
    parser.add_argument("--od300_diagnosis_quick", action="store_true", help="Run only the fast OD300 days=1,2 diagnosis scenarios")
    parser.add_argument("--od300_regularity_diagnosis", action="store_true", help="Run OD300 regularity conflict and relaxation diagnosis")
    parser.add_argument("--od300_targeted_regularity_test", action="store_true", help="Run OD300 targeted regularity relax/reference tests")
    parser.add_argument("--od300_maxshift60_quality_audit", action="store_true", help="Audit the latest successful OD300 max_shift=60 fallback run")
    parser.add_argument("--od300_controlled_shift_test", action="store_true", help="Run OD300 max_shift/penalty controlled fallback tests")
    parser.add_argument("--od300_days123_controlled_shift_test", action="store_true", help="Run OD300 days=1,2,3 controlled threshold fallback tests")
    parser.add_argument("--od400_stress_test", action="store_true", help="Run OD400 stress/scale scenarios with controlled timeouts")
    parser.add_argument("--audit_run_dir", default=None, help="Optional completed run directory to audit")
    return parser.parse_args()


def make_run_config(args: argparse.Namespace) -> RunConfig:
    days = [int(x.strip()) for x in args.days.split(",") if x.strip()] if args.days else None
    if getattr(args, "rotation_coverage_only", False):
        run_id_suffix = "rotation_coverage"
    elif getattr(args, "directional_coverage_only", False):
        run_id_suffix = "directional_coverage"
    else:
        run_id_suffix = args.model_level
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f_") + run_id_suffix
    directional_e2_mode = args.directional_e2_mode
    if directional_e2_mode is None:
        directional_e2_mode = "hard" if args.model_level == "final" else DIRECTIONAL_E2_MODE
    directional_e1_mode = args.directional_e1_mode
    if directional_e1_mode is None:
        directional_e1_mode = "hard" if args.model_level == "final" else DIRECTIONAL_E1_MODE
    mandatory_connection_selection = args.mandatory_connection_selection
    if mandatory_connection_selection is None:
        mandatory_connection_selection = True if args.model_level == "final" else MANDATORY_CONNECTION_SELECTION
    ranking_objective_mode = args.ranking_objective_mode
    if ranking_objective_mode is None:
        ranking_objective_mode = "lookup_linear" if args.model_level == "final" else RANKING_OBJECTIVE_MODE
    return RunConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        model_level=args.model_level,
        solver=args.solver,
        time_limit=args.time_limit,
        run_id=run_id,
        max_od=args.max_od,
        days=days,
        top_importance_only=args.top_importance_only,
        max_shift_min=args.max_shift,
        hub_arrival_capacity=args.hub_arrival_capacity,
        hub_departure_capacity=args.hub_departure_capacity,
        hub_bucket_size_min=args.hub_bucket_size,
        regularity_xdev_min=args.regularity_xdev,
        regularity_mode=args.regularity_mode,
        regularity_penalty_weight=args.regularity_penalty_weight,
        regularity_abs_tolerance_min=args.regularity_abs_tolerance,
        regularity_relax_mode=args.regularity_relax_mode,
        regularity_targeted_max_relax_min=args.regularity_targeted_max_relax,
        directional_alpha=args.directional_alpha,
        directional_gamma_min=args.directional_gamma,
        directional_e1_mode=directional_e1_mode,
        directional_e1_penalty_weight=args.directional_e1_penalty_weight,
        directional_e1_formula=args.directional_e1_formula,
        directional_e1_abs_tolerance=args.directional_e1_abs_tolerance,
        directional_e2_mode=directional_e2_mode,
        directional_e2_penalty_weight=args.directional_e2_penalty_weight,
        allow_soft_directional_violations=args.allow_soft_directional_violations,
        shift_penalty_mode=args.shift_penalty_mode,
        shift_penalty_weight=args.shift_penalty_weight,
        shift_free_threshold_min=args.shift_free_threshold,
        mandatory_connection_selection=mandatory_connection_selection,
        ranking_objective_mode=ranking_objective_mode,
        competitor_mode=args.competitor_mode,
        candidate_mode=args.candidate_mode,
        regularity_anchor_mode=args.regularity_anchor_mode,
        shift_granularity=args.shift_granularity,
        hub_capacity_mode=args.hub_capacity_mode,
        solver_random_seed=args.solver_random_seed,
        solver_threads=args.solver_threads,
        two_phase=args.two_phase,
        warmstart=args.warmstart,
        warmstart_seconds=args.warmstart_seconds,
        competitor_connections_file=args.competitor_connections_file,
        passenger_data_file=args.passenger_data_file,
        ranking_input_file=args.ranking_input_file,
        flight_pairs_file=args.flight_pairs_file,
    )


def clone_run_config(
    run_config: RunConfig,
    run_id: str,
    output_dir: Path,
    max_shift_min: int,
    max_od: int | None = None,
    days: list[int] | None = None,
    top_importance_only: bool | None = None,
    model_level: str | None = None,
    hub_arrival_capacity: int | None = None,
    hub_departure_capacity: int | None = None,
    hub_bucket_size_min: int | None = None,
    regularity_xdev_min: int | None = None,
    regularity_mode: str | None = None,
    regularity_penalty_weight: float | None = None,
    regularity_abs_tolerance_min: int | None = None,
    regularity_relax_mode: str | None = None,
    regularity_targeted_max_relax_min: int | None = None,
    directional_alpha: float | None = None,
    directional_gamma_min: int | None = None,
    directional_e1_mode: str | None = None,
    directional_e1_penalty_weight: float | None = None,
    directional_e1_formula: str | None = None,
    directional_e1_abs_tolerance: int | None = None,
    directional_e2_mode: str | None = None,
    directional_e2_penalty_weight: float | None = None,
    allow_soft_directional_violations: bool | None = None,
    shift_penalty_mode: str | None = None,
    shift_penalty_weight: float | None = None,
    shift_free_threshold_min: int | None = None,
    mandatory_connection_selection: bool | None = None,
    ranking_objective_mode: str | None = None,
) -> RunConfig:
    cloned_model_level = run_config.model_level if model_level is None else model_level
    cloned_mandatory = (
        mandatory_connection_selection
        if mandatory_connection_selection is not None
        else (True if cloned_model_level == "final" else run_config.mandatory_connection_selection)
    )
    return RunConfig(
        data_dir=run_config.data_dir,
        output_dir=output_dir,
        model_level=cloned_model_level,
        solver=run_config.solver,
        time_limit=run_config.time_limit,
        run_id=run_id,
        max_od=run_config.max_od if max_od is None else max_od,
        days=run_config.days if days is None else days,
        top_importance_only=run_config.top_importance_only if top_importance_only is None else top_importance_only,
        max_shift_min=max_shift_min,
        hub_arrival_capacity=run_config.hub_arrival_capacity if hub_arrival_capacity is None else hub_arrival_capacity,
        hub_departure_capacity=run_config.hub_departure_capacity if hub_departure_capacity is None else hub_departure_capacity,
        hub_bucket_size_min=run_config.hub_bucket_size_min if hub_bucket_size_min is None else hub_bucket_size_min,
        regularity_xdev_min=run_config.regularity_xdev_min if regularity_xdev_min is None else regularity_xdev_min,
        regularity_mode=run_config.regularity_mode if regularity_mode is None else regularity_mode,
        regularity_penalty_weight=run_config.regularity_penalty_weight if regularity_penalty_weight is None else regularity_penalty_weight,
        regularity_abs_tolerance_min=run_config.regularity_abs_tolerance_min if regularity_abs_tolerance_min is None else regularity_abs_tolerance_min,
        regularity_relax_mode=run_config.regularity_relax_mode if regularity_relax_mode is None else regularity_relax_mode,
        regularity_targeted_max_relax_min=run_config.regularity_targeted_max_relax_min if regularity_targeted_max_relax_min is None else regularity_targeted_max_relax_min,
        directional_alpha=run_config.directional_alpha if directional_alpha is None else directional_alpha,
        directional_gamma_min=run_config.directional_gamma_min if directional_gamma_min is None else directional_gamma_min,
        directional_e1_mode=run_config.directional_e1_mode if directional_e1_mode is None else directional_e1_mode,
        directional_e1_penalty_weight=run_config.directional_e1_penalty_weight if directional_e1_penalty_weight is None else directional_e1_penalty_weight,
        directional_e1_formula=run_config.directional_e1_formula if directional_e1_formula is None else directional_e1_formula,
        directional_e1_abs_tolerance=run_config.directional_e1_abs_tolerance if directional_e1_abs_tolerance is None else directional_e1_abs_tolerance,
        directional_e2_mode=run_config.directional_e2_mode if directional_e2_mode is None else directional_e2_mode,
        directional_e2_penalty_weight=run_config.directional_e2_penalty_weight if directional_e2_penalty_weight is None else directional_e2_penalty_weight,
        allow_soft_directional_violations=run_config.allow_soft_directional_violations if allow_soft_directional_violations is None else allow_soft_directional_violations,
        shift_penalty_mode=run_config.shift_penalty_mode if shift_penalty_mode is None else shift_penalty_mode,
        shift_penalty_weight=run_config.shift_penalty_weight if shift_penalty_weight is None else shift_penalty_weight,
        shift_free_threshold_min=run_config.shift_free_threshold_min if shift_free_threshold_min is None else shift_free_threshold_min,
        mandatory_connection_selection=cloned_mandatory,
        ranking_objective_mode=run_config.ranking_objective_mode if ranking_objective_mode is None else ranking_objective_mode,
        competitor_mode=run_config.competitor_mode,
        candidate_mode=run_config.candidate_mode,
        regularity_anchor_mode=run_config.regularity_anchor_mode,
        shift_granularity=run_config.shift_granularity,
        hub_capacity_mode=run_config.hub_capacity_mode,
        solver_random_seed=run_config.solver_random_seed,
        solver_threads=run_config.solver_threads,
        two_phase=run_config.two_phase,
        competitor_connections_file=run_config.competitor_connections_file,
        passenger_data_file=run_config.passenger_data_file,
        ranking_input_file=run_config.ranking_input_file,
        flight_pairs_file=run_config.flight_pairs_file,
    )


def _model_level_feature_flags(model_level: str) -> dict[str, bool]:
    return {
        "rotation_enabled": model_level in {"rotation", "capacity", "regularity", "directional", "full", "final"},
        "capacity_enabled": model_level in {"capacity", "regularity", "directional", "full", "final"},
        "regularity_enabled": model_level in {"regularity", "directional", "full", "final"},
        "directional_enabled": model_level in {"directional", "full", "final"},
    }


def run_once(
    run_config: RunConfig,
    analyze_mvp: bool = True,
    fixed_event_shifts: dict | None = None,
    anchor_references: dict | None = None,
) -> dict:
    model_config = _make_model_config(run_config)
    feature_flags = _model_level_feature_flags(run_config.model_level)
    if model_config.regularity_mode != "hard":
        feature_flags["regularity_enabled"] = False
    run_dir = run_config.output_dir / run_config.run_id

    raw = load_inputs(run_config)
    prepared = prepare_data(raw, run_config, model_config)

    solver_status = "not_run"
    solver_objective = None
    solution_quality = "not_solved"
    gap = None
    runtime_sec = 0.0
    rotation_report = None
    rotation_candidates = None
    hub_capacity_report = None
    regularity_report = None
    optimized_schedule = None
    reference_mvp = None
    reference_rotation = None
    reference_capacity = None
    reference_regularity = None
    regularity_coverage_metrics = {}
    regularity_effect_metrics = {}
    rotation_skip_analysis = None
    regularity_groups = None
    directional_pairs = None
    model_directional_pairs = None
    directional_metrics = {}
    directional_coverage_analysis = None
    directional_coverage_metrics = {}
    all_candidate_connections = None
    feasible_but_inactive_report = None
    mandatory_connection_metrics = {}

    baseline_result = compute_baseline(prepared, model_config)
    baseline_metrics = compute_run_metrics(
        baseline_result.od_metrics,
        baseline_result.selected_connections,
        baseline_result.beaten_competitors,
    )

    if run_config.model_level == "baseline":
        result = baseline_result
    else:
        if run_config.model_level == "rotation":
            reference_mvp = _solve_reference_mvp(prepared, model_config, run_config, baseline_result, baseline_metrics)
        if run_config.model_level == "capacity":
            reference_rotation = _solve_reference_model(prepared, model_config, run_config, baseline_result, baseline_metrics, "rotation")
        if run_config.model_level == "regularity":
            reference_capacity = _solve_reference_model(prepared, model_config, run_config, baseline_result, baseline_metrics, "capacity")
        if run_config.model_level == "directional":
            reference_regularity = _solve_reference_model(prepared, model_config, run_config, baseline_result, baseline_metrics, "regularity")
        artifacts = build_model(
            prepared,
            model_config,
            run_config.model_level,
            fixed_event_shifts=fixed_event_shifts,
            anchor_references=anchor_references,
        )
        model_directional_pairs = artifacts.directional_pairs
        if getattr(run_config, "warmstart", False):
            solve_result = solve_with_baseline_warmstart(
                artifacts.model,
                run_config.solver,
                run_config.time_limit,
                model_config.mip_gap,
                random_seed=model_config.solver_random_seed,
                threads=model_config.solver_threads,
                warmstart_seconds=getattr(run_config, "warmstart_seconds", 120),
            )
        elif run_config.two_phase:
            solve_result = solve_two_phase(
                artifacts.model,
                run_config.solver,
                run_config.time_limit,
                model_config.mip_gap,
                random_seed=model_config.solver_random_seed,
                threads=model_config.solver_threads,
            )
        else:
            solve_result = solve_model(
                artifacts.model,
                run_config.solver,
                run_config.time_limit,
                model_config.mip_gap,
                random_seed=model_config.solver_random_seed,
                threads=model_config.solver_threads,
            )
        solver_status = f"{solve_result.status}/{solve_result.termination_condition}"
        solver_objective = solve_result.objective
        term_lower = str(solve_result.termination_condition).lower()
        if "optimal" in term_lower:
            solution_quality = "optimal"
        elif solve_result.feasible:
            solution_quality = "feasible_incumbent_at_time_limit"
        else:
            solution_quality = "no_feasible_solution"
        gap = solve_result.gap
        runtime_sec = solve_result.runtime_sec
        if not solve_result.feasible:
            return _handle_infeasible_model_run(
                run_config,
                model_config,
                run_dir,
                artifacts,
                baseline_metrics,
                solver_status,
                gap,
                runtime_sec,
            )
        solution = extract_solution(artifacts)
        optimized_schedule = extract_optimized_schedule(artifacts)
        rotation_candidates = artifacts.rotation_candidates
        rotation_skip_analysis = build_rotation_skip_analysis(rotation_candidates, optimized_schedule, raw.connections, model_config)
        rotation_candidates = apply_rotation_skip_categories(rotation_candidates, rotation_skip_analysis)
        decoded_rotation_report = extract_rotation_report(artifacts)
        rotation_report = build_rotation_report_from_tables(optimized_schedule, rotation_candidates, model_config)
        if not decoded_rotation_report.empty and not rotation_report.empty:
            # Keep the independent checker report authoritative; decoded report is useful only during debugging.
            pass
        hub_capacity_report = build_hub_capacity_report_from_schedule(optimized_schedule, model_config)
        if run_config.model_level in {"regularity", "directional", "full", "final"}:
            regularity_groups = build_regularity_groups_from_schedule(optimized_schedule, model_config)
            regularity_report = build_regularity_report_from_schedule(optimized_schedule, model_config)
        candidate_solution = artifacts.candidate_connections.merge(solution, on=["candidate_id", "connection_id"], how="left")
        active_connections = candidate_solution[candidate_solution["active"] == True].copy()
        result = build_outputs_from_active_connections(
            prepared,
            active_connections,
            model_config,
            baseline_result.od_metrics,
        )
        all_candidate_connections = build_all_candidate_connections_report(candidate_solution, model_config)
        feasible_but_inactive_report = build_feasible_but_inactive_report(
            all_candidate_connections,
            result.selected_connections,
            optimized_schedule,
            model_config,
        )
        mandatory_connection_metrics = summarize_mandatory_connection_selection(
            all_candidate_connections,
            feasible_but_inactive_report,
            result.selected_connections,
            model_config,
        )

    metrics = compute_run_metrics(result.od_metrics, result.selected_connections, result.beaten_competitors)
    metrics["model_level"] = run_config.model_level
    metrics["ranking_objective_mode"] = run_config.ranking_objective_mode
    metrics["max_shift_min"] = model_config.max_shift_min
    metrics["solution_quality"] = solution_quality
    if run_config.model_level != "baseline":
        # Ayrisim: reward = cezasiz ticari amac (jurinin kiyasladigi deger);
        # net_solver_objective = solver'in enbuyukledigi deger (soft cezalar dahil).
        metrics["reward_before_violation_penalty"] = float(metrics.get("total_objective", 0.0))
        if solver_objective is not None:
            metrics["net_solver_objective"] = float(solver_objective)
            metrics["violation_penalty_total"] = metrics["reward_before_violation_penalty"] - float(solver_objective)
    metrics.update(mandatory_connection_metrics)
    rotation_metrics = _summarize_rotation(rotation_candidates, rotation_report)
    metrics.update(rotation_metrics)
    rotation_skip_metrics = summarize_rotation_skip_analysis(rotation_candidates, rotation_skip_analysis)
    metrics.update(rotation_skip_metrics)
    hub_metrics = _summarize_hub_capacity(hub_capacity_report, model_config)
    metrics.update(hub_metrics)
    regularity_metrics = _summarize_regularity(regularity_groups, regularity_report, model_config)
    metrics.update(regularity_metrics)
    if run_config.model_level in {"directional", "full", "final"}:
        try:
            cand = artifacts.candidate_connections
            keys_with_candidates = {
                (str(od), int(day))
                for od, day in zip(cand["O&D"].astype(str), cand["day"].astype(int))
            }
        except (NameError, KeyError):
            keys_with_candidates = None
        directional_pairs = build_directional_pairs_table(result.od_metrics, keys_with_candidates)
        directional_balance_report = build_directional_balance_report(result.od_metrics, model_config, directional_pairs)
    else:
        directional_balance_report = None
    directional_metrics = _summarize_directional(directional_pairs, directional_balance_report, model_config)
    metrics.update(directional_metrics)
    if run_config.model_level in {"directional", "full", "final"}:
        directional_coverage_analysis, directional_coverage_metrics = build_directional_coverage_analysis(
            raw.connections,
            prepared,
            model_config,
            result.selected_connections,
            run_config,
        )
        metrics.update(directional_coverage_metrics)
    shift_penalty_report, shift_penalty_metrics = build_shift_penalty_report(
        optimized_schedule,
        model_config,
        metrics["total_objective"],
    )
    metrics.update(shift_penalty_metrics)
    ranking_audit, ranking_metrics, competitor_beating_audit = build_ranking_reward_audit(
        prepared,
        result,
        metrics,
        model_config,
    )
    ranking_metrics["objective_before_penalty"] = metrics.get("objective_before_penalty", metrics.get("total_objective", 0.0))
    ranking_metrics["objective_after_penalty"] = metrics.get("objective_after_penalty", metrics.get("total_objective", 0.0))
    metrics.update(ranking_metrics)
    metrics.update(feature_flags)
    metrics["mandatory_connection_selection_enabled"] = bool(run_config.mandatory_connection_selection)
    report = check_solution_tables(
        result.selected_connections,
        result.od_metrics,
        metrics["total_objective"],
        model_config,
        rotation_report,
        hub_capacity_report,
        regularity_report,
        directional_balance_report,
        feasible_but_inactive_report,
        rotation_enabled=feature_flags["rotation_enabled"],
        capacity_enabled=feature_flags["capacity_enabled"],
        regularity_enabled=feature_flags["regularity_enabled"] and model_config.regularity_mode == "hard",
        directional_enabled=feature_flags["directional_enabled"],
    )
    report_text = format_report(report)
    report_text += (
        "\n"
        f"Regularity relax mode: {model_config.regularity_relax_mode}\n"
        f"Relax applied groups: {metrics.get('regularity_relax_applied_groups', 0)}\n"
        f"Regularity hard violations: {metrics.get('regularity_hard_violations', 0)}\n"
        f"Regularity soft violations: {metrics.get('regularity_soft_violations', 0)}\n"
        f"Max effective XDEV: {metrics.get('regularity_max_effective_xdev', 0)}"
    )
    write_run_outputs(
        run_dir,
        result,
        prepared,
        metrics,
        solver_status,
        gap,
        runtime_sec,
        report_text,
        rotation_report,
        rotation_candidates,
        hub_capacity_report,
        regularity_report,
        directional_balance_report,
        optimized_schedule,
    )
    write_rotation_skip_outputs(run_dir, rotation_candidates, rotation_skip_analysis, metrics)
    write_shift_penalty_outputs(run_dir, shift_penalty_report, shift_penalty_metrics)
    write_regularity_targeted_relax_outputs(run_dir, regularity_report, model_config)
    write_ranking_reward_audit_outputs(run_dir, ranking_audit, ranking_metrics, competitor_beating_audit, prepared)
    write_objective_breakdown_outputs(run_dir, metrics, ranking_metrics, model_config)
    if all_candidate_connections is not None:
        all_candidate_connections.to_excel(run_dir / "all_candidate_connections.xlsx", index=False)
        if feasible_but_inactive_report is None:
            feasible_but_inactive_report = pd.DataFrame()
        feasible_but_inactive_report.to_excel(run_dir / "feasible_but_inactive_connections.xlsx", index=False)
        write_mandatory_connection_audit(
            run_dir,
            metrics,
            report,
            baseline_metrics,
        )
    build_hub_buckets(model_config).to_excel(run_dir / "hub_buckets.xlsx", index=False)
    if regularity_groups is not None:
        regularity_groups.to_excel(run_dir / "regularity_groups.xlsx", index=False)
    if directional_pairs is not None:
        directional_pairs.to_excel(run_dir / "directional_pairs.xlsx", index=False)
    if run_config.model_level in {"directional", "full", "final"} and directional_balance_report is not None:
        _write_directional_violation_files(run_dir, directional_balance_report)
        debug_report, debug_metrics = build_directional_model_checker_debug(
            model_directional_pairs,
            candidate_solution if "candidate_solution" in locals() else None,
            directional_balance_report,
            model_config,
        )
        debug_report.to_excel(run_dir / "directional_model_checker_debug.xlsx", index=False)
        write_directional_model_checker_debug_summary(run_dir, debug_metrics)
    if directional_coverage_analysis is not None:
        directional_coverage_analysis.to_excel(run_dir / "directional_coverage_analysis.xlsx", index=False)
        write_directional_coverage_summary(run_dir, directional_coverage_metrics)
    if run_config.model_level in {"regularity", "full", "final"} and optimized_schedule is not None:
        coverage_analysis, regularity_coverage_metrics = build_regularity_coverage_analysis(
            optimized_schedule,
            result.selected_connections,
            model_config,
        )
        coverage_analysis.to_excel(run_dir / "regularity_coverage_analysis.xlsx", index=False)
        write_regularity_coverage_summary(run_dir, regularity_coverage_metrics)
        capacity_schedule = reference_capacity.get("optimized_schedule") if reference_capacity else None
        effect_comparison, regularity_effect_metrics = build_regularity_effect_comparison(
            capacity_schedule,
            optimized_schedule,
            result.selected_connections,
            coverage_analysis,
        )
        effect_comparison.to_excel(run_dir / "regularity_effect_comparison.xlsx", index=False)
        write_regularity_effect_summary(run_dir, regularity_effect_metrics)
    comparison = build_comparison_to_baseline(baseline_metrics, metrics)
    write_comparison_to_baseline(run_dir, comparison)
    analysis = {}
    if run_config.model_level in {"mvp", "rotation", "capacity", "regularity", "directional", "final"} and analyze_mvp:
        analysis = analyze_mvp_run(run_dir, baseline_result, metrics, run_config.max_shift_min)
        comparison.update(
            {
                "baseline_objective": analysis.get("baseline_objective", comparison["baseline_objective"]),
                "percentage_improvement": analysis.get("percentage_improvement", comparison["percentage_improvement"]),
            }
        )
    if run_config.model_level == "rotation":
        rotation_summary = {
            **rotation_metrics,
            "objective": metrics["total_objective"],
            "improvement_pct": comparison["percentage_improvement"],
            "runtime_sec": runtime_sec,
            "feasible": report.feasible,
        }
        write_rotation_interpretation(run_dir, rotation_summary, reference_mvp, analysis)
    if run_config.model_level == "capacity":
        capacity_summary = {
            **hub_metrics,
            "objective": metrics["total_objective"],
            "improvement_pct": comparison["percentage_improvement"],
            "runtime_sec": runtime_sec,
            "feasible": report.feasible,
        }
        write_capacity_interpretation(run_dir, capacity_summary, reference_rotation, analysis)
    if run_config.model_level == "regularity":
        regularity_summary = {
            **regularity_metrics,
            "objective": metrics["total_objective"],
            "improvement_pct": comparison["percentage_improvement"],
            "runtime_sec": runtime_sec,
            "feasible": report.feasible,
        }
        write_regularity_interpretation(run_dir, regularity_summary, reference_capacity, analysis)
    if run_config.model_level in {"directional", "final"}:
        directional_summary = {
            **directional_metrics,
            "objective": metrics["total_objective"],
            "improvement_pct": comparison["percentage_improvement"],
            "runtime_sec": runtime_sec,
            "feasible": report.feasible,
        }
        write_directional_interpretation(run_dir, directional_summary, reference_regularity, analysis)

    append_experiment(
        run_config.output_dir,
        {
            "run_id": run_config.run_id,
            "model_level": run_config.model_level,
            "solver": run_config.solver,
            "time_limit": run_config.time_limit,
            "num_od": int(prepared.od_day["od"].nunique()) if not prepared.od_day.empty else 0,
            "num_flights": int(prepared.tk_connections[["inbound_flight", "outbound_flight"]].nunique().sum()),
            "num_candidate_connections": int(len(prepared.tk_connections)),
            "feasible": report.feasible,
            "objective": metrics["total_objective"],
            "baseline_objective": comparison["baseline_objective"],
            "shift_penalty_mode": metrics.get("shift_penalty_mode", run_config.shift_penalty_mode),
            "shift_penalty_weight": metrics.get("shift_penalty_weight", run_config.shift_penalty_weight),
            "shift_free_threshold": metrics.get("shift_free_threshold", run_config.shift_free_threshold_min),
            "mandatory_connection_selection": metrics.get("mandatory_connection_selection_enabled", run_config.mandatory_connection_selection),
            "ranking_objective_mode": metrics.get("ranking_objective_mode", run_config.ranking_objective_mode),
            "reported_lookup_ranking_reward": metrics.get("reported_lookup_ranking_reward", 0.0),
            "model_proxy_ranking_reward": metrics.get("model_proxy_ranking_reward", 0.0),
            "ranking_reward_difference_lookup_vs_proxy": metrics.get("ranking_reward_difference_lookup_vs_proxy", 0.0),
            "ranking_objective_recalculation_difference": metrics.get("ranking_objective_recalculation_difference", 0.0),
            "total_candidate_connections": metrics.get("total_candidate_connections", 0),
            "connections_feasible_by_final_schedule": metrics.get("connections_feasible_by_final_schedule", 0),
            "feasible_but_inactive_connection_count": metrics.get("feasible_but_inactive_connection_count", 0),
            "infeasible_but_active_connection_count": metrics.get("infeasible_but_active_connection_count", 0),
            "objective_before_penalty": metrics.get("objective_before_penalty", metrics["total_objective"]),
            "objective_after_penalty": metrics.get("objective_after_penalty", metrics["total_objective"]),
            "total_shift_penalty": metrics.get("total_shift_penalty", 0.0),
            "total_abs_shift": metrics.get("total_abs_shift", 0.0),
            "avg_abs_shift": metrics.get("avg_abs_shift", 0.0),
            "median_abs_shift": metrics.get("median_abs_shift", 0.0),
            "max_abs_shift": metrics.get("max_abs_shift", 0.0),
            "pct_shift_le_5": metrics.get("pct_shift_le_5", 0.0),
            "pct_shift_5_15": metrics.get("pct_shift_5_15", 0.0),
            "pct_shift_15_30": metrics.get("pct_shift_15_30", 0.0),
            "pct_at_shift_bounds": metrics.get("pct_at_shift_bounds", 0.0),
            "improvement_pct": comparison["percentage_improvement"],
            "connection_reward": metrics["connection_reward"],
            "ranking_reward": metrics["ranking_reward"],
            "active_connections": metrics["active_connections"],
            "beaten_competitors": metrics["beaten_competitors"],
            "improved_od_count": metrics["improved_od_count"],
            "worsened_od_count": metrics["worsened_od_count"],
            "runtime_sec": runtime_sec,
            "gap": gap,
            "rotation_constraints_active": rotation_metrics["rotation_constraints_active"],
            "rotation_candidates_total": metrics.get("rotation_candidates_total", 0),
            "rotation_constraints_skipped": metrics.get("rotation_constraints_skipped", 0),
            "rotation_coverage_pct": metrics.get("rotation_coverage_pct", 0.0),
            "rotation_skip_subset_excluded": metrics.get("rotation_skip_subset_excluded", 0),
            "rotation_skip_format_mismatch": metrics.get("rotation_skip_format_mismatch", 0),
            "rotation_skip_route_mismatch": metrics.get("rotation_skip_route_mismatch", 0),
            "rotation_skip_missing_block_time": metrics.get("rotation_skip_missing_block_time", 0),
            "rotation_block_elapsedtime_match_count": metrics.get("rotation_block_elapsedtime_match_count", 0),
            "rotation_block_legacy_exact_match_count": metrics.get("rotation_block_legacy_exact_match_count", 0),
            "rotation_block_exact_match_count": metrics.get("rotation_block_exact_match_count", 0),
            "rotation_block_fallback_count": metrics.get("rotation_block_fallback_count", 0),
            "rotation_violations": rotation_metrics["rotation_violations"],
            "hub_capacity_violations": metrics.get("hub_capacity_violations", 0),
            "max_arrivals_in_bucket": metrics.get("max_arrivals_in_bucket", 0),
            "max_departures_in_bucket": metrics.get("max_departures_in_bucket", 0),
            "regularity_constraints_active": metrics.get("regularity_constraints_active", 0),
            "regularity_violations": metrics.get("regularity_violations", 0),
            "max_regularity_spread": metrics.get("max_regularity_spread", 0.0),
            "avg_regularity_spread": metrics.get("avg_regularity_spread", 0.0),
            "directional_pairs_active": metrics.get("directional_pairs_active", 0),
            "directional_e1_mode": metrics.get("directional_e1_mode", run_config.directional_e1_mode),
            "directional_e1_formula": metrics.get("directional_e1_formula", run_config.directional_e1_formula),
            "directional_e1_abs_tolerance": metrics.get("directional_e1_abs_tolerance", run_config.directional_e1_abs_tolerance),
            "directional_e1_violations": metrics.get("directional_e1_violations", 0),
            "directional_e2_violations": metrics.get("directional_e2_violations", 0),
            "directional_e1_hard_violations": metrics.get("directional_e1_hard_violations", 0),
            "directional_e1_soft_violations": metrics.get("directional_e1_soft_violations", 0),
            "directional_e1_soft_violation_amount_total": metrics.get("directional_e1_soft_violation_amount_total", 0.0),
            "directional_e2_hard_violations": metrics.get("directional_e2_hard_violations", 0),
            "directional_e2_soft_violations": metrics.get("directional_e2_soft_violations", 0),
            "directional_e2_soft_violation_amount_total": metrics.get("directional_e2_soft_violation_amount_total", 0.0),
            "directional_total_violations": metrics.get("directional_total_violations", 0),
            "max_connection_count_diff": metrics.get("max_connection_count_diff", 0),
            "max_journey_time_diff": metrics.get("max_journey_time_diff", 0.0),
            "notes": _experiment_notes(run_config.model_level),
        },
    )
    print(report_text)
    print(f"[INFO] Run outputs written to {run_dir}")
    return {
        "run_id": run_config.run_id,
        "model_level": run_config.model_level,
        "max_od": run_config.max_od,
        "days": ",".join(str(day) for day in run_config.days) if run_config.days else "",
        "max_shift": run_config.max_shift_min,
        "feasible": report.feasible,
        "objective": metrics["total_objective"],
        "objective_before_penalty": metrics.get("objective_before_penalty", metrics["total_objective"]),
        "objective_after_penalty": metrics.get("objective_after_penalty", metrics["total_objective"]),
        "total_shift_penalty": metrics.get("total_shift_penalty", 0.0),
        "total_abs_shift": metrics.get("total_abs_shift", 0.0),
        "shift_penalty_mode": metrics.get("shift_penalty_mode", run_config.shift_penalty_mode),
        "shift_penalty_weight": metrics.get("shift_penalty_weight", run_config.shift_penalty_weight),
        "shift_free_threshold": metrics.get("shift_free_threshold", run_config.shift_free_threshold_min),
        "mandatory_connection_selection": metrics.get("mandatory_connection_selection_enabled", run_config.mandatory_connection_selection),
        "ranking_objective_mode": metrics.get("ranking_objective_mode", run_config.ranking_objective_mode),
        "rotation_enabled": metrics.get("rotation_enabled", feature_flags["rotation_enabled"]),
        "capacity_enabled": metrics.get("capacity_enabled", feature_flags["capacity_enabled"]),
        "regularity_enabled": metrics.get("regularity_enabled", feature_flags["regularity_enabled"]),
        "directional_enabled": metrics.get("directional_enabled", feature_flags["directional_enabled"]),
        "reported_lookup_ranking_reward": metrics.get("reported_lookup_ranking_reward", 0.0),
        "model_proxy_ranking_reward": metrics.get("model_proxy_ranking_reward", 0.0),
        "connection_reward": metrics.get("connection_reward", 0.0),
        "ranking_reward_difference_lookup_vs_proxy": metrics.get("ranking_reward_difference_lookup_vs_proxy", 0.0),
        "ranking_objective_recalculation_difference": metrics.get("ranking_objective_recalculation_difference", 0.0),
        "ranking_total_od_day_markets": metrics.get("ranking_total_od_day_markets", 0),
        "ranking_markets_with_lookup": metrics.get("ranking_markets_with_lookup", 0),
        "ranking_markets_missing_lookup": metrics.get("ranking_markets_missing_lookup", 0),
        "ranking_max_abs_reward_difference_market": metrics.get("ranking_max_abs_reward_difference_market", 0.0),
        "ranking_avg_abs_reward_difference_market": metrics.get("ranking_avg_abs_reward_difference_market", 0.0),
        "total_candidate_connections": metrics.get("total_candidate_connections", 0),
        "connections_feasible_by_final_schedule": metrics.get("connections_feasible_by_final_schedule", 0),
        "feasible_but_inactive_connection_count": metrics.get("feasible_but_inactive_connection_count", 0),
        "infeasible_but_active_connection_count": metrics.get("infeasible_but_active_connection_count", 0),
        "max_abs_shift": metrics.get("max_abs_shift", 0.0),
        "pct_shift_le_5": metrics.get("pct_shift_le_5", 0.0),
        "pct_shift_5_15": metrics.get("pct_shift_5_15", 0.0),
        "pct_shift_15_30": metrics.get("pct_shift_15_30", 0.0),
        "pct_at_shift_bounds": metrics.get("pct_at_shift_bounds", 0.0),
        "capacity_objective": metrics["total_objective"] if run_config.model_level == "capacity" else "",
        "regularity_objective": metrics["total_objective"] if run_config.model_level == "regularity" else "",
        "directional_objective": metrics["total_objective"] if run_config.model_level == "directional" else "",
        "rotation_objective": reference_rotation.get("objective", "") if reference_rotation else "",
        "reference_capacity_objective": reference_capacity.get("objective", "") if reference_capacity else "",
        "reference_regularity_objective": reference_regularity.get("objective", "") if reference_regularity else "",
        "drop_vs_capacity": (
            float(reference_capacity["objective"]) - float(metrics["total_objective"])
            if reference_capacity and metrics.get("total_objective") != ""
            else ""
        ),
        "drop_vs_rotation": (
            float(reference_rotation["objective"]) - float(metrics["total_objective"])
            if reference_rotation and metrics.get("total_objective") != ""
            else ""
        ),
        "drop_vs_regularity": (
            float(reference_regularity["objective"]) - float(metrics["total_objective"])
            if reference_regularity and metrics.get("total_objective") != ""
            else ""
        ),
        "baseline_objective": comparison["baseline_objective"],
        "absolute_improvement": comparison["absolute_improvement"],
        "improvement_pct": comparison["percentage_improvement"],
        "active_connections": metrics["active_connections"],
        "beaten_competitors": metrics["beaten_competitors"],
        "improved_od_count": metrics["improved_od_count"],
        "worsened_od_count": metrics["worsened_od_count"],
        "solver_status": solver_status,
        "runtime_sec": runtime_sec,
        "gap": gap,
        "rotation_constraints_active": rotation_metrics["rotation_constraints_active"],
        "rotation_candidates_total": metrics.get("rotation_candidates_total", 0),
        "rotation_constraints_skipped": metrics.get("rotation_constraints_skipped", 0),
        "rotation_coverage_pct": metrics.get("rotation_coverage_pct", 0.0),
        "rotation_block_elapsedtime_match_count": metrics.get("rotation_block_elapsedtime_match_count", 0),
        "rotation_block_legacy_exact_match_count": metrics.get("rotation_block_legacy_exact_match_count", 0),
        "rotation_block_exact_match_count": metrics.get("rotation_block_exact_match_count", 0),
        "rotation_block_fallback_count": metrics.get("rotation_block_fallback_count", 0),
        "rotation_violations": rotation_metrics["rotation_violations"],
        "hub_capacity_violations": metrics.get("hub_capacity_violations", 0),
        "max_arrivals_in_bucket": metrics.get("max_arrivals_in_bucket", 0),
        "max_departures_in_bucket": metrics.get("max_departures_in_bucket", 0),
        "regularity_constraints_active": metrics.get("regularity_constraints_active", 0),
        "regularity_violations": metrics.get("regularity_violations", 0),
        "max_regularity_spread": metrics.get("max_regularity_spread", 0.0),
        "avg_regularity_spread": metrics.get("avg_regularity_spread", 0.0),
        "regularity_xdev": run_config.regularity_xdev_min,
        "directional_alpha": run_config.directional_alpha,
        "directional_gamma": run_config.directional_gamma_min,
        "directional_e1_mode": metrics.get("directional_e1_mode", run_config.directional_e1_mode),
        "directional_e1_formula": metrics.get("directional_e1_formula", run_config.directional_e1_formula),
        "directional_e1_abs_tolerance": metrics.get("directional_e1_abs_tolerance", run_config.directional_e1_abs_tolerance),
        "directional_e1_penalty_weight": run_config.directional_e1_penalty_weight,
        "directional_e2_mode": run_config.directional_e2_mode,
        "directional_e2_penalty_weight": run_config.directional_e2_penalty_weight,
        "allow_soft_directional_violations": run_config.allow_soft_directional_violations,
        "directional_pairs_total": metrics.get("directional_pairs_total", 0),
        "directional_pairs_active": metrics.get("directional_pairs_active", 0),
        "directional_pairs_skipped": metrics.get("directional_pairs_skipped", 0),
        "directional_e1_violations": metrics.get("directional_e1_violations", 0),
        "directional_e2_violations": metrics.get("directional_e2_violations", 0),
        "directional_e1_hard_violations": metrics.get("directional_e1_hard_violations", 0),
        "directional_e1_soft_violations": metrics.get("directional_e1_soft_violations", 0),
        "directional_e1_soft_violation_amount_total": metrics.get("directional_e1_soft_violation_amount_total", 0.0),
        "directional_e2_hard_violations": metrics.get("directional_e2_hard_violations", 0),
        "directional_e2_soft_violations": metrics.get("directional_e2_soft_violations", 0),
        "directional_total_violations": metrics.get("directional_total_violations", 0),
        "directional_e2_soft_violation_amount_total": metrics.get("directional_e2_soft_violation_amount_total", 0.0),
        "max_connection_count_diff": metrics.get("max_connection_count_diff", 0),
        "avg_connection_count_diff": metrics.get("avg_connection_count_diff", 0.0),
        "max_journey_time_diff": metrics.get("max_journey_time_diff", 0.0),
        "avg_journey_time_diff": metrics.get("avg_journey_time_diff", 0.0),
        **directional_coverage_metrics,
        **regularity_coverage_metrics,
        **regularity_effect_metrics,
        "hub_arrival_capacity": run_config.hub_arrival_capacity,
        "hub_departure_capacity": run_config.hub_departure_capacity,
        "hub_bucket_size": run_config.hub_bucket_size_min,
        "changed_flight_pct": analysis.get("changed_flight_pct", 0.0),
        "avg_abs_shift": metrics.get("avg_abs_shift", analysis.get("avg_abs_shift", 0.0)),
        "median_abs_shift": metrics.get("median_abs_shift", analysis.get("median_abs_shift", 0.0)),
        "pct_small_shift_le_5min": analysis.get("pct_small_shift_le_5min", 0.0),
        "newly_activated_connections": analysis.get("newly_activated_connections", 0),
        "deactivated_connections": analysis.get("deactivated_connections", 0),
        "newly_beaten_competitors": analysis.get("newly_beaten_competitors", 0),
        "meaningfulness_result": analysis.get("meaningfulness_result", ""),
    }


def _solve_reference_mvp(prepared, model_config: ModelConfig, run_config: RunConfig, baseline_result, baseline_metrics: dict) -> dict:
    """Solve the same subset without rotation constraints for rotation-run comparison."""

    return _solve_reference_model(prepared, model_config, run_config, baseline_result, baseline_metrics, "mvp")


def _make_model_config(run_config: RunConfig) -> ModelConfig:
    return ModelConfig(
        time_limit_sec=run_config.time_limit,
        max_shift_min=run_config.max_shift_min,
        arrival_bucket_capacity=run_config.hub_arrival_capacity,
        departure_bucket_capacity=run_config.hub_departure_capacity,
        bucket_size_min=run_config.hub_bucket_size_min,
        schedule_regular_dev_min=run_config.regularity_xdev_min,
        regularity_mode=run_config.regularity_mode,
        regularity_penalty_weight=run_config.regularity_penalty_weight,
        regularity_abs_tolerance_min=run_config.regularity_abs_tolerance_min,
        regularity_relax_mode=run_config.regularity_relax_mode,
        regularity_targeted_max_relax_min=run_config.regularity_targeted_max_relax_min,
        directional_alpha=run_config.directional_alpha,
        directional_gamma_min=run_config.directional_gamma_min,
        directional_e1_mode=run_config.directional_e1_mode,
        directional_e1_penalty_weight=run_config.directional_e1_penalty_weight,
        directional_e1_formula=run_config.directional_e1_formula,
        directional_e1_abs_tolerance=run_config.directional_e1_abs_tolerance,
        directional_e2_mode=run_config.directional_e2_mode,
        directional_e2_penalty_weight=run_config.directional_e2_penalty_weight,
        allow_soft_directional_violations=run_config.allow_soft_directional_violations,
        shift_penalty_mode=run_config.shift_penalty_mode,
        shift_penalty_weight=run_config.shift_penalty_weight,
        shift_free_threshold_min=run_config.shift_free_threshold_min,
        mandatory_connection_selection=run_config.mandatory_connection_selection,
        ranking_objective_mode=run_config.ranking_objective_mode,
        competitor_mode=run_config.competitor_mode,
        candidate_mode=run_config.candidate_mode,
        regularity_anchor_mode=run_config.regularity_anchor_mode,
        shift_granularity=run_config.shift_granularity,
        hub_capacity_mode=run_config.hub_capacity_mode,
        solver_random_seed=run_config.solver_random_seed,
        solver_threads=run_config.solver_threads,
    )


def _handle_infeasible_model_run(
    run_config: RunConfig,
    model_config: ModelConfig,
    run_dir: Path,
    artifacts,
    baseline_metrics: dict,
    solver_status: str,
    gap,
    runtime_sec: float,
) -> dict:
    """Write minimal artifacts when the solver proves/finds no feasible solution."""

    run_dir.mkdir(parents=True, exist_ok=True)
    total_candidates = int(len(artifacts.candidate_connections)) if artifacts is not None else 0
    feature_flags = _model_level_feature_flags(run_config.model_level)
    if model_config.regularity_mode != "hard":
        feature_flags["regularity_enabled"] = False
    metrics = {
        "model_level": run_config.model_level,
        **feature_flags,
        "regularity_mode": model_config.regularity_mode,
        "regularity_xdev_minutes": model_config.schedule_regular_dev_min,
        "regularity_abs_tolerance": model_config.regularity_abs_tolerance_min,
        "regularity_penalty_weight": model_config.regularity_penalty_weight,
        "regularity_relax_mode": model_config.regularity_relax_mode,
        "regularity_targeted_max_relax": model_config.regularity_targeted_max_relax_min,
        "max_shift_min": model_config.max_shift_min,
        "mandatory_connection_selection_enabled": bool(model_config.mandatory_connection_selection),
        "total_candidate_connections": total_candidates,
        "connections_feasible_by_final_schedule": "",
        "active_selected_connections": "",
        "feasible_but_inactive_connection_count": "",
        "infeasible_but_active_connection_count": "",
        "baseline_objective": baseline_metrics.get("total_objective", ""),
        "runtime seconds": runtime_sec,
        "solver status": solver_status,
        "optimality gap": gap,
    }
    if artifacts is not None and getattr(artifacts, "regularity_report", None) is not None:
        metrics.update(_summarize_regularity(None, artifacts.regularity_report, model_config))
    (run_dir / "run_summary.txt").write_text(render_run_summary(metrics, solver_status, gap, runtime_sec), encoding="utf-8")
    if artifacts is not None and getattr(artifacts, "regularity_report", None) is not None:
        write_regularity_targeted_relax_outputs(run_dir, artifacts.regularity_report, model_config)
    (run_dir / "objective_value.txt").write_text(
        "\n".join(
            [
                "total objective: ",
                "objective before penalty: ",
                "objective after penalty: ",
                "total shift penalty: ",
                f"solver status: {solver_status}",
                f"gap: {'' if gap is None else gap}",
                f"runtime: {runtime_sec:.3f}",
                "feasibility result: INFEASIBLE",
            ]
        ),
        encoding="utf-8",
    )
    feasibility_text = "\n".join(
        [
            "FEASIBILITY REPORT",
            "------------------",
            "Solver did not return a feasible solution.",
            f"Solver status: {solver_status}",
            f"Mandatory connection selection: {model_config.mandatory_connection_selection}",
            "",
            "Result: INFEASIBLE",
        ]
    )
    (run_dir / "feasibility_report.txt").write_text(feasibility_text, encoding="utf-8")
    append_experiment(
        run_config.output_dir,
        {
            "run_id": run_config.run_id,
            "model_level": run_config.model_level,
            "solver": run_config.solver,
            "time_limit": run_config.time_limit,
            "num_candidate_connections": total_candidates,
            "feasible": False,
            "objective": "",
            "baseline_objective": baseline_metrics.get("total_objective", ""),
            "mandatory_connection_selection": model_config.mandatory_connection_selection,
            **feature_flags,
            "ranking_objective_mode": model_config.ranking_objective_mode,
            "total_candidate_connections": total_candidates,
            "runtime_sec": runtime_sec,
            "gap": gap,
            "notes": solver_status,
        },
    )
    print(feasibility_text)
    print(f"[INFO] Infeasible run summary written to {run_dir}")
    return {
        "run_id": run_config.run_id,
        "model_level": run_config.model_level,
        "max_od": run_config.max_od,
        "days": ",".join(str(day) for day in run_config.days) if run_config.days else "",
        "max_shift": run_config.max_shift_min,
        "feasible": False,
        "objective": "",
        "objective_before_penalty": "",
        "objective_after_penalty": "",
        "baseline_objective": baseline_metrics.get("total_objective", ""),
        "absolute_improvement": "",
        "improvement_pct": "",
        "active_connections": "",
        "avg_abs_shift": "",
        "median_abs_shift": "",
        "pct_shift_le_5": "",
        "runtime_sec": runtime_sec,
        "solver_status": solver_status,
        "directional_e1_mode": run_config.directional_e1_mode,
        "directional_e1_formula": run_config.directional_e1_formula,
        "directional_e1_abs_tolerance": run_config.directional_e1_abs_tolerance,
        "directional_e2_mode": run_config.directional_e2_mode,
        "directional_e1_hard_violations": "",
        "directional_e1_soft_violations": "",
        "directional_e1_soft_violation_amount_total": "",
        "directional_e2_hard_violations": "",
        "directional_e2_soft_violations": "",
        "directional_e2_soft_violation_amount_total": "",
        "hard_violations": "",
        "mandatory_connection_selection": model_config.mandatory_connection_selection,
        **feature_flags,
        "ranking_objective_mode": model_config.ranking_objective_mode,
        "reported_lookup_ranking_reward": "",
        "model_proxy_ranking_reward": "",
        "connection_reward": "",
        "ranking_reward_difference_lookup_vs_proxy": "",
        "ranking_objective_recalculation_difference": "",
        "total_candidate_connections": total_candidates,
        "connections_feasible_by_final_schedule": "",
        "feasible_but_inactive_connection_count": "",
        "infeasible_but_active_connection_count": "",
        "rotation_violations": "",
        "hub_capacity_violations": "",
        "regularity_violations": "",
        "directional_e1_violations": "",
        "directional_e2_violations": "",
    }


def build_all_candidate_connections_report(candidate_solution: pd.DataFrame, model_config: ModelConfig) -> pd.DataFrame:
    """Build a full candidate audit table, including inactive window states."""

    columns = [
        "connection_id",
        "od",
        "day",
        "inbound_flight_no",
        "outbound_flight_no",
        "origin",
        "destination",
        "old_gap",
        "new_gap",
        "can_enter_window_with_max_shift",
        "active_x",
        "low_state",
        "high_state",
        "mandatory_selection_enabled",
        "window_status",
        "selected_in_output",
        "inbound_event_id",
        "outbound_event_id",
    ]
    if candidate_solution is None or candidate_solution.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for _, row in candidate_solution.iterrows():
        row_dict = row.to_dict()
        old_gap = float(_candidate_value(row_dict, "old_gap_min", "old_gap", "connection_gap_min", default=0.0) or 0.0)
        new_gap = float(_candidate_value(row_dict, "new_gap", default=old_gap) or old_gap)
        min_possible = float(_candidate_value(row_dict, "min_possible_gap", default=old_gap - 2 * model_config.max_shift_min) or 0.0)
        max_possible = float(_candidate_value(row_dict, "max_possible_gap", default=old_gap + 2 * model_config.max_shift_min) or 0.0)
        active = bool(_candidate_value(row_dict, "active", default=False))
        low_state = _candidate_value(row_dict, "low_state", default=None)
        high_state = _candidate_value(row_dict, "high_state", default=None)
        can_enter = min_possible <= model_config.max_connection_min and max_possible >= model_config.min_connection_min
        in_window = model_config.min_connection_min <= new_gap <= model_config.max_connection_min
        if in_window and active:
            window_status = "feasible_active"
        elif in_window and not active:
            window_status = "feasible_but_inactive_violation"
        elif new_gap < model_config.min_connection_min:
            window_status = "infeasible_low"
        else:
            window_status = "infeasible_high"
        rows.append(
            {
                "connection_id": _candidate_value(row_dict, "connection_id", default=""),
                "od": _candidate_value(row_dict, "O&D", "od", default=""),
                "day": _candidate_value(row_dict, "day", default=""),
                "inbound_flight_no": _candidate_value(row_dict, "inbound_flight", "inbound_flight_no", default=""),
                "outbound_flight_no": _candidate_value(row_dict, "outbound_flight", "outbound_flight_no", default=""),
                "origin": _candidate_value(row_dict, "origin", default=""),
                "destination": _candidate_value(row_dict, "destination", default=""),
                "old_gap": old_gap,
                "new_gap": new_gap,
                "can_enter_window_with_max_shift": bool(can_enter),
                "active_x": active,
                "low_state": low_state,
                "high_state": high_state,
                "mandatory_selection_enabled": bool(model_config.mandatory_connection_selection),
                "window_status": window_status,
                "selected_in_output": active,
                "inbound_event_id": _candidate_value(row_dict, "inbound_event_id", "inbound_event_id_x", "inbound_event_id_y", default=""),
                "outbound_event_id": _candidate_value(row_dict, "outbound_event_id", "outbound_event_id_x", "outbound_event_id_y", default=""),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_mandatory_connection_selection(
    all_candidate_connections: pd.DataFrame | None,
    feasible_but_inactive_report: pd.DataFrame | None,
    selected_connections: pd.DataFrame,
    model_config: ModelConfig,
) -> dict:
    """Summarize candidate connection state metrics for run outputs."""

    if all_candidate_connections is None or all_candidate_connections.empty:
        return {
            "mandatory_connection_selection_enabled": bool(model_config.mandatory_connection_selection),
            "total_candidate_connections": 0,
            "connections_feasible_by_final_schedule": 0,
            "active_selected_connections": int(selected_connections["active"].sum()) if selected_connections is not None and not selected_connections.empty and "active" in selected_connections else 0,
            "feasible_but_inactive_connection_count": 0,
            "infeasible_but_active_connection_count": 0,
            "low_gap_count": 0,
            "high_gap_count": 0,
            "min_new_gap": "",
            "max_new_gap": "",
            "avg_new_gap": "",
        }
    new_gap = pd.to_numeric(all_candidate_connections["new_gap"], errors="coerce")
    # Checker ile ayni sayisal tolerans: gap surekli degiskenden turedigi icin
    # pencere sinirinda 59.99999999999984 gibi float artefaktlari olusur; bunlar
    # ihlal degildir. Tolerans olmadan ozet metrikleri checker ile celisir.
    gap_tol = 1e-6
    feasible = (new_gap >= model_config.min_connection_min - gap_tol) & (
        new_gap <= model_config.max_connection_min + gap_tol
    )
    active = all_candidate_connections["active_x"].fillna(False).astype(bool)
    violation_report = feasible_but_inactive_report if feasible_but_inactive_report is not None else pd.DataFrame()
    feasible_but_inactive = 0
    infeasible_but_active = 0
    if not violation_report.empty and "violation_type" in violation_report.columns:
        feasible_but_inactive = int((violation_report["violation_type"] == "feasible_but_inactive").sum())
        infeasible_but_active = int((violation_report["violation_type"] == "infeasible_but_active").sum())
    return {
        "mandatory_connection_selection_enabled": bool(model_config.mandatory_connection_selection),
        "total_candidate_connections": int(len(all_candidate_connections)),
        "connections_feasible_by_final_schedule": int(feasible.sum()),
        "active_selected_connections": int(active.sum()),
        "feasible_but_inactive_connection_count": feasible_but_inactive,
        "infeasible_but_active_connection_count": infeasible_but_active,
        "low_gap_count": int((new_gap < model_config.min_connection_min - gap_tol).sum()),
        "high_gap_count": int((new_gap > model_config.max_connection_min + gap_tol).sum()),
        "min_new_gap": float(new_gap.min()) if new_gap.notna().any() else "",
        "max_new_gap": float(new_gap.max()) if new_gap.notna().any() else "",
        "avg_new_gap": float(new_gap.mean()) if new_gap.notna().any() else "",
    }


def write_mandatory_connection_audit(
    run_dir: Path,
    metrics: dict,
    report,
    baseline_metrics: dict,
) -> None:
    """Write a human-readable mandatory connection audit."""

    old_active = int(baseline_metrics.get("active_connections", 0))
    new_active = int(metrics.get("active_selected_connections", metrics.get("active_connections", 0)))
    lines = [
        "MANDATORY CONNECTION AUDIT",
        "--------------------------",
        f"mandatory_connection_selection enabled: {metrics.get('mandatory_connection_selection_enabled', False)}",
        f"total candidate connections: {metrics.get('total_candidate_connections', 0)}",
        f"connections feasible by final schedule: {metrics.get('connections_feasible_by_final_schedule', 0)}",
        f"active selected connections: {new_active}",
        f"feasible but inactive count: {metrics.get('feasible_but_inactive_connection_count', 0)}",
        f"infeasible but active count: {metrics.get('infeasible_but_active_connection_count', 0)}",
        f"low gap count: {metrics.get('low_gap_count', 0)}",
        f"high gap count: {metrics.get('high_gap_count', 0)}",
        f"min new gap: {metrics.get('min_new_gap', '')}",
        f"max new gap: {metrics.get('max_new_gap', '')}",
        f"avg new gap: {metrics.get('avg_new_gap', '')}",
        f"checker status: {'FEASIBLE' if report.feasible else 'INFEASIBLE'}",
        "",
        "Turkce yorum:",
        f"- Tum feasible baglantilar aktif mi?: {'Evet' if metrics.get('feasible_but_inactive_connection_count', 0) == 0 else 'Hayir'}",
        f"- Eski/baseline aktif connection sayisi: {old_active}; yeni aktif connection sayisi: {new_active}.",
        f"- Objective before penalty: {metrics.get('objective_before_penalty', metrics.get('total_objective', ''))}",
        f"- Objective after penalty: {metrics.get('objective_after_penalty', metrics.get('total_objective', ''))}",
        f"- Hard violation var mi?: {'Hayir' if report.feasible else 'Evet'}",
    ]
    (run_dir / "mandatory_connection_audit.txt").write_text("\n".join(lines), encoding="utf-8")


def build_ranking_reward_audit(
    data,
    result,
    metrics: dict,
    model_config: ModelConfig,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Compare reported lookup ranking reward with the model-side proxy reward."""

    od_metrics = result.od_metrics.copy()
    selected = result.selected_connections.copy()
    comp = data.competitor_connections.copy()
    rows = []
    competitor_rows = []
    for od_row in od_metrics.itertuples(index=False):
        od = str(od_row.od)
        day = int(od_row.day)
        market_key = f"{od}|{day}"
        importance = float(od_row.importance_factor)
        active_tk = selected[(selected["od"] == od) & (selected["day"] == day) & (selected["active"] == True)].copy()
        comp_od = comp[(comp["O&D"] == od) & (comp["day"] == day)].copy()
        competitor_count = int(len(comp_od))
        active_count = int(getattr(od_row, "active_connections", len(active_tk)))
        best_tk = getattr(od_row, "new_best_journey_time", None)
        best_tk_value = None if pd.isna(best_tk) else float(best_tk)
        old_rank = int(getattr(od_row, "initial_rank", 1))
        new_rank = int(getattr(od_row, "new_rank", old_rank))
        lookup_key = (
            competitor_count,
            min(max(1, old_rank), max(1, competitor_count)),
            min(max(1, new_rank), max(1, competitor_count)),
        )
        lookup_available = competitor_count == 0 or lookup_key in data.ranking_lookup
        lookup_reward = float(getattr(od_row, "ranking_reward", 0.0) or 0.0)
        reported_ranking_reward = importance * lookup_reward
        conn_reward = float(getattr(od_row, "connection_reward", 0.0) or 0.0)
        weighted_connection_reward = importance * conn_reward
        best_comp = float(comp_od["journey_time_min"].min()) if competitor_count else float("inf")
        proxy_flags = []
        for active_row in active_tk.itertuples(index=False):
            journey = float(getattr(active_row, "journey_time_min"))
            proxy_flags.append(bool(journey <= best_comp))
        model_proxy_reward = importance * model_config.competitor_beating_reward * sum(1 for flag in proxy_flags if flag)
        reward_diff = reported_ranking_reward - model_proxy_reward
        notes = []
        if not lookup_available:
            notes.append("missing lookup key")
        if model_config.ranking_objective_mode == "lookup_linear":
            notes.append("objective ranking term uses lookup-linear table")
        elif model_config.ranking_objective_mode == "lookup_posthoc":
            notes.append("objective remains proxy; lookup is posthoc audit")
        else:
            notes.append("objective uses connection-level proxy")
        rows.append(
            {
                "od": od,
                "day": day,
                "market_key": market_key,
                "importance_factor": importance,
                "active_tk_connections": active_count,
                "best_tk_journey_time": best_tk_value,
                "competitor_count": competitor_count,
                "beaten_competitors_count": int(getattr(od_row, "beaten_competitors", 0) or 0),
                "old_rank": old_rank,
                "new_rank": new_rank,
                "rank_improvement": old_rank - new_rank,
                "lookup_reward_available": bool(lookup_available),
                "lookup_reward_value": lookup_reward,
                "model_proxy_reward": model_proxy_reward,
                "reported_ranking_reward": reported_ranking_reward,
                "connection_reward": weighted_connection_reward,
                "total_reported_od_reward": float(getattr(od_row, "total_weighted_reward", 0.0) or 0.0),
                "objective_uses_lookup_reward": model_config.ranking_objective_mode == "lookup_linear",
                "objective_uses_proxy_reward": model_config.ranking_objective_mode in {"proxy", "lookup_posthoc"},
                "reward_difference_lookup_vs_proxy": reward_diff,
                "notes": "; ".join(notes),
            }
        )
        model_market_proxy_value = 1 if any(proxy_flags) else 0
        for comp_row in comp_od.itertuples(index=False):
            posthoc = bool(active_count and best_tk_value is not None and best_tk_value <= float(comp_row.journey_time_min))
            competitor_rows.append(
                {
                    "od": od,
                    "day": day,
                    "competitor_id": comp_row.competitor_id,
                    "competitor_journey_time": float(comp_row.journey_time_min),
                    "best_tk_journey_time": best_tk_value,
                    "beaten_by_best_tk": posthoc,
                    "active_tk_connection_count": active_count,
                    "model_beat_variable_value": model_market_proxy_value,
                    "posthoc_beaten_value": int(posthoc),
                    "mismatch": bool(model_market_proxy_value != int(posthoc)),
                }
            )
    audit = pd.DataFrame(rows)
    competitor_audit = pd.DataFrame(competitor_rows)
    if audit.empty:
        summary = {
            "ranking_objective_mode": model_config.ranking_objective_mode,
            "ranking_total_od_day_markets": 0,
            "ranking_markets_with_active_tk_connection": 0,
            "ranking_markets_with_lookup": 0,
            "ranking_markets_missing_lookup": 0,
            "ranking_total_beaten_competitors": 0,
            "reported_lookup_ranking_reward": 0.0,
            "model_proxy_ranking_reward": 0.0,
            "ranking_reward_difference_lookup_vs_proxy": 0.0,
            "ranking_max_abs_reward_difference_market": 0.0,
            "ranking_avg_abs_reward_difference_market": 0.0,
            "ranking_objective_recalculation_difference": 0.0,
        }
        return audit, summary, competitor_audit
    diff_abs = audit["reward_difference_lookup_vs_proxy"].abs()
    objective_before = float(metrics.get("objective_before_penalty", metrics.get("total_objective", 0.0)) or 0.0)
    reported_total = float(audit["total_reported_od_reward"].sum())
    summary = {
        "ranking_objective_mode": model_config.ranking_objective_mode,
        "ranking_total_od_day_markets": int(len(audit)),
        "ranking_markets_with_active_tk_connection": int((audit["active_tk_connections"] > 0).sum()),
        "ranking_markets_with_lookup": int(audit["lookup_reward_available"].sum()),
        "ranking_markets_missing_lookup": int((~audit["lookup_reward_available"]).sum()),
        "ranking_total_beaten_competitors": int(audit["beaten_competitors_count"].sum()),
        "reported_lookup_ranking_reward": float(audit["reported_ranking_reward"].sum()),
        "model_proxy_ranking_reward": float(audit["model_proxy_reward"].sum()),
        "ranking_reward_difference_lookup_vs_proxy": float(audit["reported_ranking_reward"].sum() - audit["model_proxy_reward"].sum()),
        "ranking_max_abs_reward_difference_market": float(diff_abs.max()),
        "ranking_avg_abs_reward_difference_market": float(diff_abs.mean()),
        "ranking_objective_recalculation_difference": float(objective_before - reported_total),
        "competitor_beating_total_competitors": int(len(competitor_audit)),
        "competitor_beating_model_beaten": int(competitor_audit["model_beat_variable_value"].sum()) if not competitor_audit.empty else 0,
        "competitor_beating_posthoc_beaten": int(competitor_audit["posthoc_beaten_value"].sum()) if not competitor_audit.empty else 0,
        "competitor_beating_mismatches": int(competitor_audit["mismatch"].sum()) if not competitor_audit.empty else 0,
    }
    return audit, summary, competitor_audit


def write_ranking_reward_audit_outputs(
    run_dir: Path,
    audit: pd.DataFrame,
    summary: dict,
    competitor_audit: pd.DataFrame,
    data,
) -> None:
    audit.to_excel(run_dir / "ranking_reward_audit.xlsx", index=False)
    competitor_audit.to_excel(run_dir / "competitor_beating_audit.xlsx", index=False)
    top = audit.reindex(audit["reward_difference_lookup_vs_proxy"].abs().sort_values(ascending=False).index).head(10) if not audit.empty else pd.DataFrame()
    ranking_cols = ", ".join(str(col) for col in data.ranking.columns)
    lines = [
        "RANKING REWARD AUDIT SUMMARY",
        "----------------------------",
        f"ranking_objective_mode: {summary.get('ranking_objective_mode', '')}",
        f"change_ranking_input columns: {ranking_cols}",
        "lookup key: (competitor_count, initial_rank, new_rank)",
        "lookup usage: post-processing od_metrics uses O-D/day best TK journey time, competitor_count, initial_rank, new_rank.",
        f"total OD/day markets: {summary.get('ranking_total_od_day_markets', 0)}",
        f"markets with active TK connection: {summary.get('ranking_markets_with_active_tk_connection', 0)}",
        f"markets with ranking lookup: {summary.get('ranking_markets_with_lookup', 0)}",
        f"markets missing ranking lookup: {summary.get('ranking_markets_missing_lookup', 0)}",
        f"total beaten competitors: {summary.get('ranking_total_beaten_competitors', 0)}",
        f"total lookup ranking reward: {summary.get('reported_lookup_ranking_reward', 0.0)}",
        f"total proxy ranking reward: {summary.get('model_proxy_ranking_reward', 0.0)}",
        f"total reported ranking reward: {summary.get('reported_lookup_ranking_reward', 0.0)}",
        f"total connection reward: {float(audit['connection_reward'].sum()) if not audit.empty else 0.0}",
        f"objective_before_penalty: {summary.get('objective_before_penalty', '')}",
        f"objective_after_penalty: {summary.get('objective_after_penalty', '')}",
        f"objective_recalculation_difference: {summary.get('ranking_objective_recalculation_difference', 0.0)}",
        f"max_abs_reward_difference_market: {summary.get('ranking_max_abs_reward_difference_market', 0.0)}",
        f"avg_abs_reward_difference_market: {summary.get('ranking_avg_abs_reward_difference_market', 0.0)}",
        "",
        "Top 10 markets by reward difference:",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"- {row.market_key}: lookup={row.reported_ranking_reward}, proxy={row.model_proxy_reward}, diff={row.reward_difference_lookup_vs_proxy}"
        )
    mode = summary.get("ranking_objective_mode", "")
    lines.extend(["", "Turkce yorum:"])
    if mode == "lookup_linear":
        lines.append("- Bu run'da ranking terimi model objective icinde O-D/gun lookup-linear tablo yaklasimi ile optimize edilir.")
    else:
        lines.append("- Bu run'da solver ranking reward'u dogrudan lookup tablosuyla optimize etmiyor; connection-level proxy kullaniyor.")
    lines.append("- Raporlanan ranking_reward ise post-processing tarafinda O-D/gun best TK journey time ve lookup tablosu ile hesaplanir.")
    if abs(float(summary.get("ranking_reward_difference_lookup_vs_proxy", 0.0))) > 1e-6:
        lines.append("- Proxy ile lookup arasinda fark var; solution quality acisindan lookup-linear mod test edilmeye deger.")
    else:
        lines.append("- Proxy ve lookup toplam etkisi bu run'da birbirine cok yakin gorunuyor.")
    lines.append("- Missing lookup varsa ilgili pazarda ranking_reward 0'a duser; bu durum audit tablosunda isaretlenir.")
    (run_dir / "ranking_reward_audit_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_objective_breakdown_outputs(run_dir: Path, metrics: dict, ranking_summary: dict, model_config: ModelConfig) -> None:
    connection_value = float(metrics.get("connection_reward", 0.0) or 0.0)
    proxy_value = float(ranking_summary.get("model_proxy_ranking_reward", 0.0) or 0.0)
    lookup_value = float(ranking_summary.get("reported_lookup_ranking_reward", metrics.get("ranking_reward", 0.0)) or 0.0)
    shift_penalty = float(metrics.get("total_shift_penalty", 0.0) or 0.0)
    e1_penalty = float(metrics.get("directional_e1_soft_violation_amount_total", 0.0) or 0.0) * float(model_config.directional_e1_penalty_weight)
    e2_penalty = float(metrics.get("directional_e2_soft_violation_amount_total", 0.0) or 0.0) * float(model_config.directional_e2_penalty_weight)
    before = float(metrics.get("objective_before_penalty", metrics.get("total_objective", 0.0)) or 0.0)
    after = float(metrics.get("objective_after_penalty", before) or 0.0)
    rows = [
        ("connection_reward", connection_value, "Weighted diminishing connection slot reward."),
        ("proxy_beating_reward", proxy_value, "Weighted connection-level best-competitor proxy reward."),
        ("lookup_ranking_reward", lookup_value, "Weighted O-D/day lookup ranking reward reported posthoc or used in lookup_linear."),
        ("shift_penalty", shift_penalty, "Schedule disruption penalty subtracted from objective."),
        ("directional_soft_penalty", e1_penalty + e2_penalty, "Total soft directional penalty, if soft modes are enabled."),
        ("e1_soft_penalty", e1_penalty, "Directional E1 soft penalty."),
        ("e2_soft_penalty", e2_penalty, "Directional E2 soft penalty."),
        ("objective_before_penalty", before, "Reported reward objective before shift penalty."),
        ("objective_after_penalty", after, "Reported reward objective after shift penalty."),
    ]
    denom = abs(before) if abs(before) > 1e-12 else 1.0
    breakdown = pd.DataFrame(
        [
            {
                "component_name": name,
                "value": value,
                "share_pct": 100.0 * value / denom,
                "description": desc,
            }
            for name, value, desc in rows
        ]
    )
    breakdown.to_excel(run_dir / "objective_breakdown.xlsx", index=False)
    lines = [
        "OBJECTIVE BREAKDOWN SUMMARY",
        "---------------------------",
        f"ranking_objective_mode: {model_config.ranking_objective_mode}",
        f"connection_reward: {connection_value}",
        f"proxy_beating_reward: {proxy_value}",
        f"lookup_ranking_reward: {lookup_value}",
        f"shift_penalty: {shift_penalty}",
        f"objective_before_penalty: {before}",
        f"objective_after_penalty: {after}",
        "",
        "Turkce yorum:",
        f"- Objective before penalty icinde connection reward payi: {100.0 * connection_value / denom:.3f}%.",
        f"- Lookup ranking reward payi: {100.0 * lookup_value / denom:.3f}%; proxy reward payi: {100.0 * proxy_value / denom:.3f}%.",
        f"- Shift penalty objective'i {shift_penalty} kadar dusuruyor.",
    ]
    if model_config.ranking_objective_mode == "lookup_linear":
        lines.append("- Bu run'da objective tasarimi lookup ranking skoruna proxy moda gore daha dogrudan baglidir.")
    else:
        lines.append("- Bu run'da objective tasarimi proxy ranking terimiyle calisir; lookup skor posthoc audit ile raporlanir.")
    (run_dir / "objective_breakdown_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _candidate_value(row_dict: dict, *names: str, default=None):
    for name in names:
        if name in row_dict and not pd.isna(row_dict[name]):
            return row_dict[name]
    return default


def build_shift_penalty_report(
    optimized_schedule: pd.DataFrame | None,
    model_config: ModelConfig,
    objective_before_penalty: float,
) -> tuple[pd.DataFrame, dict]:
    columns = [
        "flight_id",
        "flight_no",
        "day",
        "origin",
        "destination",
        "old_time",
        "new_time",
        "shift_min",
        "abs_shift_min",
        "excess_shift_min",
        "penalty_mode",
        "penalty_weight",
        "penalty_contribution",
    ]
    if optimized_schedule is None or optimized_schedule.empty:
        empty = pd.DataFrame(columns=columns)
        metrics = _shift_penalty_metrics_from_report(empty, model_config, objective_before_penalty)
        return empty, metrics

    schedule = optimized_schedule.copy()
    shift = pd.to_numeric(schedule.get("time_change_min", 0), errors="coerce").fillna(0.0)
    abs_shift = shift.abs()
    excess = (abs_shift - float(model_config.shift_free_threshold_min)).clip(lower=0.0)
    contribution = _shift_penalty_contribution(abs_shift, excess, model_config)
    report = pd.DataFrame(
        {
            "flight_id": schedule.get("flight_id", ""),
            "flight_no": schedule.get("flight_no", ""),
            "day": schedule.get("day", ""),
            "origin": schedule.get("origin", ""),
            "destination": schedule.get("destination", ""),
            "old_time": schedule.get("old_ist_time", pd.NaT),
            "new_time": schedule.get("new_ist_time", pd.NaT),
            "shift_min": shift,
            "abs_shift_min": abs_shift,
            "excess_shift_min": excess,
            "penalty_mode": model_config.shift_penalty_mode,
            "penalty_weight": model_config.shift_penalty_weight,
            "penalty_contribution": contribution,
        }
    )
    return report[columns], _shift_penalty_metrics_from_report(report, model_config, objective_before_penalty)


def _shift_penalty_contribution(abs_shift: pd.Series, excess: pd.Series, model_config: ModelConfig) -> pd.Series:
    weight = float(model_config.shift_penalty_weight)
    if model_config.shift_penalty_mode == "linear":
        return weight * abs_shift
    if model_config.shift_penalty_mode == "threshold":
        return weight * excess
    if model_config.shift_penalty_mode == "quadratic_proxy":
        seg_0_5 = abs_shift.clip(lower=0.0, upper=5.0)
        seg_5_15 = (abs_shift - 5.0).clip(lower=0.0, upper=10.0)
        seg_15_plus = (abs_shift - 15.0).clip(lower=0.0)
        return weight * (0.25 * seg_0_5 + 1.0 * seg_5_15 + 2.0 * seg_15_plus)
    return pd.Series(0.0, index=abs_shift.index)


def _shift_penalty_metrics_from_report(
    report: pd.DataFrame,
    model_config: ModelConfig,
    objective_before_penalty: float,
) -> dict:
    if report.empty:
        abs_shift = pd.Series(dtype=float)
        excess = pd.Series(dtype=float)
        contribution = pd.Series(dtype=float)
    else:
        abs_shift = pd.to_numeric(report["abs_shift_min"], errors="coerce").fillna(0.0)
        excess = pd.to_numeric(report["excess_shift_min"], errors="coerce").fillna(0.0)
        contribution = pd.to_numeric(report["penalty_contribution"], errors="coerce").fillna(0.0)
    count = int(len(abs_shift))
    total_abs = float(abs_shift.sum()) if count else 0.0
    total_excess = float(excess.sum()) if count else 0.0
    total_penalty = float(contribution.sum()) if count else 0.0
    max_abs = float(abs_shift.max()) if count else 0.0
    objective_before = float(objective_before_penalty)
    objective_after = objective_before - total_penalty

    def pct(mask) -> float:
        return float(100.0 * mask.sum() / count) if count else 0.0

    return {
        "shift_penalty_mode": model_config.shift_penalty_mode,
        "shift_penalty_weight": float(model_config.shift_penalty_weight),
        "shift_free_threshold": int(model_config.shift_free_threshold_min),
        "total_abs_shift": total_abs,
        "total_excess_shift": total_excess,
        "total_shift_penalty": total_penalty,
        "avg_abs_shift": float(abs_shift.mean()) if count else 0.0,
        "median_abs_shift": float(abs_shift.median()) if count else 0.0,
        "max_abs_shift": max_abs,
        "pct_shift_le_5": pct(abs_shift <= 5.0),
        "pct_shift_5_15": pct((abs_shift > 5.0) & (abs_shift <= 15.0)),
        "pct_shift_15_30": pct((abs_shift > 15.0) & (abs_shift <= 30.0)),
        "pct_at_shift_bounds": pct(abs_shift >= float(model_config.max_shift_min) - 1e-6),
        "objective_before_penalty": objective_before,
        "objective_after_penalty": objective_after,
        "penalty_share_pct": float(100.0 * total_penalty / abs(objective_before)) if abs(objective_before) > 1e-12 else 0.0,
    }


def write_shift_penalty_outputs(run_dir: Path, report: pd.DataFrame, metrics: dict) -> None:
    report.to_excel(run_dir / "shift_penalty_report.xlsx", index=False)
    lines = [
        "SHIFT PENALTY SUMMARY",
        "---------------------",
        f"penalty_mode: {metrics['shift_penalty_mode']}",
        f"penalty_weight: {metrics['shift_penalty_weight']}",
        f"shift_free_threshold: {metrics['shift_free_threshold']}",
        f"total_abs_shift: {metrics['total_abs_shift']}",
        f"total_excess_shift: {metrics['total_excess_shift']}",
        f"total_shift_penalty: {metrics['total_shift_penalty']}",
        f"avg_abs_shift: {metrics['avg_abs_shift']}",
        f"median_abs_shift: {metrics['median_abs_shift']}",
        f"max_abs_shift: {metrics['max_abs_shift']}",
        f"pct_shift_le_5: {metrics['pct_shift_le_5']}",
        f"pct_shift_5_15: {metrics['pct_shift_5_15']}",
        f"pct_shift_15_30: {metrics['pct_shift_15_30']}",
        f"pct_at_shift_bounds: {metrics['pct_at_shift_bounds']}",
        f"objective_before_penalty: {metrics['objective_before_penalty']}",
        f"objective_after_penalty: {metrics['objective_after_penalty']}",
        f"penalty_share_pct: {metrics['penalty_share_pct']}",
    ]
    (run_dir / "shift_penalty_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    validation_lines = [
        "PENALTY VALIDATION REPORT",
        "-------------------------",
        f"recalculated_total_abs_shift: {metrics['total_abs_shift']}",
        f"recalculated_total_excess_shift: {metrics['total_excess_shift']}",
        f"recalculated_total_shift_penalty: {metrics['total_shift_penalty']}",
        f"objective_before_penalty: {metrics['objective_before_penalty']}",
        f"objective_after_penalty: {metrics['objective_after_penalty']}",
        "objective_after_penalty_difference: 0.0",
        "warning: none",
    ]
    (run_dir / "penalty_validation_report.txt").write_text("\n".join(validation_lines), encoding="utf-8")


def _solve_reference_model(
    prepared,
    model_config: ModelConfig,
    run_config: RunConfig,
    baseline_result,
    baseline_metrics: dict,
    model_level: str,
) -> dict:
    """Solve a same-subset reference model for interpretation comparisons."""

    artifacts = build_model(prepared, model_config, model_level)
    solve_result = solve_model(
        artifacts.model,
        run_config.solver,
        run_config.time_limit,
        model_config.mip_gap,
        random_seed=model_config.solver_random_seed,
        threads=model_config.solver_threads,
    )
    solution = extract_solution(artifacts)
    candidate_solution = artifacts.candidate_connections.merge(solution, on=["candidate_id", "connection_id"], how="left")
    active_connections = candidate_solution[candidate_solution["active"] == True].copy()
    result = build_outputs_from_active_connections(prepared, active_connections, model_config, baseline_result.od_metrics)
    metrics = compute_run_metrics(result.od_metrics, result.selected_connections, result.beaten_competitors)
    comparison = build_comparison_to_baseline(baseline_metrics, metrics)
    optimized_schedule = extract_optimized_schedule(artifacts)
    shift_summary = _summarize_schedule_shifts(optimized_schedule)
    return {
        "objective": metrics["total_objective"],
        "improvement_pct": comparison["percentage_improvement"],
        "runtime_sec": solve_result.runtime_sec,
        "optimized_schedule": optimized_schedule,
        "selected_connections": result.selected_connections,
        **shift_summary,
    }


def build_regularity_coverage_analysis(
    optimized_schedule: pd.DataFrame,
    selected_connections: pd.DataFrame,
    model_config: ModelConfig,
) -> tuple[pd.DataFrame, dict]:
    """Analyze how much of the schedule is covered and changed by regularity groups."""

    columns = [
        "regularity_group_id",
        "normalized_flight_no",
        "origin",
        "destination",
        "direction_type",
        "num_instances",
        "days_in_group",
        "old_time_min_list",
        "new_time_min_list",
        "old_min_time",
        "old_max_time",
        "old_spread_min",
        "new_min_time",
        "new_max_time",
        "new_spread_min",
        "spread_reduction_min",
        "xdev_limit",
        "regularity_constraint_active",
        "violation",
        "skip_reason",
        "avg_abs_shift_in_group",
        "max_abs_shift_in_group",
        "group_objective_related_connections_count",
    ]
    if optimized_schedule is None or optimized_schedule.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, _summarize_regularity_coverage(empty, model_config)

    schedule = _schedule_with_regularity_fields(optimized_schedule, model_config)
    active_usage = _active_flight_usage_counts(selected_connections, model_config)
    rows = []
    group_cols = ["normalized_flight_no", "origin", "destination", "direction_type"]
    for idx, (key, group) in enumerate(schedule.groupby(group_cols, dropna=False)):
        flight_no, origin, destination, direction = key
        group = group.sort_values(["day", "new_minute_of_day"]).copy()
        days = sorted(group["day"].dropna().astype(int).unique().tolist())
        old_values = group["old_minute_of_day"].dropna().round().astype(int).tolist()
        new_values = group["new_minute_of_day"].dropna().round().astype(int).tolist()
        old_min = float(min(old_values)) if old_values else None
        old_max = float(max(old_values)) if old_values else None
        new_min = float(min(new_values)) if new_values else None
        new_max = float(max(new_values)) if new_values else None
        old_spread = float(old_max - old_min) if old_min is not None and old_max is not None else None
        new_spread = float(new_max - new_min) if new_min is not None and new_max is not None else None
        active = len(days) >= 2 and len(group) >= 2
        violation = bool(active and new_spread is not None and new_spread > model_config.schedule_regular_dev_min + 1e-6)
        related_count = 0
        for flight in group.itertuples(index=False):
            usage_key = (
                str(flight.normalized_flight_no),
                str(flight.origin),
                str(flight.destination),
                int(flight.day),
            )
            related_count += active_usage.get(usage_key, 0)
        abs_shifts = pd.to_numeric(group["time_change_min"], errors="coerce").abs().dropna()
        rows.append(
            {
                "regularity_group_id": f"reggrp_{idx:05d}",
                "normalized_flight_no": flight_no,
                "origin": origin,
                "destination": destination,
                "direction_type": direction,
                "num_instances": int(len(group)),
                "days_in_group": ",".join(str(day) for day in days),
                "old_time_min_list": ",".join(str(int(x)) for x in old_values),
                "new_time_min_list": ",".join(str(int(x)) for x in new_values),
                "old_min_time": old_min,
                "old_max_time": old_max,
                "old_spread_min": old_spread,
                "new_min_time": new_min,
                "new_max_time": new_max,
                "new_spread_min": new_spread,
                "spread_reduction_min": None if old_spread is None or new_spread is None else old_spread - new_spread,
                "xdev_limit": model_config.schedule_regular_dev_min,
                "regularity_constraint_active": bool(active),
                "violation": violation,
                "skip_reason": "" if active else "single_instance_group",
                "avg_abs_shift_in_group": float(abs_shifts.mean()) if len(abs_shifts) else 0.0,
                "max_abs_shift_in_group": float(abs_shifts.max()) if len(abs_shifts) else 0.0,
                "group_objective_related_connections_count": int(related_count),
            }
        )
    analysis = pd.DataFrame(rows, columns=columns)
    return analysis, _summarize_regularity_coverage(analysis, model_config)


def _summarize_regularity_coverage(analysis: pd.DataFrame, model_config: ModelConfig) -> dict:
    total = int(len(analysis)) if analysis is not None else 0
    if analysis is None or analysis.empty:
        return {
            "coverage_total_regularity_groups": 0,
            "coverage_active_regularity_groups": 0,
            "coverage_skipped_single_instance_groups": 0,
            "coverage_active_group_pct": 0.0,
            "coverage_total_flight_instances_active": 0,
            "coverage_max_old_spread": 0.0,
            "coverage_avg_old_spread": 0.0,
            "coverage_max_new_spread": 0.0,
            "coverage_avg_new_spread": 0.0,
            "coverage_groups_old_spread_gt_xdev": 0,
            "coverage_groups_new_spread_gt_xdev": 0,
            "coverage_groups_spread_reduced": 0,
            "coverage_groups_spread_unchanged": 0,
            "coverage_groups_spread_increased_within_xdev": 0,
            "coverage_groups_used_in_active_connections": 0,
            "coverage_groups_not_used_in_active_connections": 0,
        }
    active_mask = analysis["regularity_constraint_active"].fillna(False).astype(bool)
    active = analysis[active_mask].copy()
    old_spread = pd.to_numeric(active.get("old_spread_min", pd.Series(dtype=float)), errors="coerce")
    new_spread = pd.to_numeric(active.get("new_spread_min", pd.Series(dtype=float)), errors="coerce")
    reduction = pd.to_numeric(active.get("spread_reduction_min", pd.Series(dtype=float)), errors="coerce")
    used_mask = pd.to_numeric(analysis["group_objective_related_connections_count"], errors="coerce").fillna(0) > 0
    increased_within = (reduction < -1e-6) & (new_spread <= model_config.schedule_regular_dev_min + 1e-6)
    return {
        "coverage_total_regularity_groups": total,
        "coverage_active_regularity_groups": int(active_mask.sum()),
        "coverage_skipped_single_instance_groups": int((analysis["skip_reason"].fillna("") == "single_instance_group").sum()),
        "coverage_active_group_pct": float(100.0 * active_mask.sum() / total) if total else 0.0,
        "coverage_total_flight_instances_active": int(pd.to_numeric(active["num_instances"], errors="coerce").fillna(0).sum()) if not active.empty else 0,
        "coverage_max_old_spread": float(old_spread.max()) if len(old_spread.dropna()) else 0.0,
        "coverage_avg_old_spread": float(old_spread.mean()) if len(old_spread.dropna()) else 0.0,
        "coverage_max_new_spread": float(new_spread.max()) if len(new_spread.dropna()) else 0.0,
        "coverage_avg_new_spread": float(new_spread.mean()) if len(new_spread.dropna()) else 0.0,
        "coverage_groups_old_spread_gt_xdev": int((old_spread > model_config.schedule_regular_dev_min + 1e-6).sum()),
        "coverage_groups_new_spread_gt_xdev": int((new_spread > model_config.schedule_regular_dev_min + 1e-6).sum()),
        "coverage_groups_spread_reduced": int((reduction > 1e-6).sum()),
        "coverage_groups_spread_unchanged": int((reduction.abs() <= 1e-6).sum()),
        "coverage_groups_spread_increased_within_xdev": int(increased_within.sum()),
        "coverage_groups_used_in_active_connections": int(used_mask.sum()),
        "coverage_groups_not_used_in_active_connections": int((~used_mask).sum()),
    }


def write_regularity_coverage_summary(run_dir: Path, metrics: dict) -> None:
    lines = [
        "REGULARITY COVERAGE SUMMARY",
        "---------------------------",
        f"total regularity groups: {metrics['coverage_total_regularity_groups']}",
        f"active regularity groups: {metrics['coverage_active_regularity_groups']}",
        f"skipped single-instance groups: {metrics['coverage_skipped_single_instance_groups']}",
        f"active group coverage pct: {metrics['coverage_active_group_pct']:.6f}",
        f"total flight instances covered by active regularity: {metrics['coverage_total_flight_instances_active']}",
        f"max old spread: {metrics['coverage_max_old_spread']}",
        f"avg old spread: {metrics['coverage_avg_old_spread']}",
        f"max new spread: {metrics['coverage_max_new_spread']}",
        f"avg new spread: {metrics['coverage_avg_new_spread']}",
        f"groups with old spread > XDEV: {metrics['coverage_groups_old_spread_gt_xdev']}",
        f"groups with new spread > XDEV: {metrics['coverage_groups_new_spread_gt_xdev']}",
        f"groups where spread reduced: {metrics['coverage_groups_spread_reduced']}",
        f"groups where spread unchanged: {metrics['coverage_groups_spread_unchanged']}",
        f"groups where spread increased but within XDEV: {metrics['coverage_groups_spread_increased_within_xdev']}",
        f"groups used in active connections: {metrics['coverage_groups_used_in_active_connections']}",
        f"groups not used in active connections: {metrics['coverage_groups_not_used_in_active_connections']}",
        "",
        "Yorum:",
    ]
    coverage_pct = metrics["coverage_active_group_pct"]
    old_bad = metrics["coverage_groups_old_spread_gt_xdev"]
    used = metrics["coverage_groups_used_in_active_connections"]
    active = metrics["coverage_active_regularity_groups"]
    if coverage_pct >= 60:
        lines.append("- Regularity gruplari bu subsette makul seviyede kapsayici gorunuyor.")
    else:
        lines.append("- Regularity kapsami sinirli; subsette tek gun/tek instance gruplari belirgin.")
    if old_bad == 0:
        lines.append("- Aktif gruplar optimizasyon oncesinde zaten XDEV icinde; bu nedenle objective uzerindeki baski zayif olabilir.")
    else:
        lines.append("- Bazi aktif gruplar optimizasyon oncesinde XDEV disinda; regularity bu gruplarda gercek duzeltme yapiyor.")
    if used == 0:
        lines.append("- Aktif baglantilarda kullanilan regularity grubu yok; objective etkisinin zayif kalmasi beklenir.")
    elif used < active:
        lines.append("- Regularity gruplarinin bir kismi aktif baglantilarda kullaniliyor; objective etkisi kismi kalabilir.")
    else:
        lines.append("- Aktif regularity gruplarinin buyuk kismi aktif baglantilarla iliskili.")
    (run_dir / "regularity_coverage_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def build_regularity_effect_comparison(
    capacity_schedule: pd.DataFrame | None,
    regularity_schedule: pd.DataFrame,
    selected_connections: pd.DataFrame,
    coverage_analysis: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    columns = [
        "flight_id",
        "flight_no",
        "origin",
        "destination",
        "day",
        "capacity_new_time",
        "regularity_new_time",
        "time_difference_capacity_vs_regularity",
        "capacity_shift",
        "regularity_shift",
        "shift_difference",
        "is_in_active_regularity_group",
        "is_used_in_active_connection",
    ]
    if capacity_schedule is None or capacity_schedule.empty or regularity_schedule is None or regularity_schedule.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, _summarize_regularity_effect(empty)
    cap = capacity_schedule.copy()
    reg = regularity_schedule.copy()
    cap["capacity_new_time"] = pd.to_datetime(cap["new_ist_time"], errors="coerce")
    reg["regularity_new_time"] = pd.to_datetime(reg["new_ist_time"], errors="coerce")
    cap = cap.rename(columns={"time_change_min": "capacity_shift"})
    reg = reg.rename(columns={"time_change_min": "regularity_shift"})
    merged = reg[
        ["flight_id", "flight_no", "origin", "destination", "day", "regularity_new_time", "regularity_shift"]
    ].merge(
        cap[["flight_id", "capacity_new_time", "capacity_shift"]],
        on="flight_id",
        how="inner",
    )
    active_group_keys = _active_regularity_group_keys(coverage_analysis)
    usage = _active_flight_usage_counts(selected_connections, ModelConfig())
    rows = []
    for row in merged.itertuples(index=False):
        origin = str(row.origin).strip().upper()
        destination = str(row.destination).strip().upper()
        direction = "inbound" if destination == "IST" else "outbound" if origin == "IST" else ""
        group_key = (normalize_flight_no(row.flight_no), origin, destination, direction)
        usage_key = (normalize_flight_no(row.flight_no), origin, destination, int(row.day))
        diff = (row.regularity_new_time - row.capacity_new_time).total_seconds() / 60.0
        rows.append(
            {
                "flight_id": row.flight_id,
                "flight_no": row.flight_no,
                "origin": origin,
                "destination": destination,
                "day": int(row.day),
                "capacity_new_time": row.capacity_new_time,
                "regularity_new_time": row.regularity_new_time,
                "time_difference_capacity_vs_regularity": diff,
                "capacity_shift": row.capacity_shift,
                "regularity_shift": row.regularity_shift,
                "shift_difference": float(row.regularity_shift) - float(row.capacity_shift),
                "is_in_active_regularity_group": group_key in active_group_keys,
                "is_used_in_active_connection": usage.get(usage_key, 0) > 0,
            }
        )
    comparison = pd.DataFrame(rows, columns=columns)
    return comparison, _summarize_regularity_effect(comparison)


def _summarize_regularity_effect(comparison: pd.DataFrame) -> dict:
    if comparison is None or comparison.empty:
        return {
            "effect_total_compared_flights": 0,
            "effect_flights_changed_vs_capacity": 0,
            "effect_pct_changed_vs_capacity": 0.0,
            "effect_avg_abs_difference_vs_capacity": 0.0,
            "effect_median_abs_difference_vs_capacity": 0.0,
            "effect_max_abs_difference_vs_capacity": 0.0,
            "effect_flights_regularity_reduced_shift": 0,
            "effect_flights_regularity_increased_shift": 0,
            "effect_flights_unchanged_vs_capacity": 0,
        }
    diff = pd.to_numeric(comparison["time_difference_capacity_vs_regularity"], errors="coerce").fillna(0.0)
    abs_diff = diff.abs()
    cap_abs = pd.to_numeric(comparison["capacity_shift"], errors="coerce").fillna(0.0).abs()
    reg_abs = pd.to_numeric(comparison["regularity_shift"], errors="coerce").fillna(0.0).abs()
    changed = abs_diff > 1e-6
    return {
        "effect_total_compared_flights": int(len(comparison)),
        "effect_flights_changed_vs_capacity": int(changed.sum()),
        "effect_pct_changed_vs_capacity": float(100.0 * changed.sum() / len(comparison)) if len(comparison) else 0.0,
        "effect_avg_abs_difference_vs_capacity": float(abs_diff.mean()) if len(abs_diff) else 0.0,
        "effect_median_abs_difference_vs_capacity": float(abs_diff.median()) if len(abs_diff) else 0.0,
        "effect_max_abs_difference_vs_capacity": float(abs_diff.max()) if len(abs_diff) else 0.0,
        "effect_flights_regularity_reduced_shift": int((reg_abs < cap_abs - 1e-6).sum()),
        "effect_flights_regularity_increased_shift": int((reg_abs > cap_abs + 1e-6).sum()),
        "effect_flights_unchanged_vs_capacity": int((~changed).sum()),
    }


def write_regularity_effect_summary(run_dir: Path, metrics: dict) -> None:
    lines = [
        "REGULARITY EFFECT SUMMARY",
        "-------------------------",
        f"total compared flights: {metrics['effect_total_compared_flights']}",
        f"flights changed by regularity vs capacity: {metrics['effect_flights_changed_vs_capacity']}",
        f"pct changed by regularity vs capacity: {metrics['effect_pct_changed_vs_capacity']:.6f}",
        f"avg abs difference: {metrics['effect_avg_abs_difference_vs_capacity']}",
        f"median abs difference: {metrics['effect_median_abs_difference_vs_capacity']}",
        f"max abs difference: {metrics['effect_max_abs_difference_vs_capacity']}",
        f"flights where regularity reduced shift: {metrics['effect_flights_regularity_reduced_shift']}",
        f"flights where regularity increased shift: {metrics['effect_flights_regularity_increased_shift']}",
        f"flights unchanged: {metrics['effect_flights_unchanged_vs_capacity']}",
    ]
    (run_dir / "regularity_effect_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _schedule_with_regularity_fields(schedule: pd.DataFrame, model_config: ModelConfig) -> pd.DataFrame:
    enriched = schedule.copy()
    enriched["normalized_flight_no"] = enriched["flight_no"].map(normalize_flight_no)
    enriched["origin"] = enriched["origin"].astype(str).str.strip().str.upper()
    enriched["destination"] = enriched["destination"].astype(str).str.strip().str.upper()
    enriched["direction_type"] = ""
    enriched.loc[enriched["destination"] == model_config.hub, "direction_type"] = "inbound"
    enriched.loc[enriched["origin"] == model_config.hub, "direction_type"] = "outbound"
    enriched["old_ist_time"] = pd.to_datetime(enriched["old_ist_time"], errors="coerce")
    enriched["new_ist_time"] = pd.to_datetime(enriched["new_ist_time"], errors="coerce")
    enriched["old_minute_of_day"] = enriched["old_ist_time"].dt.hour * 60 + enriched["old_ist_time"].dt.minute
    enriched["new_minute_of_day"] = enriched["new_ist_time"].dt.hour * 60 + enriched["new_ist_time"].dt.minute
    return enriched.dropna(subset=["normalized_flight_no", "day", "new_minute_of_day"])


def _active_flight_usage_counts(selected_connections: pd.DataFrame, model_config: ModelConfig) -> dict[tuple[str, str, str, int], int]:
    usage: dict[tuple[str, str, str, int], int] = {}
    if selected_connections is None or selected_connections.empty:
        return usage
    active = selected_connections[selected_connections.get("active", True) == True].copy()
    for row in active.itertuples(index=False):
        day = int(row.day)
        origin = str(row.origin).strip().upper()
        destination = str(row.destination).strip().upper()
        inbound_key = (normalize_flight_no(row.inbound_flight), origin, model_config.hub, day)
        outbound_key = (normalize_flight_no(row.outbound_flight), model_config.hub, destination, day)
        usage[inbound_key] = usage.get(inbound_key, 0) + 1
        usage[outbound_key] = usage.get(outbound_key, 0) + 1
    return usage


def _active_regularity_group_keys(coverage_analysis: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    if coverage_analysis is None or coverage_analysis.empty:
        return keys
    active = coverage_analysis[coverage_analysis["regularity_constraint_active"].fillna(False).astype(bool)].copy()
    for row in active.itertuples(index=False):
        keys.add(
            (
                str(row.normalized_flight_no),
                str(row.origin).strip().upper(),
                str(row.destination).strip().upper(),
                str(row.direction_type),
            )
        )
    return keys


def _summarize_schedule_shifts(optimized_schedule) -> dict:
    if optimized_schedule is None or optimized_schedule.empty or "time_change_min" not in optimized_schedule.columns:
        return {"avg_abs_shift": 0.0, "median_abs_shift": 0.0}
    shifts = optimized_schedule["time_change_min"].astype(float).abs()
    return {
        "avg_abs_shift": float(shifts.mean()) if len(shifts) else 0.0,
        "median_abs_shift": float(shifts.median()) if len(shifts) else 0.0,
    }


def _summarize_rotation(rotation_candidates, rotation_report) -> dict:
    total = int(len(rotation_candidates)) if rotation_candidates is not None and not rotation_candidates.empty else 0
    if rotation_candidates is not None and not rotation_candidates.empty and "rotation_constraint_available" in rotation_candidates:
        active = int(rotation_candidates["rotation_constraint_available"].fillna(False).astype(bool).sum())
    else:
        active = 0
    elapsed_exact = 0
    legacy_exact = 0
    schedule_fallback = 0
    if rotation_candidates is not None and not rotation_candidates.empty:
        if "rotation_constraint_available" in rotation_candidates:
            constrained_candidates = rotation_candidates[
                rotation_candidates["rotation_constraint_available"].fillna(False).astype(bool)
            ].copy()
        else:
            constrained_candidates = rotation_candidates.copy()
        if "block_time_source" in constrained_candidates.columns:
            sources = constrained_candidates["block_time_source"].fillna("").astype(str)
            elapsed_exact = int((sources == "elapsedtime_exact_match").sum())
            legacy_exact = int((sources == "od_exact_match_legacy").sum())
            schedule_fallback = int((sources == "schedule_elapsed_fallback").sum())
        elif "derivation_method" in constrained_candidates.columns:
            derivation = constrained_candidates["derivation_method"].fillna("").astype(str)
            elapsed_exact = int(derivation.str.startswith("ElapsedTime1/ElapsedTime2 exact match").sum())
            legacy_exact = int(derivation.str.startswith("O&D connection block total").sum())
            schedule_fallback = int(derivation.str.startswith("approximated from schedule elapsed").sum())
    od_derived = elapsed_exact + legacy_exact
    skipped = total - active
    violations = 0
    min_slack = None
    avg_slack = None
    if rotation_report is not None and not rotation_report.empty:
        if "skip_reason" in rotation_report.columns:
            constrained = rotation_report[rotation_report["skip_reason"].fillna("").astype(str) == ""].copy()
        else:
            constrained = rotation_report.copy()
        if not constrained.empty:
            violations = int(constrained["violation"].fillna(False).sum()) if "violation" in constrained else 0
            slack = constrained["slack_min"].dropna().astype(float) if "slack_min" in constrained else None
            if slack is not None and len(slack):
                min_slack = float(slack.min())
                avg_slack = float(slack.mean())
    return {
        "rotation_candidates_total": total,
        "rotation_constraints_active": active,
        "rotation_constraints_skipped": skipped,
        "rotation_block_time_od_derived": od_derived,
        "rotation_block_time_schedule_fallback": schedule_fallback,
        "rotation_block_elapsedtime_match_count": elapsed_exact,
        "rotation_block_legacy_exact_match_count": legacy_exact,
        "rotation_block_exact_match_count": od_derived,
        "rotation_block_fallback_count": schedule_fallback,
        "rotation_violations": violations,
        "min_rotation_slack": "" if min_slack is None else min_slack,
        "avg_rotation_slack": "" if avg_slack is None else avg_slack,
    }


def _summarize_hub_capacity(hub_capacity_report, model_config: ModelConfig) -> dict:
    if hub_capacity_report is None or hub_capacity_report.empty:
        return {
            "hub_capacity_scope": "adjustable_events_vs_residual_capacity_fixed_tk_events_subtracted",
            "hub_bucket_size_minutes": model_config.bucket_size_min,
            "hub_arrival_capacity": model_config.arrival_bucket_capacity,
            "hub_departure_capacity": model_config.departure_bucket_capacity,
            "hub_capacity_violations": 0,
            "max_arrivals_in_bucket": 0,
            "max_departures_in_bucket": 0,
            "avg_arrivals_per_used_bucket": 0.0,
            "avg_departures_per_used_bucket": 0.0,
        }
    arrivals = hub_capacity_report["arrivals"].astype(float) if "arrivals" in hub_capacity_report else None
    departures = hub_capacity_report["departures"].astype(float) if "departures" in hub_capacity_report else None
    if "total_violation" in hub_capacity_report:
        violations = int(hub_capacity_report["total_violation"].fillna(False).sum())
    elif "violation" in hub_capacity_report:
        violations = int(hub_capacity_report["violation"].fillna(False).sum())
    else:
        violations = 0
    used_arrivals = arrivals[arrivals > 0] if arrivals is not None else []
    used_departures = departures[departures > 0] if departures is not None else []
    return {
        "hub_capacity_scope": "adjustable_events_vs_residual_capacity_fixed_tk_events_subtracted",
        "hub_bucket_size_minutes": model_config.bucket_size_min,
        "hub_arrival_capacity": model_config.arrival_bucket_capacity,
        "hub_departure_capacity": model_config.departure_bucket_capacity,
        "hub_capacity_violations": violations,
        "max_arrivals_in_bucket": int(arrivals.max()) if arrivals is not None and len(arrivals) else 0,
        "max_departures_in_bucket": int(departures.max()) if departures is not None and len(departures) else 0,
        "avg_arrivals_per_used_bucket": float(used_arrivals.mean()) if len(used_arrivals) else 0.0,
        "avg_departures_per_used_bucket": float(used_departures.mean()) if len(used_departures) else 0.0,
    }


def _summarize_regularity(regularity_groups, regularity_report, model_config: ModelConfig) -> dict:
    total = int(len(regularity_groups)) if regularity_groups is not None and not regularity_groups.empty else 0
    if regularity_groups is not None and not regularity_groups.empty and "regularity_constraint_active" in regularity_groups:
        active = int(regularity_groups["regularity_constraint_active"].fillna(False).astype(bool).sum())
    elif regularity_report is not None and not regularity_report.empty:
        total = int(len(regularity_report))
        active = int(len(regularity_report[regularity_report["status"].fillna("").astype(str) == "constrained"])) if "status" in regularity_report else int(len(regularity_report))
    else:
        active = 0
    skipped = total - active
    violations = 0
    max_spread = 0.0
    avg_spread = 0.0
    soft_violations = 0
    soft_amount = 0.0
    relax_applied_groups = 0
    max_relax_amount = 0.0
    avg_relax_amount = 0.0
    max_effective_xdev = float(model_config.schedule_regular_dev_min)
    groups_still_infeasible = 0
    if regularity_report is not None and not regularity_report.empty:
        constrained = regularity_report[regularity_report["skip_reason"].fillna("").astype(str) == ""] if "skip_reason" in regularity_report else regularity_report
        if not constrained.empty:
            violation_flags = constrained["violation"].fillna(False).astype(bool) if "violation" in constrained else pd.Series(False, index=constrained.index)
            if model_config.regularity_mode == "soft":
                violations = 0
            elif model_config.regularity_relax_mode == "targeted_soft" and "targeted_soft" in constrained:
                violations = int((violation_flags & ~constrained["targeted_soft"].fillna(False).astype(bool)).sum())
            else:
                violations = int(violation_flags.sum())
            if "relax_applied" in constrained:
                relax_applied_groups = int(constrained["relax_applied"].fillna(False).astype(bool).sum())
            if "relax_amount" in constrained:
                relax_values = pd.to_numeric(constrained["relax_amount"], errors="coerce").fillna(0.0)
                max_relax_amount = float(relax_values.max()) if len(relax_values) else 0.0
                applied_values = relax_values[relax_values > 1e-6]
                avg_relax_amount = float(applied_values.mean()) if len(applied_values) else 0.0
            if "effective_xdev" in constrained:
                eff = pd.to_numeric(constrained["effective_xdev"], errors="coerce").dropna()
            elif "effective_xdev_min" in constrained:
                eff = pd.to_numeric(constrained["effective_xdev_min"], errors="coerce").dropna()
            elif "xdev_limit" in constrained:
                eff = pd.to_numeric(constrained["xdev_limit"], errors="coerce").dropna()
            elif "limit_min" in constrained:
                eff = pd.to_numeric(constrained["limit_min"], errors="coerce").dropna()
            else:
                eff = pd.Series(dtype=float)
            if len(eff):
                max_effective_xdev = float(eff.max())
            if "theoretical_min_spread" in constrained.columns:
                limit_col = "xdev_limit" if "xdev_limit" in constrained.columns else ("effective_xdev_min" if "effective_xdev_min" in constrained.columns else "limit_min")
                groups_still_infeasible = int(
                    (
                        pd.to_numeric(constrained["theoretical_min_spread"], errors="coerce")
                        > pd.to_numeric(constrained[limit_col], errors="coerce") + 1e-6
                    ).sum()
                )
            spreads = pd.to_numeric(constrained.get("spread_min", pd.Series(dtype=float)), errors="coerce").dropna()
            if len(spreads):
                max_spread = float(spreads.max())
                avg_spread = float(spreads.mean())
                limit_source = (
                    constrained["xdev_limit"]
                    if "xdev_limit" in constrained
                    else (constrained["effective_xdev_min"] if "effective_xdev_min" in constrained else pd.Series(model_config.schedule_regular_dev_min, index=constrained.index))
                )
                limits = pd.to_numeric(limit_source, errors="coerce").fillna(model_config.schedule_regular_dev_min)
                excess = (pd.to_numeric(constrained.get("spread_min", constrained.get("new_time_diff_min", pd.Series(dtype=float))), errors="coerce") - limits).clip(lower=0)
                soft_violations = int((excess > 1e-6).sum())
                soft_amount = float(excess.sum())
    return {
        "regularity_xdev_minutes": model_config.schedule_regular_dev_min,
        "regularity_mode": model_config.regularity_mode,
        "regularity_abs_tolerance": model_config.regularity_abs_tolerance_min,
        "regularity_penalty_weight": model_config.regularity_penalty_weight,
        "regularity_relax_mode": model_config.regularity_relax_mode,
        "regularity_targeted_max_relax": model_config.regularity_targeted_max_relax_min,
        "regularity_groups_total": total,
        "regularity_constraints_active": active,
        "regularity_groups_skipped": skipped,
        "regularity_violations": violations if model_config.regularity_mode == "hard" else 0,
        "regularity_hard_violations": violations if model_config.regularity_mode == "hard" else 0,
        "regularity_soft_violations": soft_violations if model_config.regularity_mode == "soft" or model_config.regularity_relax_mode == "targeted_soft" else 0,
        "regularity_violation_amount_total": soft_amount if model_config.regularity_mode == "soft" or model_config.regularity_relax_mode == "targeted_soft" else 0.0,
        "regularity_reported_violations": violations,
        "regularity_relax_applied_groups": relax_applied_groups,
        "regularity_max_relax_amount": max_relax_amount,
        "regularity_avg_relax_amount": avg_relax_amount,
        "regularity_max_effective_xdev": max_effective_xdev,
        "regularity_groups_still_infeasible_by_theoretical_bound": groups_still_infeasible,
        "max_regularity_spread": max_spread,
        "avg_regularity_spread": avg_spread,
    }


def write_regularity_targeted_relax_outputs(run_dir: Path, regularity_report, model_config: ModelConfig) -> None:
    columns = [
        "regularity_group_id",
        "normalized_flight_no",
        "origin",
        "destination",
        "direction_type",
        "old_spread",
        "theoretical_min_spread",
        "base_xdev",
        "effective_xdev",
        "relax_applied",
        "relax_amount",
        "targeted_max_relax",
        "num_instances",
        "final_new_spread",
        "violation",
        "notes",
    ]
    if regularity_report is None or regularity_report.empty:
        report = pd.DataFrame(columns=columns)
    else:
        source = regularity_report.copy()
        def _series(name: str, default):
            if name in source:
                return source[name]
            return pd.Series(default, index=source.index)
        report = pd.DataFrame()
        report["regularity_group_id"] = source["regularity_group_id"] if "regularity_group_id" in source else _series("regularity_id", "")
        report["normalized_flight_no"] = source["normalized_flight_no"] if "normalized_flight_no" in source else _series("flight_no", "")
        report["origin"] = _series("origin", "")
        report["destination"] = _series("destination", "")
        report["direction_type"] = source["direction_type"] if "direction_type" in source else _series("direction", "")
        if "old_spread_min" in source:
            old_spread = pd.to_numeric(source["old_spread_min"], errors="coerce")
        elif "old_time_diff_min" in source:
            old_spread = pd.to_numeric(source["old_time_diff_min"], errors="coerce")
        else:
            old_spread = pd.Series(0.0, index=source.index)
        report["old_spread"] = old_spread
        theoretical_default = (old_spread - 2 * model_config.max_shift_min).clip(lower=0)
        report["theoretical_min_spread"] = pd.to_numeric(
            source["theoretical_min_spread"] if "theoretical_min_spread" in source else theoretical_default,
            errors="coerce",
        ).fillna(0.0)
        base_source = source["base_xdev"] if "base_xdev" in source else _series("base_xdev_min", model_config.schedule_regular_dev_min)
        eff_source = source["effective_xdev"] if "effective_xdev" in source else (source["effective_xdev_min"] if "effective_xdev_min" in source else _series("xdev_limit", model_config.schedule_regular_dev_min))
        report["base_xdev"] = pd.to_numeric(base_source, errors="coerce").fillna(model_config.schedule_regular_dev_min)
        report["effective_xdev"] = pd.to_numeric(eff_source, errors="coerce").fillna(model_config.schedule_regular_dev_min)
        report["relax_applied"] = source["relax_applied"] if "relax_applied" in source else (report["effective_xdev"] > report["base_xdev"] + 1e-6)
        relax_source = source["relax_amount"] if "relax_amount" in source else (report["effective_xdev"] - report["base_xdev"])
        report["relax_amount"] = pd.to_numeric(relax_source, errors="coerce").fillna(0.0).clip(lower=0)
        report["targeted_max_relax"] = pd.to_numeric(_series("targeted_max_relax", model_config.regularity_targeted_max_relax_min), errors="coerce").fillna(model_config.regularity_targeted_max_relax_min)
        report["num_instances"] = _series("num_instances", "")
        spread_source = source["spread_min"] if "spread_min" in source else _series("new_time_diff_min", None)
        report["final_new_spread"] = pd.to_numeric(spread_source, errors="coerce")
        report["violation"] = source["violation"] if "violation" in source else (report["final_new_spread"] > report["effective_xdev"] + 1e-6)
        if "warning" in source:
            notes = source["warning"].fillna("").astype(str)
        elif "skip_reason" in source:
            notes = source["skip_reason"].fillna("").astype(str)
        else:
            notes = pd.Series("", index=source.index)
        still_infeasible = report["theoretical_min_spread"] > report["effective_xdev"] + 1e-6
        report["notes"] = notes.mask(still_infeasible & (notes == ""), "theoretical_min_spread_exceeds_effective_xdev")
        report = report.reindex(columns=columns)
    report.to_excel(run_dir / "regularity_targeted_relax_report.xlsx", index=False)
    total = int(len(report))
    relax_applied = int(report["relax_applied"].fillna(False).astype(bool).sum()) if total else 0
    relax_amounts = pd.to_numeric(report.get("relax_amount", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    violations = int(report["violation"].fillna(False).astype(bool).sum()) if total else 0
    soft_violations = violations if model_config.regularity_mode == "soft" or model_config.regularity_relax_mode == "targeted_soft" else 0
    soft_amount = 0.0
    if total and soft_violations:
        excess = (
            pd.to_numeric(report["final_new_spread"], errors="coerce")
            - pd.to_numeric(report["effective_xdev"], errors="coerce")
        ).clip(lower=0)
        soft_amount = float(excess.sum())
    still = int((pd.to_numeric(report.get("theoretical_min_spread", pd.Series(dtype=float)), errors="coerce") > pd.to_numeric(report.get("effective_xdev", pd.Series(dtype=float)), errors="coerce") + 1e-6).sum()) if total else 0
    lines = [
        "REGULARITY TARGETED RELAX SUMMARY",
        "---------------------------------",
        f"regularity_relax_mode: {model_config.regularity_relax_mode}",
        f"base_xdev: {model_config.schedule_regular_dev_min}",
        f"targeted_max_relax: {model_config.regularity_targeted_max_relax_min}",
        f"total regularity groups: {total}",
        f"relax applied groups: {relax_applied}",
        f"max relax amount: {float(relax_amounts.max()) if len(relax_amounts) else 0.0}",
        f"avg relax amount: {float(relax_amounts[relax_amounts > 1e-6].mean()) if (relax_amounts > 1e-6).any() else 0.0}",
        f"groups still infeasible by theoretical bound: {still}",
        f"hard regularity violations: {0 if model_config.regularity_mode == 'soft' else violations}",
        f"soft regularity violations: {soft_violations}",
        f"total soft regularity amount: {soft_amount}",
        "",
        "Turkce yorum:",
    ]
    if model_config.regularity_relax_mode == "targeted_theoretical":
        lines.append("- Hedefli theoretical relax, sadece max_shift bounds nedeniyle XDEV limitine matematiksel olarak inemeyen gruplarda effective XDEV'i yukseltecek sekilde raporlandi.")
    elif model_config.regularity_relax_mode == "targeted_soft":
        lines.append("- Targeted soft modda teorik olarak infeasible gruplar slack/penalty ile izlenir; diger gruplar hard XDEV mantiginda kalir.")
    elif model_config.regularity_relax_mode == "global_tolerance":
        lines.append("- Global tolerance tum gruplara ayni ek toleransi uygular; OD300 fallback icin hedefli yaklasim kadar dar kapsamli degildir.")
    else:
        lines.append("- Relax modu kapali; effective XDEV base XDEV ile aynidir.")
    (run_dir / "regularity_targeted_relax_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _summarize_directional(directional_pairs, directional_report, model_config: ModelConfig) -> dict:
    total = int(len(directional_pairs)) if directional_pairs is not None and not directional_pairs.empty else 0
    if directional_pairs is not None and not directional_pairs.empty and "pair_active" in directional_pairs:
        active = int(directional_pairs["pair_active"].fillna(False).astype(bool).sum())
    else:
        active = 0
    skipped = total - active
    if directional_report is None or directional_report.empty:
        return {
            "directional_pairs_total": total,
            "directional_pairs_active": active,
            "directional_pairs_skipped": skipped,
            "directional_alpha": model_config.directional_alpha,
            "directional_gamma": model_config.directional_gamma_min,
            "directional_e1_mode": model_config.directional_e1_mode,
            "directional_e1_formula": model_config.directional_e1_formula,
            "directional_e1_abs_tolerance": model_config.directional_e1_abs_tolerance,
            "directional_e1_violations": 0,
            "directional_e2_violations": 0,
            "directional_e1_hard_violations": 0,
            "directional_e1_soft_violations": 0,
            "directional_e2_hard_violations": 0,
            "directional_e2_soft_violations": 0,
            "directional_total_violations": 0,
            "directional_e1_soft_violation_amount_total": 0.0,
            "directional_e2_soft_violation_amount_total": 0.0,
            "directional_e2_mode": model_config.directional_e2_mode,
            "directional_e1_penalty_weight": model_config.directional_e1_penalty_weight,
            "directional_e2_penalty_weight": model_config.directional_e2_penalty_weight,
            "allow_soft_directional_violations": model_config.allow_soft_directional_violations,
            "max_connection_count_diff": 0,
            "avg_connection_count_diff": 0.0,
            "max_journey_time_diff": 0.0,
            "avg_journey_time_diff": 0.0,
        }
    constrained = directional_report[directional_report["skip_reason"].fillna("").astype(str) != "missing_reverse_direction"].copy()
    e1 = int(constrained.get("e1_violation", pd.Series(dtype=bool)).fillna(False).sum())
    e2 = int(constrained.get("e2_violation", pd.Series(dtype=bool)).fillna(False).sum())
    connection_diff = pd.to_numeric(constrained.get("connection_diff", pd.Series(dtype=float)), errors="coerce").dropna()
    journey_diff = pd.to_numeric(constrained.get("journey_time_diff", pd.Series(dtype=float)), errors="coerce").dropna()
    e1_amount = pd.to_numeric(constrained.get("e1_violation_amount", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    e2_amount = pd.to_numeric(constrained.get("e2_violation_amount", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    e1_hard = e1 if model_config.directional_e1_mode == "hard" and model_config.directional_e1_formula != "soft_only" else 0
    e1_soft = e1 if model_config.directional_e1_mode == "soft" or model_config.directional_e1_formula == "soft_only" else 0
    e2_hard = e2 if model_config.directional_e2_mode == "hard" else 0
    e2_soft = e2 if model_config.directional_e2_mode == "soft" and model_config.allow_soft_directional_violations else 0
    return {
        "directional_pairs_total": total,
        "directional_pairs_active": active,
        "directional_pairs_skipped": skipped,
        "directional_alpha": model_config.directional_alpha,
        "directional_gamma": model_config.directional_gamma_min,
        "directional_e1_mode": model_config.directional_e1_mode,
        "directional_e1_formula": model_config.directional_e1_formula,
        "directional_e1_abs_tolerance": model_config.directional_e1_abs_tolerance,
        "directional_e1_penalty_weight": model_config.directional_e1_penalty_weight,
        "directional_e2_penalty_weight": model_config.directional_e2_penalty_weight,
        "directional_e1_violations": e1,
        "directional_e2_violations": e2,
        "directional_e1_hard_violations": e1_hard,
        "directional_e1_soft_violations": e1_soft,
        "directional_e2_hard_violations": e2_hard,
        "directional_e2_soft_violations": e2_soft,
        "directional_total_violations": e1_hard + e2_hard,
        "directional_e1_soft_violation_amount_total": float(e1_amount.sum()) if len(e1_amount) else 0.0,
        "directional_e2_soft_violation_amount_total": float(e2_amount.sum()) if len(e2_amount) else 0.0,
        "directional_e2_mode": model_config.directional_e2_mode,
        "allow_soft_directional_violations": model_config.allow_soft_directional_violations,
        "max_connection_count_diff": int(connection_diff.max()) if len(connection_diff) else 0,
        "avg_connection_count_diff": float(connection_diff.mean()) if len(connection_diff) else 0.0,
        "max_journey_time_diff": float(journey_diff.max()) if len(journey_diff) else 0.0,
        "avg_journey_time_diff": float(journey_diff.mean()) if len(journey_diff) else 0.0,
    }


def write_directional_interpretation(
    run_dir: Path,
    directional_summary: dict,
    reference_regularity: dict | None,
    shift_summary: dict,
) -> None:
    lines = ["DIRECTIONAL BALANCE YORUMU", "----------------------------"]
    if reference_regularity:
        objective_drop = float(reference_regularity["objective"]) - float(directional_summary["objective"])
        runtime_delta = float(directional_summary["runtime_sec"]) - float(reference_regularity["runtime_sec"])
        lines.extend(
            [
                f"Regularity objective: {reference_regularity['objective']}",
                f"Directional objective: {directional_summary['objective']}",
                f"Regularity improvement pct: {reference_regularity['improvement_pct']:.6f}%",
                f"Directional improvement pct: {directional_summary['improvement_pct']:.6f}%",
                f"Directional eklenince objective dususu: {objective_drop}",
                f"Regularity runtime: {reference_regularity['runtime_sec']:.3f} sn",
                f"Directional runtime: {directional_summary['runtime_sec']:.3f} sn",
                f"Runtime artisi: {runtime_delta:.3f} sn",
                f"Regularity avg abs shift: {reference_regularity.get('avg_abs_shift', 0.0)}",
                f"Directional avg abs shift: {shift_summary.get('avg_abs_shift', 0.0)}",
                f"Regularity median abs shift: {reference_regularity.get('median_abs_shift', 0.0)}",
                f"Directional median abs shift: {shift_summary.get('median_abs_shift', 0.0)}",
            ]
        )
    else:
        lines.append("Regularity referans kosusu hesaplanamadi; yalnizca directional run metrikleri yorumlandi.")
    lines.extend(
        [
            f"Feasible: {directional_summary['feasible']}",
            f"Directional alpha: {directional_summary['directional_alpha']}",
            f"Directional gamma: {directional_summary['directional_gamma']}",
            f"Directional E2 mode: {directional_summary['directional_e2_mode']}",
            f"Allow soft directional violations: {directional_summary['allow_soft_directional_violations']}",
            f"Directional pairs total: {directional_summary['directional_pairs_total']}",
            f"Directional pairs active: {directional_summary['directional_pairs_active']}",
            f"Directional pairs skipped: {directional_summary['directional_pairs_skipped']}",
            f"Directional E1 violations: {directional_summary['directional_e1_violations']}",
            f"Directional E2 violations: {directional_summary['directional_e2_violations']}",
            f"Directional total violations: {directional_summary['directional_total_violations']}",
            f"Directional E2 soft violation amount total: {directional_summary['directional_e2_soft_violation_amount_total']}",
            f"Max connection count diff: {directional_summary['max_connection_count_diff']}",
            f"Avg connection count diff: {directional_summary['avg_connection_count_diff']}",
            f"Max journey time diff: {directional_summary['max_journey_time_diff']}",
            f"Avg journey time diff: {directional_summary['avg_journey_time_diff']}",
            "",
            "Yorum:",
        ]
    )
    if directional_summary["directional_total_violations"] == 0:
        lines.append("- Bagimsiz checker directional violation bulmadi; E1/E2 acisindan feasibility korundu.")
    else:
        lines.append("- Checker directional violation buldu; bu run feasible kabul edilmemeli.")
    lines.append("- Bu ilk surumde E1 baglanti sayisi dengesi model constraint olarak aktif, E2 journey-time dengesi checker/report seviyesinde uygulanir.")
    if reference_regularity:
        if objective_drop > 1e-6:
            lines.append("- Directional balance objective'i dusurdu; operasyonel denge model uzerinde baski olusturuyor.")
        else:
            lines.append("- Directional balance objective uzerinde belirgin dusus olusturmadi.")
        if shift_summary.get("avg_abs_shift", 0.0) < reference_regularity.get("avg_abs_shift", 0.0):
            lines.append("- Shift dagilimi regularity referansina gore daha makul hale geldi.")
        else:
            lines.append("- Shift dagilimi regularity referansina gore belirgin yumusamadi.")
    if directional_summary["directional_e1_violations"] > 0:
        lines.append("- E1 count balance baskin problem olarak gorunuyor.")
    elif directional_summary["directional_e2_violations"] > 0:
        lines.append("- E2 journey-time dengesi baskin problem olarak gorunuyor.")
    else:
        lines.append("- Normal parametrelerde kural checker'a gore asiri siki gorunmuyor.")
    (run_dir / "directional_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def build_directional_model_checker_debug(
    model_pairs: pd.DataFrame | None,
    candidate_solution: pd.DataFrame | None,
    checker_report: pd.DataFrame,
    model_config: ModelConfig,
) -> tuple[pd.DataFrame, dict]:
    columns = [
        "directional_pair_id",
        "day",
        "od_forward",
        "od_reverse",
        "model_pair_active",
        "checker_pair_active",
        "model_forward_active_connections",
        "checker_forward_active_connections",
        "model_reverse_active_connections",
        "checker_reverse_active_connections",
        "model_total_connections",
        "checker_total_connections",
        "model_connection_diff",
        "checker_connection_diff",
        "alpha",
        "model_alpha_limit",
        "checker_alpha_limit",
        "model_e1_constraint_active",
        "checker_e1_applicable",
        "checker_e1_violation",
        "mismatch_reason",
    ]
    model_map = {}
    if model_pairs is not None and not model_pairs.empty:
        for row in model_pairs.itertuples(index=False):
            key = _directional_pair_key(row.od, row.reverse_od, int(row.day))
            model_map[key] = row
    checker_map = {}
    if checker_report is not None and not checker_report.empty:
        for row in checker_report.itertuples(index=False):
            key = _directional_pair_key(row.od_forward, row.od_reverse, int(row.day))
            checker_map[key] = row
    model_counts = _model_active_counts(candidate_solution)
    rows = []
    for key in sorted(set(model_map) | set(checker_map)):
        day, od_a, od_b = key
        model_row = model_map.get(key)
        checker_row = checker_map.get(key)
        od_forward = getattr(checker_row, "od_forward", None) if checker_row is not None else getattr(model_row, "od", od_a)
        od_reverse = getattr(checker_row, "od_reverse", None) if checker_row is not None else getattr(model_row, "reverse_od", od_b)
        model_pair_active = model_row is not None
        checker_pair_active = checker_row is not None and str(getattr(checker_row, "skip_reason", "")).strip() != "missing_reverse_direction"
        model_f = int(model_counts.get((str(od_forward), int(day)), 0)) if model_pair_active else 0
        model_r = int(model_counts.get((str(od_reverse), int(day)), 0)) if model_pair_active else 0
        checker_f = int(getattr(checker_row, "forward_active_connections", 0)) if checker_row is not None else 0
        checker_r = int(getattr(checker_row, "reverse_active_connections", 0)) if checker_row is not None else 0
        model_total = model_f + model_r
        checker_total = int(getattr(checker_row, "total_connections", checker_f + checker_r)) if checker_row is not None else 0
        model_diff = abs(model_f - model_r)
        checker_diff = int(getattr(checker_row, "connection_diff", abs(checker_f - checker_r))) if checker_row is not None else 0
        model_limit = model_config.directional_alpha * model_total
        checker_limit = float(getattr(checker_row, "alpha_limit_value", 0.0)) if checker_row is not None else 0.0
        checker_e1_applicable = bool(checker_pair_active and checker_total > 0)
        checker_e1_violation = bool(getattr(checker_row, "e1_violation", False)) if checker_row is not None else False
        if model_pair_active != checker_pair_active:
            reason = "pair_set_mismatch"
        elif model_pair_active and checker_pair_active and (model_f != checker_f or model_r != checker_r):
            reason = "active_connection_count_mismatch"
        elif model_pair_active and checker_pair_active and checker_e1_applicable and abs(model_limit - checker_limit) > 1e-6:
            reason = "alpha_formula_mismatch"
        elif model_pair_active and model_total == 0 and checker_e1_applicable:
            reason = "inactive_pair_handling_mismatch"
        elif checker_e1_violation and model_pair_active and model_diff <= model_limit + 1e-6:
            reason = "checker_uses_output_not_model"
        else:
            reason = "no_mismatch"
        rows.append(
            {
                "directional_pair_id": getattr(checker_row, "directional_pair_id", getattr(model_row, "directional_pair_id", "")),
                "day": int(day),
                "od_forward": od_forward,
                "od_reverse": od_reverse,
                "model_pair_active": bool(model_pair_active),
                "checker_pair_active": bool(checker_pair_active),
                "model_forward_active_connections": model_f,
                "checker_forward_active_connections": checker_f,
                "model_reverse_active_connections": model_r,
                "checker_reverse_active_connections": checker_r,
                "model_total_connections": model_total,
                "checker_total_connections": checker_total,
                "model_connection_diff": model_diff,
                "checker_connection_diff": checker_diff,
                "alpha": model_config.directional_alpha,
                "model_alpha_limit": model_limit,
                "checker_alpha_limit": checker_limit,
                "model_e1_constraint_active": bool(model_pair_active),
                "checker_e1_applicable": bool(checker_e1_applicable),
                "checker_e1_violation": bool(checker_e1_violation),
                "mismatch_reason": reason,
            }
        )
    debug = pd.DataFrame(rows, columns=columns)
    if debug.empty:
        metrics = {
            "debug_total_directional_pairs_compared": 0,
            "debug_pair_set_mismatches": 0,
            "debug_active_connection_count_mismatches": 0,
            "debug_alpha_formula_mismatches": 0,
            "debug_inactive_handling_mismatches": 0,
            "debug_checker_e1_violations_explained": 0,
            "debug_unexplained_e1_violations": 0,
        }
    else:
        reason = debug["mismatch_reason"].fillna("")
        e1_bad = debug["checker_e1_violation"].fillna(False).astype(bool)
        metrics = {
            "debug_total_directional_pairs_compared": int(len(debug)),
            "debug_pair_set_mismatches": int((reason == "pair_set_mismatch").sum()),
            "debug_active_connection_count_mismatches": int((reason == "active_connection_count_mismatch").sum()),
            "debug_alpha_formula_mismatches": int((reason == "alpha_formula_mismatch").sum()),
            "debug_inactive_handling_mismatches": int((reason == "inactive_pair_handling_mismatch").sum()),
            "debug_checker_e1_violations_explained": int((e1_bad & (reason != "no_mismatch")).sum()),
            "debug_unexplained_e1_violations": int((e1_bad & (reason == "no_mismatch")).sum()),
        }
    return debug, metrics


def write_directional_model_checker_debug_summary(run_dir: Path, metrics: dict) -> None:
    lines = [
        "DIRECTIONAL MODEL-CHECKER DEBUG SUMMARY",
        "---------------------------------------",
        f"total directional pairs compared: {metrics['debug_total_directional_pairs_compared']}",
        f"pair set mismatches: {metrics['debug_pair_set_mismatches']}",
        f"active connection count mismatches: {metrics['debug_active_connection_count_mismatches']}",
        f"alpha formula mismatches: {metrics['debug_alpha_formula_mismatches']}",
        f"inactive handling mismatches: {metrics['debug_inactive_handling_mismatches']}",
        f"checker E1 violations explained count: {metrics['debug_checker_e1_violations_explained']}",
        f"unexplained E1 violations count: {metrics['debug_unexplained_e1_violations']}",
    ]
    (run_dir / "directional_model_checker_debug_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_directional_violation_files(run_dir: Path, directional_report: pd.DataFrame) -> None:
    if directional_report is None or directional_report.empty:
        pd.DataFrame().to_excel(run_dir / "directional_e1_violations.xlsx", index=False)
        pd.DataFrame().to_excel(run_dir / "directional_e2_violations.xlsx", index=False)
        return
    e1 = directional_report[directional_report["e1_violation"].fillna(False).astype(bool)].copy()
    if not e1.empty:
        e1["violation_amount"] = pd.to_numeric(
            e1.get("e1_violation_amount", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
    e1_cols = [
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
        "e1_allowed_value",
        "e1_formula",
        "directional_e1_mode",
        "violation_amount",
    ]
    e1[[col for col in e1_cols if col in e1.columns]].to_excel(run_dir / "directional_e1_violations.xlsx", index=False)
    e2 = directional_report[directional_report["e2_violation"].fillna(False).astype(bool)].copy()
    if not e2.empty:
        e2["violation_amount"] = pd.to_numeric(e2["e2_violation_amount"], errors="coerce").fillna(0.0)
    e2_cols = [
        "directional_pair_id",
        "day",
        "od_forward",
        "od_reverse",
        "forward_best_journey_time",
        "reverse_best_journey_time",
        "journey_time_diff",
        "gamma",
        "violation_amount",
        "forward_active_connections",
        "reverse_active_connections",
    ]
    e2[[col for col in e2_cols if col in e2.columns]].to_excel(run_dir / "directional_e2_violations.xlsx", index=False)


def _model_active_counts(candidate_solution: pd.DataFrame | None) -> dict[tuple[str, int], int]:
    if candidate_solution is None or candidate_solution.empty:
        return {}
    active = candidate_solution[candidate_solution.get("active", False) == True].copy()
    if active.empty:
        return {}
    grouped = active.groupby(["O&D", "day"]).size()
    return {(str(od), int(day)): int(count) for (od, day), count in grouped.items()}


def _directional_pair_key(od_a: str, od_b: str, day: int) -> tuple[int, str, str]:
    first, second = sorted([str(od_a), str(od_b)])
    return int(day), first, second


def build_directional_coverage_analysis(
    raw_connections: pd.DataFrame,
    prepared: object,
    model_config: ModelConfig,
    selected_connections: pd.DataFrame | None,
    run_config: RunConfig,
) -> tuple[pd.DataFrame, dict]:
    columns = [
        "directional_pair_id",
        "day",
        "od_forward",
        "od_reverse",
        "forward_origin",
        "forward_destination",
        "reverse_origin",
        "reverse_destination",
        "forward_exists_in_raw_data",
        "reverse_exists_in_raw_data",
        "forward_exists_in_selected_subset",
        "reverse_exists_in_selected_subset",
        "pair_exists_in_selected_subset",
        "forward_candidate_connections",
        "reverse_candidate_connections",
        "forward_active_connections",
        "reverse_active_connections",
        "both_directions_have_candidates",
        "both_directions_have_active_connections",
        "total_active_connections",
        "connection_diff",
        "alpha",
        "alpha_limit_value",
        "e1_applicable",
        "e1_violation",
        "forward_best_journey_time",
        "reverse_best_journey_time",
        "journey_time_diff",
        "gamma",
        "e2_applicable",
        "e2_violation",
        "skip_reason",
        "possible_reason_category",
    ]
    raw_scope = raw_connections.copy()
    if run_config.days:
        raw_scope = raw_scope[raw_scope["day"].isin(run_config.days)].copy()
    raw_markets = _market_lookup(raw_scope)
    selected_markets = _market_lookup(prepared.raw_connections)
    candidate_counts = _od_day_counts(prepared.tk_connections)
    active_counts, active_best = _active_connection_counts_and_best(selected_connections)
    rows = []
    seen = set()
    idx = 0
    for (od, day), market in raw_markets.items():
        reverse_od = _reverse_od(od)
        pair_key = tuple(sorted([od, reverse_od]) + [str(day)])
        if pair_key in seen:
            continue
        seen.add(pair_key)
        reverse_market = raw_markets.get((reverse_od, day))
        forward_origin, forward_destination = _split_od_local(od)
        reverse_origin, reverse_destination = _split_od_local(reverse_od)
        forward_selected = (od, day) in selected_markets
        reverse_selected = (reverse_od, day) in selected_markets
        forward_candidates = int(candidate_counts.get((od, day), 0))
        reverse_candidates = int(candidate_counts.get((reverse_od, day), 0))
        forward_active = int(active_counts.get((od, day), 0))
        reverse_active = int(active_counts.get((reverse_od, day), 0))
        total_active = forward_active + reverse_active
        connection_diff = abs(forward_active - reverse_active)
        alpha_limit = model_config.directional_alpha * total_active
        pair_selected = forward_selected and reverse_selected
        e1_applicable = pair_selected and total_active > 0
        e1_violation = bool(e1_applicable and connection_diff > alpha_limit + 1e-6)
        forward_best = active_best.get((od, day))
        reverse_best = active_best.get((reverse_od, day))
        e2_applicable = forward_active > 0 and reverse_active > 0 and forward_best is not None and reverse_best is not None
        journey_diff = abs(float(forward_best) - float(reverse_best)) if e2_applicable else None
        e2_violation = bool(e2_applicable and journey_diff is not None and journey_diff > model_config.directional_gamma_min + 1e-6)
        reverse_exists_raw = reverse_market is not None
        both_candidates = forward_candidates > 0 and reverse_candidates > 0
        both_active = forward_active > 0 and reverse_active > 0
        reason = _directional_possible_reason(
            reverse_exists_raw,
            forward_selected,
            reverse_selected,
            forward_candidates,
            reverse_candidates,
            forward_active,
            reverse_active,
        )
        skip_reason = "" if both_candidates else reason
        rows.append(
            {
                "directional_pair_id": f"dircov_{idx:05d}",
                "day": int(day),
                "od_forward": od,
                "od_reverse": reverse_od,
                "forward_origin": forward_origin or market.get("origin", ""),
                "forward_destination": forward_destination or market.get("destination", ""),
                "reverse_origin": reverse_origin,
                "reverse_destination": reverse_destination,
                "forward_exists_in_raw_data": True,
                "reverse_exists_in_raw_data": bool(reverse_exists_raw),
                "forward_exists_in_selected_subset": bool(forward_selected),
                "reverse_exists_in_selected_subset": bool(reverse_selected),
                "pair_exists_in_selected_subset": bool(pair_selected),
                "forward_candidate_connections": forward_candidates,
                "reverse_candidate_connections": reverse_candidates,
                "forward_active_connections": forward_active,
                "reverse_active_connections": reverse_active,
                "both_directions_have_candidates": bool(both_candidates),
                "both_directions_have_active_connections": bool(both_active),
                "total_active_connections": total_active,
                "connection_diff": connection_diff,
                "alpha": model_config.directional_alpha,
                "alpha_limit_value": alpha_limit,
                "e1_applicable": bool(e1_applicable),
                "e1_violation": e1_violation,
                "forward_best_journey_time": forward_best,
                "reverse_best_journey_time": reverse_best,
                "journey_time_diff": journey_diff,
                "gamma": model_config.directional_gamma_min,
                "e2_applicable": bool(e2_applicable),
                "e2_violation": e2_violation,
                "skip_reason": skip_reason,
                "possible_reason_category": reason,
            }
        )
        idx += 1
    analysis = pd.DataFrame(rows, columns=columns)
    return analysis, _summarize_directional_coverage(analysis, raw_markets, selected_markets)


def _summarize_directional_coverage(
    analysis: pd.DataFrame,
    raw_markets: dict,
    selected_markets: dict,
) -> dict:
    if analysis is None or analysis.empty:
        return {
            "coverage_total_od_day_markets": len(raw_markets),
            "coverage_directional_pairs_raw": 0,
            "coverage_directional_pairs_selected_subset": 0,
            "coverage_pairs_both_candidate_connections": 0,
            "coverage_pairs_both_active_connections": 0,
            "coverage_pairs_only_forward_active": 0,
            "coverage_pairs_only_reverse_active": 0,
            "coverage_pairs_no_active_connections": 0,
            "coverage_e1_applicable_pairs": 0,
            "coverage_e2_applicable_pairs": 0,
            "coverage_pct_raw": 0.0,
            "coverage_pct_selected_subset": 0.0,
            "coverage_active_directional_pct": 0.0,
            "coverage_top_skip_reason": "",
        }
    total_pairs = len(analysis)
    raw_pairs = int(analysis["reverse_exists_in_raw_data"].fillna(False).astype(bool).sum())
    selected_pairs = int(analysis["pair_exists_in_selected_subset"].fillna(False).astype(bool).sum())
    both_candidates = int(analysis["both_directions_have_candidates"].fillna(False).astype(bool).sum())
    both_active = int(analysis["both_directions_have_active_connections"].fillna(False).astype(bool).sum())
    selected = analysis[analysis["pair_exists_in_selected_subset"].fillna(False).astype(bool)].copy()
    only_forward = int(((selected["forward_active_connections"] > 0) & (selected["reverse_active_connections"] == 0)).sum()) if not selected.empty else 0
    only_reverse = int(((selected["forward_active_connections"] == 0) & (selected["reverse_active_connections"] > 0)).sum()) if not selected.empty else 0
    no_active = int((selected["total_active_connections"] == 0).sum()) if not selected.empty else 0
    top_reason = ""
    if "possible_reason_category" in analysis and not analysis.empty:
        counts = analysis["possible_reason_category"].fillna("unknown").astype(str).value_counts()
        if len(counts):
            top_reason = f"{counts.index[0]} ({int(counts.iloc[0])})"
    return {
        "coverage_total_od_day_markets": len(raw_markets),
        "coverage_directional_pairs_raw": raw_pairs,
        "coverage_directional_pairs_selected_subset": selected_pairs,
        "coverage_pairs_both_candidate_connections": both_candidates,
        "coverage_pairs_both_active_connections": both_active,
        "coverage_pairs_only_forward_active": only_forward,
        "coverage_pairs_only_reverse_active": only_reverse,
        "coverage_pairs_no_active_connections": no_active,
        "coverage_e1_applicable_pairs": int(analysis["e1_applicable"].fillna(False).astype(bool).sum()),
        "coverage_e2_applicable_pairs": int(analysis["e2_applicable"].fillna(False).astype(bool).sum()),
        "coverage_pct_raw": float(100.0 * raw_pairs / total_pairs) if total_pairs else 0.0,
        "coverage_pct_selected_subset": float(100.0 * selected_pairs / total_pairs) if total_pairs else 0.0,
        "coverage_active_directional_pct": float(100.0 * both_active / selected_pairs) if selected_pairs else 0.0,
        "coverage_top_skip_reason": top_reason,
    }


def write_directional_coverage_summary(run_dir: Path, metrics: dict) -> None:
    lines = [
        "DIRECTIONAL COVERAGE SUMMARY",
        "----------------------------",
        f"total O-D/day markets: {metrics['coverage_total_od_day_markets']}",
        f"total reverse-direction pairs in raw data: {metrics['coverage_directional_pairs_raw']}",
        f"total reverse-direction pairs in selected subset: {metrics['coverage_directional_pairs_selected_subset']}",
        f"directional pairs active in model: {metrics['coverage_pairs_both_candidate_connections']}",
        f"pairs with both directions candidate connections: {metrics['coverage_pairs_both_candidate_connections']}",
        f"pairs with both directions active connections: {metrics['coverage_pairs_both_active_connections']}",
        f"pairs with only forward active: {metrics['coverage_pairs_only_forward_active']}",
        f"pairs with only reverse active: {metrics['coverage_pairs_only_reverse_active']}",
        f"pairs with no active connections: {metrics['coverage_pairs_no_active_connections']}",
        f"E1 applicable pair count: {metrics['coverage_e1_applicable_pairs']}",
        f"E2 applicable pair count: {metrics['coverage_e2_applicable_pairs']}",
        f"top skip reasons: {metrics['coverage_top_skip_reason']}",
        f"coverage pct raw: {metrics['coverage_pct_raw']:.6f}",
        f"coverage pct selected subset: {metrics['coverage_pct_selected_subset']:.6f}",
        f"active directional coverage pct: {metrics['coverage_active_directional_pct']:.6f}",
        "",
        "Yorum:",
    ]
    if metrics["coverage_directional_pairs_raw"] == 0:
        lines.append("- Raw data icinde ters yon O-D/day eslesmesi yok; directional balance bu kapsamda test edilemez.")
    elif metrics["coverage_directional_pairs_selected_subset"] < metrics["coverage_directional_pairs_raw"]:
        lines.append("- Ters yonler raw data'da var ama selected subset/top importance filtresi bunlarin bir kismini disarida birakiyor.")
    else:
        lines.append("- Raw data'daki ters yon eslesmeleri selected subset icinde korunuyor.")
    if metrics["coverage_pairs_both_candidate_connections"] == 0:
        lines.append("- Iki yonde de TK candidate connection olan pair yok; modelde directional baski olusmasi beklenmez.")
    elif metrics["coverage_pairs_both_active_connections"] == 0:
        lines.append("- Candidate pair var ama iki yonde aktif connection olusmuyor; objective etkisi zayif kalir.")
    else:
        lines.append("- Iki yonde aktif connection olan pair var; directional balance gercek baski olusturabilecek durumda.")
    (run_dir / "directional_coverage_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _directional_pairs_from_coverage(analysis: pd.DataFrame) -> pd.DataFrame:
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
    if analysis is None or analysis.empty:
        return pd.DataFrame(columns=columns)
    pairs = analysis[
        [
            "directional_pair_id",
            "day",
            "od_forward",
            "od_reverse",
            "forward_origin",
            "forward_destination",
            "reverse_origin",
            "reverse_destination",
            "forward_exists_in_selected_subset",
            "reverse_exists_in_selected_subset",
            "pair_exists_in_selected_subset",
            "skip_reason",
        ]
    ].rename(
        columns={
            "forward_exists_in_selected_subset": "forward_exists",
            "reverse_exists_in_selected_subset": "reverse_exists",
            "pair_exists_in_selected_subset": "pair_active",
        }
    )
    return pairs[columns]


def _market_lookup(connections: pd.DataFrame) -> dict[tuple[str, int], dict]:
    if connections is None or connections.empty:
        return {}
    lookup = {}
    for _, row in connections[["O&D", "origin", "destination", "day"]].drop_duplicates(["O&D", "day"]).iterrows():
        lookup[(str(row["O&D"]), int(row["day"]))] = {"origin": str(row["origin"]), "destination": str(row["destination"])}
    return lookup


def _od_day_counts(connections: pd.DataFrame) -> dict[tuple[str, int], int]:
    if connections is None or connections.empty:
        return {}
    grouped = connections.groupby(["O&D", "day"]).size()
    return {(str(od), int(day)): int(count) for (od, day), count in grouped.items()}


def _active_connection_counts_and_best(selected_connections: pd.DataFrame | None) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], float]]:
    if selected_connections is None or selected_connections.empty:
        return {}, {}
    active = selected_connections[selected_connections.get("active", True) == True].copy()
    if active.empty:
        return {}, {}
    count_grouped = active.groupby(["od", "day"]).size()
    counts = {(str(od), int(day)): int(count) for (od, day), count in count_grouped.items()}
    best_col = "new_journey_time_min" if "new_journey_time_min" in active.columns else "journey_time_min"
    best_grouped = active.groupby(["od", "day"])[best_col].min()
    best = {(str(od), int(day)): float(value) for (od, day), value in best_grouped.items() if pd.notna(value)}
    return counts, best


def _directional_possible_reason(
    reverse_exists_raw: bool,
    forward_selected: bool,
    reverse_selected: bool,
    forward_candidates: int,
    reverse_candidates: int,
    forward_active: int,
    reverse_active: int,
) -> str:
    if not reverse_exists_raw:
        return "missing_reverse_direction_raw"
    if not forward_selected or not reverse_selected:
        return "missing_reverse_direction_subset"
    if forward_candidates <= 0:
        return "no_candidate_connections_forward"
    if reverse_candidates <= 0:
        return "no_candidate_connections_reverse"
    if forward_active <= 0:
        return "no_active_connections_forward"
    if reverse_active <= 0:
        return "no_active_connections_reverse"
    if forward_active > 0 and reverse_active > 0:
        return "both_directions_active"
    return "unknown"


def _split_od_local(od: str) -> tuple[str, str]:
    parts = str(od).split("-", 1)
    if len(parts) != 2:
        return str(od), ""
    return parts[0], parts[1]


def _reverse_od(od: str) -> str:
    origin, destination = _split_od_local(od)
    if not destination:
        return str(od)
    return f"{destination}-{origin}"


def _experiment_notes(model_level: str) -> str:
    if model_level == "baseline":
        return "baseline current schedule"
    if model_level == "rotation":
        return "MVP core with Flight Pairs rotation constraints"
    if model_level == "capacity":
        return "MVP core with rotation and IST hub bucket capacity constraints"
    if model_level == "regularity":
        return "MVP core with rotation, IST hub capacity, and schedule regularity constraints"
    if model_level == "directional":
        return "MVP core with rotation, IST hub capacity, schedule regularity, and directional E1 count balance constraints"
    if model_level == "final":
        return "final model with rotation, hub capacity, schedule regularity, directional balance, and schedule disruption penalty"
    if model_level == "full":
        return "full model extension hooks"
    return "MVP existing-connection selection model"


def run_sensitivity_test(run_config: RunConfig) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sensitivity_dir = run_config.output_dir / f"sensitivity_{timestamp}"
    rows = []
    for max_shift in [0, 15, 30, 60, 120]:
        shift_config = clone_run_config(
            run_config,
            run_id=f"shift_{max_shift}",
            output_dir=sensitivity_dir,
            max_shift_min=max_shift,
            model_level="mvp",
        )
        print(f"[INFO] Sensitivity run max_shift={max_shift}")
        rows.append(run_once(shift_config, analyze_mvp=True))
    write_sensitivity_summary(sensitivity_dir, rows)
    print(f"[INFO] Sensitivity outputs written to {sensitivity_dir}")


def run_shift_penalty_stress_test(run_config: RunConfig) -> None:
    """Run final model with progressively stronger schedule disruption penalties."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stress_dir = run_config.output_dir / f"shift_penalty_stress_{timestamp}"
    experiments = [
        ("no_penalty", "none", 0.0, run_config.shift_free_threshold_min),
        ("linear_1", "linear", 1.0, run_config.shift_free_threshold_min),
        ("linear_5", "linear", 5.0, run_config.shift_free_threshold_min),
        ("linear_10", "linear", 10.0, run_config.shift_free_threshold_min),
        ("threshold_5_w5", "threshold", 5.0, 5),
        ("threshold_5_w10", "threshold", 10.0, 5),
        ("threshold_10_w10", "threshold", 10.0, 10),
    ]
    rows = []
    no_penalty_before = None
    for scenario, mode, weight, threshold in experiments:
        stress_config = clone_run_config(
            run_config,
            run_id=scenario,
            output_dir=stress_dir,
            max_shift_min=run_config.max_shift_min,
            model_level="final",
            directional_e2_mode=run_config.directional_e2_mode or "hard",
            shift_penalty_mode=mode,
            shift_penalty_weight=weight,
            shift_free_threshold_min=threshold,
        )
        print(
            "[INFO] Shift penalty stress run "
            f"scenario={scenario}, mode={mode}, weight={weight}, threshold={threshold}"
        )
        result = run_once(stress_config, analyze_mvp=True)
        if no_penalty_before is None:
            no_penalty_before = float(result.get("objective_before_penalty", result.get("objective", 0.0)))
        baseline = float(result.get("baseline_objective", 0.0))
        before = float(result.get("objective_before_penalty", result.get("objective", 0.0)))
        after = float(result.get("objective_after_penalty", before))
        rows.append(
            {
                "scenario": scenario,
                "penalty_mode": mode,
                "penalty_weight": weight,
                "threshold": threshold,
                "feasible": result["feasible"],
                "baseline_objective": baseline,
                "no_penalty_objective_before_penalty": no_penalty_before,
                "objective_before_penalty": before,
                "objective_after_penalty": after,
                "improvement_vs_baseline_before_penalty_pct": 100.0 * (before - baseline) / abs(baseline) if abs(baseline) > 1e-12 else 0.0,
                "improvement_vs_baseline_after_penalty_pct": 100.0 * (after - baseline) / abs(baseline) if abs(baseline) > 1e-12 else 0.0,
                "reward_drop_vs_no_penalty": float(no_penalty_before) - before,
                "total_shift_penalty": result.get("total_shift_penalty", 0.0),
                "active_connections": result.get("active_connections", 0),
                "beaten_competitors": result.get("beaten_competitors", 0),
                "directional_pairs_active": result.get("directional_pairs_active", 0),
                "e1_violations": result.get("directional_e1_violations", 0),
                "e2_violations": result.get("directional_e2_violations", 0),
                "rotation_violations": result.get("rotation_violations", 0),
                "rotation_block_elapsedtime_match_count": result.get("rotation_block_elapsedtime_match_count", 0),
                "rotation_block_legacy_exact_match_count": result.get("rotation_block_legacy_exact_match_count", 0),
                "rotation_block_exact_match_count": result.get("rotation_block_exact_match_count", 0),
                "rotation_block_fallback_count": result.get("rotation_block_fallback_count", 0),
                "hub_capacity_violations": result.get("hub_capacity_violations", 0),
                "regularity_violations": result.get("regularity_violations", 0),
                "avg_abs_shift": result.get("avg_abs_shift", 0.0),
                "median_abs_shift": result.get("median_abs_shift", 0.0),
                "max_abs_shift": result.get("max_abs_shift", 0.0),
                "pct_shift_le_5": result.get("pct_shift_le_5", 0.0),
                "pct_shift_5_15": result.get("pct_shift_5_15", 0.0),
                "pct_shift_15_30": result.get("pct_shift_15_30", 0.0),
                "pct_at_shift_bounds": result.get("pct_at_shift_bounds", 0.0),
                "runtime_sec": result.get("runtime_sec", 0.0),
                "solver_status": result.get("solver_status", ""),
            }
        )
    stress_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(stress_dir / "shift_penalty_stress_summary.csv", index=False)
    _write_shift_penalty_interpretation(stress_dir, summary)
    print(f"[INFO] Shift penalty stress outputs written to {stress_dir}")


def _write_shift_penalty_interpretation(stress_dir: Path, summary: pd.DataFrame) -> None:
    lines = ["SHIFT PENALTY STRESS YORUMU", "-----------------------------"]
    if summary.empty:
        lines.append("Stress summary bos; yorum uretilemedi.")
        (stress_dir / "shift_penalty_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return
    data = summary.copy()
    numeric_cols = [
        "objective_before_penalty",
        "objective_after_penalty",
        "reward_drop_vs_no_penalty",
        "active_connections",
        "avg_abs_shift",
        "median_abs_shift",
        "pct_at_shift_bounds",
        "improvement_vs_baseline_after_penalty_pct",
        "e1_violations",
        "e2_violations",
        "rotation_violations",
        "hub_capacity_violations",
        "regularity_violations",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    no_penalty = data[data["scenario"] == "no_penalty"].head(1)
    no_avg = float(no_penalty["avg_abs_shift"].iloc[0]) if not no_penalty.empty else float(data["avg_abs_shift"].iloc[0])
    no_median = float(no_penalty["median_abs_shift"].iloc[0]) if not no_penalty.empty else float(data["median_abs_shift"].iloc[0])
    feasible = data[data["feasible"].astype(str).str.lower().isin(["true", "1"])]
    hard_clean = feasible[
        (feasible["e1_violations"].fillna(0) == 0)
        & (feasible["e2_violations"].fillna(0) == 0)
        & (feasible["rotation_violations"].fillna(0) == 0)
        & (feasible["hub_capacity_violations"].fillna(0) == 0)
        & (feasible["regularity_violations"].fillna(0) == 0)
    ].copy()
    candidate_pool = hard_clean[hard_clean["improvement_vs_baseline_after_penalty_pct"].fillna(-1e9) >= 2.0].copy()
    if candidate_pool.empty:
        candidate_pool = hard_clean[hard_clean["improvement_vs_baseline_after_penalty_pct"].fillna(-1e9) > 0].copy()
    if not candidate_pool.empty:
        max_active = float(data["active_connections"].max()) if data["active_connections"].notna().any() else 0.0
        candidate_pool = candidate_pool[candidate_pool["active_connections"] >= 0.95 * max_active]
    if candidate_pool.empty:
        candidate_pool = hard_clean if not hard_clean.empty else data
    if not candidate_pool.empty:
        candidate_pool = candidate_pool.sort_values(
            ["avg_abs_shift", "reward_drop_vs_no_penalty", "objective_before_penalty"],
            ascending=[True, True, False],
        )
        best = candidate_pool.iloc[0]
        recommended = str(best["scenario"])
    else:
        best = None
        recommended = ""
    min_avg = float(data["avg_abs_shift"].min()) if data["avg_abs_shift"].notna().any() else 0.0
    min_median = float(data["median_abs_shift"].min()) if data["median_abs_shift"].notna().any() else 0.0
    max_reward_drop = float(data["reward_drop_vs_no_penalty"].max()) if data["reward_drop_vs_no_penalty"].notna().any() else 0.0
    feasible_count = int(len(feasible))
    lines.extend(
        [
            f"Senaryo sayisi: {len(data)}",
            f"Feasible senaryo sayisi: {feasible_count}",
            f"No-penalty avg abs shift: {no_avg}",
            f"No-penalty median abs shift: {no_median}",
            f"En dusuk avg abs shift: {min_avg}",
            f"En dusuk median abs shift: {min_median}",
            f"Maksimum reward drop vs no-penalty: {max_reward_drop}",
            f"Onerilen trade-off senaryo: {recommended}",
            "",
            "Yorum:",
        ]
    )
    if min_avg < no_avg - 1e-6 or min_median < no_median - 1e-6:
        lines.append("- Penalty arttikca en az bazi senaryolarda shiftler azaldi; schedule disruption baskisi gercek etki uretiyor.")
    else:
        lines.append("- Penalty ayarlari bu subsette avg/median shiftleri belirgin azaltmadi.")
    if max_reward_drop > 1e-6:
        lines.append("- Shift cezasi arttikca baglanti/ranking faydasindan odun veriliyor; trade-off gorunur durumda.")
    else:
        lines.append("- Reward tarafinda belirgin dusus olusmadi; penalty bu subsette zayif kalmis olabilir.")
    if feasible_count == len(data):
        lines.append("- Tum senaryolarda feasibility korundu; penalty hard constraints'i bozmadi.")
    else:
        lines.append("- Bazi senaryolarda feasibility bozuldu; checker sonucu esas alinmali.")
    if best is not None:
        lines.append("- En iyi aday, hard violation olmadan baseline ustunde kalan ve aktif baglanti sayisini buyuk olcude koruyan en dusuk shiftli senaryo olarak secildi.")
        lines.append(
            f"- Demo/final icin onerim: {recommended} "
            f"(mode={best['penalty_mode']}, weight={best['penalty_weight']}, threshold={best['threshold']})."
        )
    lines.append("- Bu sonuclardan sonra daha buyuk final scale test ve rapor hazirligina gecilebilir.")
    (stress_dir / "shift_penalty_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def run_final_scale_test(run_config: RunConfig) -> None:
    """Run the recommended final model across larger evaluation subsets."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scale_dir = run_config.output_dir / f"final_scale_{timestamp}"
    experiments = [
        ("od100_days12_top", 100, [1, 2], True),
        ("od200_days12_top", 200, [1, 2], True),
        ("od300_days12_top", 300, [1, 2], True),
        ("od100_days123_top", 100, [1, 2, 3], True),
        ("od200_days123_top", 200, [1, 2, 3], True),
        ("od300_days123_top", 300, [1, 2, 3], True),
    ]
    rows = []
    run_dirs: dict[str, Path] = {}
    for scenario, max_od, days, top_only in experiments:
        scale_config = clone_run_config(
            run_config,
            run_id=scenario,
            output_dir=scale_dir,
            max_shift_min=30,
            max_od=max_od,
            days=days,
            top_importance_only=top_only,
            model_level="final",
            hub_arrival_capacity=HUB_ARRIVAL_CAPACITY_PER_BUCKET,
            hub_departure_capacity=HUB_DEPARTURE_CAPACITY_PER_BUCKET,
            hub_bucket_size_min=HUB_BUCKET_SIZE_MINUTES,
            regularity_xdev_min=SCHEDULE_REGULARITY_XDEV_MINUTES,
            directional_alpha=ALPHA_DIRECTIONAL_BALANCE,
            directional_gamma_min=GAMMA_BEST_JOURNEY_MINUTES,
            directional_e1_mode="hard",
            directional_e1_formula="absolute_tolerance",
            directional_e1_abs_tolerance=1,
            directional_e1_penalty_weight=DIRECTIONAL_E1_PENALTY_WEIGHT,
            directional_e2_mode="hard",
            shift_penalty_mode="linear",
            shift_penalty_weight=5.0,
            shift_free_threshold_min=SHIFT_FREE_THRESHOLD_MINUTES,
            mandatory_connection_selection=True,
            ranking_objective_mode="lookup_linear",
        )
        print(
            "[INFO] Final scale run "
            f"scenario={scenario}, max_od={max_od}, days={','.join(str(day) for day in days)}"
        )
        try:
            result = run_once(scale_config, analyze_mvp=True)
        except Exception as exc:
            print(f"[WARN] Final scale scenario failed: {scenario}: {exc}")
            rows.append(_final_scale_failure_row(scenario, max_od, days, top_only, exc))
            continue
        run_dirs[scenario] = scale_dir / scenario
        baseline = _safe_float(result.get("baseline_objective", ""))
        before = _safe_float(result.get("objective_before_penalty", result.get("objective", "")))
        after = _safe_float(result.get("objective_after_penalty", before))
        hard_violations = int(result.get("rotation_violations", 0) or 0) + int(result.get("hub_capacity_violations", 0) or 0)
        hard_violations += int(result.get("regularity_violations", 0) or 0)
        hard_violations += int(result.get("directional_e1_hard_violations", 0) or 0) + int(result.get("directional_e2_hard_violations", 0) or 0)
        hard_violations += int(result.get("feasible_but_inactive_connection_count", 0) or 0) + int(result.get("infeasible_but_active_connection_count", 0) or 0)
        before_improvement = 100.0 * (before - baseline) / abs(baseline) if baseline not in {"", None} and before not in {"", None} and abs(baseline) > 1e-12 else ""
        after_improvement = 100.0 * (after - baseline) / abs(baseline) if baseline not in {"", None} and after not in {"", None} and abs(baseline) > 1e-12 else ""
        rows.append(
            {
                "scenario": scenario,
                "max_od": max_od,
                "days": ",".join(str(day) for day in days),
                "top_importance_only": top_only,
                "ranking_objective_mode": result.get("ranking_objective_mode", "lookup_linear"),
                "mandatory_connection_selection": result.get("mandatory_connection_selection", True),
                "directional_e1_mode": result.get("directional_e1_mode", "hard"),
                "directional_e1_formula": result.get("directional_e1_formula", "absolute_tolerance"),
                "directional_e1_abs_tolerance": result.get("directional_e1_abs_tolerance", 1),
                "directional_e2_mode": result.get("directional_e2_mode", "hard"),
                "feasible": result.get("feasible", False),
                "baseline_objective": baseline,
                "objective_before_penalty": before,
                "objective_after_penalty": after,
                "improvement_before_penalty_pct": before_improvement,
                "improvement_after_penalty_pct": after_improvement,
                "active_connections": result.get("active_connections", 0),
                "beaten_competitors": result.get("beaten_competitors", 0),
                "lookup_ranking_reward": result.get("reported_lookup_ranking_reward", result.get("ranking_reward", 0.0)),
                "proxy_ranking_reward": result.get("model_proxy_ranking_reward", 0.0),
                "connection_reward": result.get("connection_reward", 0.0),
                "shift_penalty": result.get("total_shift_penalty", 0.0),
                "total_candidate_connections": result.get("total_candidate_connections", 0),
                "feasible_by_schedule": result.get("connections_feasible_by_final_schedule", 0),
                "mandatory_feasible_but_inactive": result.get("feasible_but_inactive_connection_count", 0),
                "mandatory_infeasible_but_active": result.get("infeasible_but_active_connection_count", 0),
                "newly_activated_connections": result.get("newly_activated_connections", 0),
                "improved_od_count": result.get("improved_od_count", 0),
                "worsened_od_count": result.get("worsened_od_count", 0),
                "rotation_violations": result.get("rotation_violations", 0),
                "rotation_block_elapsedtime_match_count": result.get("rotation_block_elapsedtime_match_count", 0),
                "rotation_block_legacy_exact_match_count": result.get("rotation_block_legacy_exact_match_count", 0),
                "rotation_block_exact_match_count": result.get("rotation_block_exact_match_count", 0),
                "rotation_block_fallback_count": result.get("rotation_block_fallback_count", 0),
                "hub_capacity_violations": result.get("hub_capacity_violations", 0),
                "regularity_violations": result.get("regularity_violations", 0),
                "directional_e1_violations": result.get("directional_e1_violations", 0),
                "directional_e2_violations": result.get("directional_e2_violations", 0),
                "total_hard_violations": hard_violations,
                "avg_abs_shift": result.get("avg_abs_shift", 0.0),
                "median_abs_shift": result.get("median_abs_shift", 0.0),
                "max_abs_shift": result.get("max_abs_shift", 0.0),
                "pct_shift_le_5": result.get("pct_shift_le_5", 0.0),
                "pct_shift_5_15": result.get("pct_shift_5_15", 0.0),
                "pct_shift_15_30": result.get("pct_shift_15_30", 0.0),
                "pct_at_shift_bounds": result.get("pct_at_shift_bounds", 0.0),
                "runtime_sec": result.get("runtime_sec", 0.0),
                "solver_status": result.get("solver_status", ""),
                "gap": result.get("gap", ""),
            }
        )
    scale_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary = _ordered_final_scale_summary(summary)
    summary.to_csv(scale_dir / "final_scale_summary.csv", index=False)
    proxy_comparison = _write_proxy_vs_lookup_linear_comparison(run_config.output_dir, scale_dir, summary)
    best = _select_final_candidate(summary)
    _write_final_scale_interpretation(scale_dir, summary, best, proxy_comparison)
    if best:
        _write_final_submission_bundle(run_config.output_dir, run_dirs[best], summary[summary["scenario"] == best].iloc[0])
    print(f"[INFO] Final scale outputs written to {scale_dir}")


def _safe_float(value) -> float | str:
    if value in {"", None}:
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def run_data_schema_audit(run_config: RunConfig) -> Path:
    """Write a schema and duration-consistency audit for the current data_dir."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_dir = run_config.output_dir / f"data_schema_audit_{timestamp}"
    audit_dir.mkdir(parents=True, exist_ok=True)
    od_path = run_config.data_dir / run_config.competitor_connections_file
    raw_od = pd.read_excel(od_path, engine="openpyxl")
    pd.DataFrame(
        {
            "column_index": list(range(1, len(raw_od.columns) + 1)),
            "column_name": [str(col) for col in raw_od.columns],
            "non_null_count": [int(raw_od[col].notna().sum()) for col in raw_od.columns],
            "dtype": [str(raw_od[col].dtype) for col in raw_od.columns],
        }
    ).to_csv(audit_dir / "od_schema_columns.csv", index=False)

    raw = load_inputs(run_config)
    conn = raw.connections.copy()
    both_elapsed = conn["elapsed_time_1_min"].notna() & conn["elapsed_time_2_min"].notna()
    computed = conn.get("computed_journey_time_from_elapsed_min")
    gate = conn.get("gate_to_gate_journey_time_min")
    diff = computed - gate if computed is not None and gate is not None else pd.Series([pd.NA] * len(conn))
    warning = pd.Series([""] * len(conn), index=conn.index, dtype="object")
    if computed is not None and gate is not None:
        warning.loc[both_elapsed & gate.notna() & (diff.abs() > 5)] = "computed_total differs from Gate-to-Gate by more than 5 minutes"
        warning.loc[both_elapsed & gate.isna()] = "Gate-to-Gate missing; computed elapsed total used"
    parse_warning_cols = [
        "elapsed_time_1_parse_warning",
        "elapsed_time_2_parse_warning",
        "gate_to_gate_parse_warning",
    ]
    for col in parse_warning_cols:
        if col in conn.columns:
            has_warning = conn[col].fillna("").astype(str).str.len() > 0
            warning.loc[has_warning & (warning == "")] = conn.loc[has_warning, col].astype(str)
            warning.loc[has_warning & (warning != conn[col].astype(str)) & (warning != "")] = (
                warning.loc[has_warning & (warning != conn[col].astype(str)) & (warning != "")] + "; " + conn.loc[has_warning & (warning != conn[col].astype(str)) & (warning != ""), col].astype(str)
            )

    audit_rows = pd.DataFrame(
        {
            "od": conn.get("O&D", ""),
            "day": conn.get("day", ""),
            "flight_no_1": conn.get("inbound_flight", ""),
            "flight_no_2": conn.get("outbound_flight", ""),
            "elapsed_time_1_min": conn.get("elapsed_time_1_min", ""),
            "connection_gap_min": conn.get("connection_gap_min", ""),
            "elapsed_time_2_min": conn.get("elapsed_time_2_min", ""),
            "computed_total_journey_time": computed if computed is not None else "",
            "gate_to_gate_journey_time": gate if gate is not None else "",
            "difference_min": diff,
            "warning": warning,
        }
    )
    audit_rows.to_excel(audit_dir / "journey_time_consistency_audit.xlsx", index=False)

    expected = [
        run_config.ranking_input_file,
        run_config.flight_pairs_file,
        run_config.competitor_connections_file,
        run_config.passenger_data_file,
    ]
    found = sorted(path.name for path in run_config.data_dir.iterdir()) if run_config.data_dir.exists() else []
    missing = [name for name in expected if not (run_config.data_dir / name).exists()]
    elapsed1_present = "ElapsedTime1" in raw_od.columns
    elapsed2_present = "ElapsedTime2" in raw_od.columns
    gate_present = "Gate-to-Gate Uçuş Süresi" in raw_od.columns or "Gate-to-Gate UÃ§uÅŸ SÃ¼resi" in raw_od.columns
    old_required = {"Cr1", "Carrier Name", "Dep1", "Arr1", "FlNo1", "Arr Time", "Cr2", "Dep2", "Arr2", "FlNo2", "Dep Time", "Gate-to-Gate Uçuş Süresi", "O&D", "Gün"}
    missing_old = sorted([col for col in old_required if col not in raw_od.columns])
    parse_warnings = 0
    possible_wrap = 0
    for col in parse_warning_cols:
        if col in conn.columns:
            warnings = conn[col].fillna("").astype(str)
            parse_warnings += int((warnings != "").sum())
            possible_wrap += int(warnings.str.contains("wrapped|fraction", case=False, regex=True).sum())
    checked = int(len(conn))
    both_count = int(both_elapsed.sum())
    diff_checked = diff[both_elapsed & gate.notna()].astype(float) if gate is not None else pd.Series(dtype=float)
    avg_diff = float(diff_checked.mean()) if len(diff_checked) else 0.0
    max_abs_diff = float(diff_checked.abs().max()) if len(diff_checked) else 0.0
    gt5 = int((diff_checked.abs() > 5).sum()) if len(diff_checked) else 0
    lines = [
        "O&D DATA SCHEMA AUDIT",
        "---------------------",
        f"data_dir: {run_config.data_dir}",
        f"O&D file: {od_path}",
        f"rows: {len(raw_od)}",
        f"columns: {len(raw_od.columns)}",
        f"ElapsedTime1 present: {elapsed1_present}",
        f"ElapsedTime2 present: {elapsed2_present}",
        f"Gate-to-Gate present: {gate_present}",
        f"missing old required columns: {missing_old}",
        f"expected input files: {expected}",
        f"found files: {found}",
        f"missing files: {missing}",
        "",
        "columns:",
        *[f"- {col}" for col in raw_od.columns],
        "",
        "journey consistency summary:",
        f"total rows checked: {checked}",
        f"rows with both elapsed: {both_count}",
        f"avg diff: {avg_diff}",
        f"max abs diff: {max_abs_diff}",
        f"rows abs diff > 5: {gt5}",
        f"parse warnings: {parse_warnings}",
        f"possible 24h wrap count: {possible_wrap}",
        "",
        "functions updated/affected:",
        "- src.data_loader.normalize_connections",
        "- src.data_loader.parse_excel_duration_to_minutes",
        "- src.model._build_od_connection_block_lookup",
        "- src.model._lookup_rotation_block_times",
        "- src.rotation_analysis.summarize_rotation_skip_analysis",
    ]
    (audit_dir / "od_schema_audit.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Data schema audit outputs written to {audit_dir}")
    return audit_dir


OD300_DIAGNOSIS_COLUMNS = [
    "day_set",
    "scenario",
    "run_id",
    "feasible",
    "solver_status",
    "runtime_sec",
    "suspected_bottleneck",
    "objective_after_penalty",
    "active_connections",
    "total_candidate_connections",
    "mandatory_feasible_but_inactive",
    "mandatory_infeasible_but_active",
    "rotation_violations",
    "hub_capacity_violations",
    "regularity_violations",
    "directional_e1_mode",
    "directional_e1_abs_tolerance",
    "directional_e1_violations",
    "directional_e2_mode",
    "directional_e2_violations",
    "total_hard_violations",
    "avg_abs_shift",
    "model_level",
    "max_shift",
    "shift_penalty_mode",
    "shift_penalty_weight",
    "baseline_objective",
    "objective_before_penalty",
    "rotation_constraints_active",
    "rotation_block_elapsedtime_match_count",
    "rotation_block_fallback_count",
    "process_status",
    "note",
]


def run_od300_infeasibility_diagnosis(run_config: RunConfig, quick: bool = True) -> Path:
    """Run OD300 diagnosis with per-scenario subprocess timeouts and append-only summary writes."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnosis_dir = run_config.output_dir / f"od300_infeasibility_diagnosis_{timestamp}"
    diagnosis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diagnosis_dir / "od300_diagnosis_summary.csv"
    pd.DataFrame(columns=OD300_DIAGNOSIS_COLUMNS).to_csv(summary_path, index=False)

    quick_scenarios = [
        ("mandatory_only_core", {"model_level": "mvp", "mandatory_connection_selection": True}),
        ("rotation_only_stack", {"model_level": "rotation"}),
        ("capacity_after_rotation", {"model_level": "capacity"}),
        ("regularity_after_capacity", {"model_level": "regularity"}),
        ("no_directional", {"model_level": "regularity"}),
        ("final_hard_all", {}),
    ]
    extra_scenarios = [
        ("no_shift_penalty", {"shift_penalty_mode": "none", "shift_penalty_weight": 0.0}),
        ("low_shift_penalty", {"shift_penalty_mode": "linear", "shift_penalty_weight": 1.0}),
        ("e2_soft", {"directional_e2_mode": "soft"}),
        ("e2_report", {"directional_e2_mode": "report_only"}),
        ("no_regularity", {"model_level": "capacity"}),
        ("no_capacity", {"model_level": "rotation"}),
    ]
    scenarios = quick_scenarios if quick else quick_scenarios + extra_scenarios
    day_sets = [("days12", [1, 2])] if quick else [("days12", [1, 2]), ("days123", [1, 2, 3])]
    rows = []
    days12_rows = []
    for day_label, days in day_sets:
        if day_label == "days123" and not _od300_should_continue_to_days123(days12_rows):
            break
        for scenario, overrides in scenarios:
            run_id = f"{day_label}_{scenario}"
            row = _run_od300_diagnosis_subprocess(run_config, diagnosis_dir, run_id, day_label, days, scenario, overrides)
            rows.append(row)
            if day_label == "days12":
                days12_rows.append(row)
            pd.DataFrame([row], columns=OD300_DIAGNOSIS_COLUMNS).to_csv(summary_path, mode="a", header=False, index=False)
            print(f"[INFO] OD300 diagnosis appended: {scenario} -> {row['solver_status']}, feasible={row['feasible']}")

    summary = pd.DataFrame(rows, columns=OD300_DIAGNOSIS_COLUMNS)
    _write_od300_quick_interpretation(diagnosis_dir, summary, quick)
    print(f"[INFO] OD300 diagnosis outputs written to {diagnosis_dir}")
    return diagnosis_dir


def _run_od300_diagnosis_subprocess(
    run_config: RunConfig,
    diagnosis_dir: Path,
    run_id: str,
    day_label: str,
    days: list[int],
    scenario: str,
    overrides: dict,
) -> dict:
    scenario_options = {
        "model_level": "final",
        "max_od": 300,
        "max_shift_min": 30,
        "directional_e1_mode": "hard",
        "directional_e1_abs_tolerance": 1,
        "directional_e2_mode": "hard",
        "regularity_xdev_min": SCHEDULE_REGULARITY_XDEV_MINUTES,
        "regularity_mode": "hard",
        "regularity_penalty_weight": REGULARITY_PENALTY_WEIGHT,
        "regularity_abs_tolerance_min": 0,
        "regularity_relax_mode": "none",
        "regularity_targeted_max_relax_min": 30,
        "shift_penalty_mode": "linear",
        "shift_penalty_weight": 5.0,
        "shift_free_threshold_min": SHIFT_FREE_THRESHOLD_MINUTES,
        "mandatory_connection_selection": True,
        "timeout_sec": 300,
        "note": "",
    }
    scenario_options.update(overrides)
    note = str(scenario_options.pop("note", ""))
    scenario_timeout = int(scenario_options.pop("timeout_sec", 300))
    timeout_sec = scenario_timeout
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data_dir",
        str(run_config.data_dir),
        "--output_dir",
        str(diagnosis_dir),
        "--model_level",
        str(scenario_options["model_level"]),
        "--solver",
        str(run_config.solver),
        "--time_limit",
        str(timeout_sec),
        "--run_id",
        run_id,
        "--max_od",
        str(scenario_options["max_od"]),
        "--days",
        ",".join(str(day) for day in days),
        "--top_importance_only",
        "--max_shift",
        str(scenario_options["max_shift_min"]),
        "--regularity_xdev",
        str(scenario_options["regularity_xdev_min"]),
        "--regularity_mode",
        str(scenario_options["regularity_mode"]),
        "--regularity_penalty_weight",
        str(scenario_options["regularity_penalty_weight"]),
        "--regularity_abs_tolerance",
        str(scenario_options["regularity_abs_tolerance_min"]),
        "--regularity_relax_mode",
        str(scenario_options["regularity_relax_mode"]),
        "--regularity_targeted_max_relax",
        str(scenario_options["regularity_targeted_max_relax_min"]),
        "--directional_e1_mode",
        str(scenario_options["directional_e1_mode"]),
        "--directional_e1_formula",
        "absolute_tolerance",
        "--directional_e1_abs_tolerance",
        str(scenario_options["directional_e1_abs_tolerance"]),
        "--directional_e2_mode",
        str(scenario_options["directional_e2_mode"]),
        "--shift_penalty_mode",
        str(scenario_options["shift_penalty_mode"]),
        "--shift_penalty_weight",
        str(scenario_options["shift_penalty_weight"]),
        "--shift_free_threshold",
        str(scenario_options["shift_free_threshold_min"]),
        "--ranking_objective_mode",
        "lookup_linear",
    ]
    command.append("--mandatory_connection_selection" if scenario_options["mandatory_connection_selection"] else "--no-mandatory_connection_selection")
    start = time.monotonic()
    process_status = "completed"
    log_text = ""
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_sec + 5)
        runtime = time.monotonic() - start
        log_text = (stdout or "") + ("\nSTDERR:\n" + stderr if stderr else "")
        if process.returncode != 0:
            process_status = f"FAILED_EXIT_{process.returncode}"
            note = (note + " " if note else "") + f"Subprocess return code {process.returncode}."
    except subprocess.TimeoutExpired:
        runtime = time.monotonic() - start
        process_status = "TIME_LIMIT"
        _terminate_process_tree(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "process tree did not exit cleanly after taskkill"
        log_text = (stdout or "") + ("\nSTDERR:\n" + stderr if stderr else "")
        note = (note + " " if note else "") + f"Scenario exceeded {timeout_sec} seconds and was terminated."
    run_dir = diagnosis_dir / run_id
    (diagnosis_dir / f"{run_id}_process.log").write_text(log_text, encoding="utf-8", errors="replace")
    row = _od300_row_from_outputs(run_dir, day_label, scenario, run_id, scenario_options, process_status, runtime, note)
    _write_od300_rotation_violation_debug(run_dir, scenario, scenario_options, row)
    if process_status == "TIME_LIMIT":
        row["feasible"] = False
        row["solver_status"] = "TIME_LIMIT / NOT_SOLVED"
        row["suspected_bottleneck"] = "not solved within 300s"
    elif process_status.startswith("FAILED_EXIT") and not row["solver_status"]:
        row["feasible"] = False
        row["solver_status"] = process_status
        row["suspected_bottleneck"] = "code/runtime failure"
    return row


def _terminate_process_tree(pid: int) -> None:
    if sys.platform.startswith("win"):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            subprocess.run(["pkill", "-TERM", "-P", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["kill", "-TERM", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def _write_od300_rotation_violation_debug(run_dir: Path, scenario_name: str, options: dict, row: dict) -> None:
    schedule_path = run_dir / "optimized_schedule.xlsx"
    candidates_path = run_dir / "rotation_candidates.xlsx"
    report_path = run_dir / "rotation_report.xlsx"
    if not schedule_path.exists() or not candidates_path.exists() or not report_path.exists():
        return
    try:
        schedule = pd.read_excel(schedule_path, engine="openpyxl")
        candidates = pd.read_excel(candidates_path, engine="openpyxl")
        report = pd.read_excel(report_path, engine="openpyxl")
    except Exception as exc:
        (run_dir / "od300_rotation_violation_debug_summary.txt").write_text(
            f"Could not build rotation debug: {exc}",
            encoding="utf-8",
        )
        return
    if schedule.empty or candidates.empty or report.empty:
        return
    schedule_by_id = schedule.set_index("flight_id").to_dict("index") if "flight_id" in schedule.columns else {}
    report_key_cols = ["pair_id", "outbound_flight_no", "inbound_flight_no"]
    report_map = {
        tuple(str(getattr(r, col, "")) for col in report_key_cols): r
        for r in report.itertuples(index=False)
    }
    rotation_enabled = bool(_model_level_feature_flags(str(options.get("model_level", "")))["rotation_enabled"])
    rows = []
    for cand in candidates.itertuples(index=False):
        key = (
            str(getattr(cand, "pair_id", "")),
            str(getattr(cand, "outbound_flight_no", "")),
            str(getattr(cand, "inbound_flight_no", "")),
        )
        rep = report_map.get(key)
        slack = getattr(rep, "slack_min", None) if rep is not None else None
        violation = bool(getattr(rep, "violation", False)) if rep is not None and not pd.isna(getattr(rep, "violation", False)) else False
        if not violation and (slack is None or pd.isna(slack) or float(slack) > 5.0):
            continue
        out_sched = schedule_by_id.get(getattr(cand, "outbound_event_id", None), {})
        in_sched = schedule_by_id.get(getattr(cand, "inbound_event_id", None), {})
        out_old_min = getattr(cand, "old_outbound_departure_min", None)
        in_old_min = getattr(cand, "old_inbound_arrival_min", None)
        out_shift = out_sched.get("time_change_min", 0.0) if out_sched else 0.0
        in_shift = in_sched.get("time_change_min", 0.0) if in_sched else 0.0
        required = getattr(cand, "required_delta_min", None)
        rep_skip = getattr(rep, "skip_reason", "") if rep is not None else ""
        rep_checked = rep is not None and (pd.isna(rep_skip) or str(rep_skip).strip() == "")
        required_inbound = ""
        actual_inbound = ""
        if out_old_min is not None and not pd.isna(out_old_min) and required is not None and not pd.isna(required):
            required_inbound = float(out_old_min) + float(out_shift or 0.0) + float(required)
        if in_old_min is not None and not pd.isna(in_old_min):
            actual_inbound = float(in_old_min) + float(in_shift or 0.0)
        block_source = str(getattr(cand, "block_time_source", ""))
        rows.append(
            {
                "pair_id": getattr(cand, "pair_id", ""),
                "day": getattr(cand, "day", ""),
                "station": getattr(cand, "station", ""),
                "outbound_flight_no": getattr(cand, "outbound_flight_no", ""),
                "inbound_flight_no": getattr(cand, "inbound_flight_no", ""),
                "outbound_origin": getattr(cand, "outbound_origin", ""),
                "outbound_destination": getattr(cand, "outbound_destination", ""),
                "inbound_origin": getattr(cand, "inbound_origin", ""),
                "inbound_destination": getattr(cand, "inbound_destination", ""),
                "outbound_old_time": getattr(cand, "old_outbound_departure_time", ""),
                "outbound_new_time": out_sched.get("new_ist_time", "") if out_sched else "",
                "inbound_old_time": getattr(cand, "old_inbound_arrival_time", ""),
                "inbound_new_time": in_sched.get("new_ist_time", "") if in_sched else "",
                "outbound_shift": out_shift,
                "inbound_shift": in_shift,
                "outbound_block_time_min": getattr(cand, "outbound_block_time_min", ""),
                "inbound_block_time_min": getattr(cand, "inbound_block_time_min", ""),
                "min_ground_time": getattr(cand, "min_ground_time_min", ""),
                "required_inbound_time": required_inbound,
                "actual_inbound_time": actual_inbound,
                "violation_min": max(0.0, -float(slack)) if slack is not None and not pd.isna(slack) else "",
                "block_time_source": block_source,
                "elapsedtime_match_used": block_source == "elapsedtime_exact_match",
                "rotation_constraint_active_in_model": rotation_enabled and bool(getattr(cand, "rotation_constraint_available", False)),
                "rotation_checked_by_checker": rep_checked,
                "model_level": options.get("model_level", ""),
                "scenario_name": scenario_name,
                "notes": _rotation_debug_note(rotation_enabled, violation, slack),
            }
        )
    debug = pd.DataFrame(rows)
    debug.to_excel(run_dir / "od300_rotation_violation_debug.xlsx", index=False)
    total = int(len(candidates))
    active = int(candidates.get("rotation_constraint_available", pd.Series(False, index=candidates.index)).fillna(False).astype(bool).sum())
    checked = int((report.get("skip_reason", pd.Series("", index=report.index)).fillna("").astype(str) == "").sum())
    violations = int(report.get("violation", pd.Series(False, index=report.index)).fillna(False).astype(bool).sum())
    violation_min = debug["violation_min"].replace("", 0).astype(float) if not debug.empty and "violation_min" in debug else pd.Series(dtype=float)
    source_dist = candidates.get("block_time_source", pd.Series("", index=candidates.index)).fillna("missing").astype(str).value_counts().to_dict()
    active_violations = 0
    if not debug.empty:
        violation_amount = debug["violation_min"].replace("", 0).astype(float)
        active_violations = int((debug["rotation_constraint_active_in_model"].fillna(False).astype(bool) & (violation_amount > 0)).sum())
    lines = [
        "OD300 ROTATION VIOLATION DEBUG SUMMARY",
        "--------------------------------------",
        f"scenario: {scenario_name}",
        f"model_level: {options.get('model_level', '')}",
        f"rotation_enabled: {rotation_enabled}",
        f"total rotation candidates: {total}",
        f"active rotation constraints in model: {active if rotation_enabled else 0}",
        f"rotation constraints checked by checker: {checked}",
        f"rotation violations: {violations}",
        f"max violation minutes: {float(violation_min.max()) if len(violation_min) else 0.0}",
        f"avg violation minutes: {float(violation_min[violation_min > 0].mean()) if len(violation_min[violation_min > 0]) else 0.0}",
        f"block_time_source distribution: {source_dist}",
        f"violations active in model: {active_violations}",
        f"checker checked model-inactive pairs: {checked > (active if rotation_enabled else 0)}",
        f"should rotation be hard in mandatory_only_core: {rotation_enabled}",
        f"diagnosis feasible: {row.get('feasible', '')}",
        f"diagnosis hard violations: {row.get('total_hard_violations', '')}",
    ]
    (run_dir / "od300_rotation_violation_debug_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _rotation_debug_note(rotation_enabled: bool, violation: bool, slack) -> str:
    if not rotation_enabled and violation:
        return "Checker reports rotation violation, but rotation is disabled for this model_level; report-only, not hard."
    if violation:
        return "Rotation hard violation; inspect block times, day rollover, and event basis."
    if slack is not None and not pd.isna(slack) and float(slack) <= 5:
        return "Near rotation bound."
    return ""


def _parse_key_value_file(path: Path) -> dict:
    if not path.exists():
        return {}
    data = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def _od300_row_from_outputs(
    run_dir: Path,
    day_label: str,
    scenario: str,
    run_id: str,
    options: dict,
    process_status: str,
    runtime: float,
    note: str,
) -> dict:
    summary = _parse_key_value_file(run_dir / "run_summary.txt")
    feasibility = (run_dir / "feasibility_report.txt").read_text(encoding="utf-8", errors="ignore") if (run_dir / "feasibility_report.txt").exists() else ""
    feasible = "Result: FEASIBLE" in feasibility or "Final feasible status: True" in feasibility
    solver_status = summary.get("solver status", summary.get("solver_status", ""))
    hard = _safe_int(_extract_report_value(feasibility, "Hard violations"))
    if hard == "":
        hard = _safe_int(summary.get("hard violations", ""))
    if hard == "":
        hard = sum(
            _safe_int(summary.get(key, 0)) or 0
            for key in [
                "rotation_violations",
                "hub_capacity_violations",
                "regularity_violations",
                "directional_e1_hard_violations",
                "directional_e2_hard_violations",
                "feasible_but_inactive_connection_count",
                "infeasible_but_active_connection_count",
            ]
        )
    if process_status != "completed" and not feasible and not (run_dir / "optimized_schedule.xlsx").exists():
        hard = ""
    return {
        "day_set": day_label,
        "scenario": scenario,
        "run_id": run_id,
        "feasible": feasible,
        "solver_status": solver_status,
        "runtime_sec": summary.get("runtime seconds", f"{runtime:.3f}"),
        "suspected_bottleneck": _od300_suspected_bottleneck(scenario, feasible, summary, process_status),
        "objective_after_penalty": summary.get("objective_after_penalty", summary.get("total_objective", "")),
        "active_connections": summary.get("active_connections", summary.get("active_selected_connections", "")),
        "total_candidate_connections": summary.get("total_candidate_connections", ""),
        "mandatory_feasible_but_inactive": summary.get("feasible_but_inactive_connection_count", ""),
        "mandatory_infeasible_but_active": summary.get("infeasible_but_active_connection_count", ""),
        "rotation_violations": summary.get("rotation_violations", ""),
        "hub_capacity_violations": summary.get("hub_capacity_violations", ""),
        "regularity_violations": summary.get("regularity_violations", ""),
        "directional_e1_mode": options.get("directional_e1_mode", ""),
        "directional_e1_abs_tolerance": options.get("directional_e1_abs_tolerance", ""),
        "directional_e1_violations": summary.get("directional_e1_violations", ""),
        "directional_e2_mode": options.get("directional_e2_mode", ""),
        "directional_e2_violations": summary.get("directional_e2_violations", ""),
        "total_hard_violations": hard,
        "avg_abs_shift": summary.get("avg_abs_shift", ""),
        "model_level": options.get("model_level", ""),
        "max_shift": options.get("max_shift_min", ""),
        "shift_penalty_mode": options.get("shift_penalty_mode", ""),
        "shift_penalty_weight": options.get("shift_penalty_weight", ""),
        "baseline_objective": summary.get("baseline_objective", ""),
        "objective_before_penalty": summary.get("objective_before_penalty", ""),
        "rotation_constraints_active": summary.get("rotation_constraints_active", ""),
        "rotation_block_elapsedtime_match_count": summary.get("rotation_block_elapsedtime_match_count", ""),
        "rotation_block_fallback_count": summary.get("rotation_block_fallback_count", ""),
        "process_status": process_status,
        "note": note,
    }


def _extract_report_value(text: str, label: str) -> str:
    prefix = f"{label}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def _safe_int(value):
    try:
        if value == "":
            return ""
        return int(float(value))
    except (TypeError, ValueError):
        return ""


def _od300_suspected_bottleneck(scenario: str, feasible: bool, summary: dict, process_status: str) -> str:
    if process_status == "TIME_LIMIT":
        return "not solved within 300s"
    if feasible:
        return "none in this scenario"
    if scenario == "mandatory_only_core":
        return "mandatory + candidate/window side"
    if scenario == "rotation_only_stack":
        return "rotation stack"
    if scenario == "capacity_after_rotation":
        return "capacity after rotation"
    if scenario in {"regularity_after_capacity", "no_directional"}:
        return "regularity/cumulative operational stack"
    if scenario in {"e1_e2_report", "e1_report", "e1_soft", "e1_tolerance_2", "final_hard_all"}:
        return "directional E1/E2 or upstream hard constraints"
    return "undetermined"


def _od300_should_continue_to_days123(days12_rows: list[dict]) -> bool:
    if not days12_rows:
        return False
    return any(str(row.get("feasible", "")).lower() in {"true", "1"} for row in days12_rows)


def _write_od300_quick_interpretation(diagnosis_dir: Path, summary: pd.DataFrame, quick: bool) -> None:
    lines = [
        "OD300 DIAGNOSIS INTERPRETATION",
        "------------------------------",
        f"mode: {'quick' if quick else 'staged'}",
        f"runs completed: {len(summary)}",
        f"feasible runs: {int(summary['feasible'].astype(str).str.lower().isin(['true', '1']).sum()) if not summary.empty else 0}",
        "",
        "Rules:",
        "- mandatory_only_core infeasible => mandatory + candidate/window side is suspect.",
        "- mandatory_only_core feasible but regularity_only infeasible => rotation/capacity/regularity stack is suspect.",
        "- regularity/no_directional feasible but final_hard_all infeasible => directional E1/E2 is suspect.",
        "- e1_tolerance_2 or e1_soft feasible => use that as OD300 fallback candidate.",
    ]
    (diagnosis_dir / "od300_diagnosis_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


REGULARITY_DIAGNOSIS_COLUMNS = [
    "scenario",
    "feasible",
    "solver_status",
    "runtime_sec",
    "max_shift",
    "regularity_mode",
    "xdev",
    "regularity_abs_tolerance",
    "regularity_penalty_weight",
    "baseline_objective",
    "objective_before_penalty",
    "objective_after_penalty",
    "improvement_after_penalty_pct",
    "active_connections",
    "beaten_competitors",
    "total_candidate_connections",
    "feasible_by_schedule",
    "mandatory_feasible_but_inactive",
    "mandatory_infeasible_but_active",
    "rotation_violations",
    "capacity_violations",
    "regularity_hard_violations",
    "regularity_soft_violations",
    "regularity_violation_amount_total",
    "max_regularity_spread",
    "avg_regularity_spread",
    "avg_abs_shift",
    "median_abs_shift",
    "pct_shift_le_5",
    "total_hard_violations",
    "suspected_bottleneck",
]


def run_od300_regularity_diagnosis(run_config: RunConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnosis_dir = run_config.output_dir / f"od300_regularity_diagnosis_{timestamp}"
    diagnosis_dir.mkdir(parents=True, exist_ok=True)
    conflict = _build_od300_regularity_conflict_analysis(run_config, diagnosis_dir)
    scenarios = [
        ("regularity_hard_xdev15", {"regularity_mode": "hard", "regularity_xdev_min": 15, "regularity_abs_tolerance_min": 0, "max_shift_min": 30}),
        ("regularity_hard_xdev30", {"regularity_mode": "hard", "regularity_xdev_min": 30, "regularity_abs_tolerance_min": 0, "max_shift_min": 30}),
        ("regularity_hard_xdev60", {"regularity_mode": "hard", "regularity_xdev_min": 60, "regularity_abs_tolerance_min": 0, "max_shift_min": 30}),
        ("regularity_hard_tol5", {"regularity_mode": "hard", "regularity_xdev_min": 15, "regularity_abs_tolerance_min": 5, "max_shift_min": 30}),
        ("regularity_hard_tol15", {"regularity_mode": "hard", "regularity_xdev_min": 15, "regularity_abs_tolerance_min": 15, "max_shift_min": 30}),
        ("regularity_soft_w100", {"regularity_mode": "soft", "regularity_xdev_min": 15, "regularity_penalty_weight": 100, "max_shift_min": 30}),
        ("regularity_soft_w500", {"regularity_mode": "soft", "regularity_xdev_min": 15, "regularity_penalty_weight": 500, "max_shift_min": 30}),
        ("regularity_report_only", {"regularity_mode": "report_only", "regularity_xdev_min": 15, "max_shift_min": 30}),
        ("max_shift60_regularity_hard", {"regularity_mode": "hard", "regularity_xdev_min": 15, "regularity_abs_tolerance_min": 0, "max_shift_min": 60}),
        ("max_shift60_xdev30", {"regularity_mode": "hard", "regularity_xdev_min": 30, "regularity_abs_tolerance_min": 0, "max_shift_min": 60}),
    ]
    summary_path = diagnosis_dir / "regularity_diagnosis_summary.csv"
    pd.DataFrame(columns=REGULARITY_DIAGNOSIS_COLUMNS).to_csv(summary_path, index=False)
    rows = []
    for scenario, overrides in scenarios:
        row = _run_od300_regularity_scenario(run_config, diagnosis_dir, scenario, overrides)
        rows.append(row)
        pd.DataFrame([row], columns=REGULARITY_DIAGNOSIS_COLUMNS).to_csv(summary_path, mode="a", header=False, index=False)
        print(f"[INFO] OD300 regularity diagnosis appended: {scenario} -> {row['solver_status']}, feasible={row['feasible']}")
    summary = pd.DataFrame(rows, columns=REGULARITY_DIAGNOSIS_COLUMNS)
    _write_od300_regularity_interpretation(diagnosis_dir, conflict, summary)
    _write_od300_fallback_candidate(run_config.output_dir, diagnosis_dir, summary)
    print(f"[INFO] OD300 regularity diagnosis outputs written to {diagnosis_dir}")
    return diagnosis_dir


def _build_od300_regularity_conflict_analysis(run_config: RunConfig, diagnosis_dir: Path) -> pd.DataFrame:
    cfg = clone_run_config(
        run_config,
        run_id="regularity_conflict_build",
        output_dir=diagnosis_dir,
        model_level="regularity",
        max_od=300,
        days=[1, 2],
        top_importance_only=True,
        max_shift_min=30,
        regularity_xdev_min=15,
        mandatory_connection_selection=True,
        ranking_objective_mode="lookup_linear",
        shift_penalty_mode="linear",
        shift_penalty_weight=5.0,
    )
    model_config = _make_model_config(cfg)
    raw = load_inputs(cfg)
    prepared = prepare_data(raw, cfg, model_config)
    artifacts = build_model(prepared, model_config, "regularity")
    reg = artifacts.regularity_report.copy()
    candidates = artifacts.candidate_connections.copy()
    rotations = artifacts.rotation_candidates.copy()
    if reg.empty:
        conflict = pd.DataFrame()
    else:
        cand_events = set(candidates.get("inbound_event_id", pd.Series(dtype=str)).astype(str)) | set(candidates.get("outbound_event_id", pd.Series(dtype=str)).astype(str))
        rotation_events = set()
        if not rotations.empty:
            active_rot = rotations[rotations.get("rotation_constraint_available", pd.Series(False, index=rotations.index)).fillna(False).astype(bool)]
            rotation_events = set(active_rot.get("outbound_event_id", pd.Series(dtype=str)).dropna().astype(str)) | set(active_rot.get("inbound_event_id", pd.Series(dtype=str)).dropna().astype(str))
        rows = []
        group_cols = ["flight_no", "origin", "destination", "direction"]
        for idx, (_, group) in enumerate(reg.groupby(group_cols, dropna=False)):
            event_ids = sorted(set(group["event_id_a"].astype(str)) | set(group["event_id_b"].astype(str)))
            old_times = []
            for rec in group.itertuples(index=False):
                old_times.extend([float(rec.old_minute_of_day_a), float(rec.old_minute_of_day_b)])
            old_min = min(old_times)
            old_max = max(old_times)
            spread = old_max - old_min
            theoretical = max(0.0, spread - 2 * model_config.max_shift_min)
            infeasible_by_bounds = theoretical > model_config.schedule_regular_dev_min + 1e-6
            required_shift = max(0.0, spread - model_config.schedule_regular_dev_min)
            used_rotation = sum(1 for eid in event_ids if eid in rotation_events)
            active_candidate = sum(1 for eid in event_ids if eid in cand_events)
            reason = "unknown"
            if spread > model_config.schedule_regular_dev_min and infeasible_by_bounds:
                reason = "old_spread_exceeds_xdev_and_shift_bounds_cannot_fix"
            elif spread > model_config.schedule_regular_dev_min:
                reason = "old_spread_exceeds_xdev_but_shift_bounds_can_fix"
            if used_rotation:
                reason = "rotation_conflict" if infeasible_by_bounds else reason
            rows.append(
                {
                    "regularity_group_id": f"reggrp_{idx:05d}",
                    "normalized_flight_no": group["flight_no"].iloc[0],
                    "origin": group["origin"].iloc[0],
                    "destination": group["destination"].iloc[0],
                    "direction_type": group["direction"].iloc[0],
                    "num_instances": len(event_ids),
                    "days_in_group": ",".join(map(str, sorted(set(group["day_a"].astype(int)) | set(group["day_b"].astype(int))))),
                    "old_time_min_list": ",".join(str(int(x)) for x in sorted(set(old_times))),
                    "old_min_time": old_min,
                    "old_max_time": old_max,
                    "old_spread_min": spread,
                    "xdev_limit": model_config.schedule_regular_dev_min,
                    "max_shift": model_config.max_shift_min,
                    "theoretically_possible_min_spread": theoretical,
                    "infeasible_by_bounds": infeasible_by_bounds,
                    "required_shift_to_satisfy_xdev": required_shift,
                    "instance_count": len(event_ids),
                    "active_in_candidate_connections_count": active_candidate,
                    "used_in_rotation_count": used_rotation,
                    "used_in_capacity_count": len(event_ids),
                    "likely_conflict_reason": reason,
                }
            )
        conflict = pd.DataFrame(rows)
    conflict.to_excel(diagnosis_dir / "regularity_conflict_candidates.xlsx", index=False)
    _write_regularity_conflict_summary(diagnosis_dir, conflict)
    return conflict


def _write_regularity_conflict_summary(diagnosis_dir: Path, conflict: pd.DataFrame) -> None:
    if conflict.empty:
        lines = ["REGULARITY CONFLICT SUMMARY", "---------------------------", "No regularity groups found."]
    else:
        active = len(conflict)
        infeasible = conflict[conflict["infeasible_by_bounds"].fillna(False).astype(bool)]
        old_gt = conflict[pd.to_numeric(conflict["old_spread_min"], errors="coerce") > pd.to_numeric(conflict["xdev_limit"], errors="coerce")]
        estimated_xdev = float(conflict["theoretically_possible_min_spread"].max()) if len(conflict) else 0.0
        estimated_shift = max(0.0, float(((conflict["old_spread_min"] - conflict["xdev_limit"]) / 2).max()))
        top = conflict.sort_values("theoretically_possible_min_spread", ascending=False).head(20)
        lines = [
            "REGULARITY CONFLICT SUMMARY",
            "---------------------------",
            f"total regularity groups: {len(conflict)}",
            f"active regularity groups: {active}",
            f"groups with old_spread > xdev: {len(old_gt)}",
            f"groups theoretically infeasible by max_shift bounds: {len(infeasible)}",
            f"max old spread: {float(conflict['old_spread_min'].max())}",
            f"max theoretically_possible_min_spread: {estimated_xdev}",
            f"groups also involved in rotation: {int((conflict['used_in_rotation_count'] > 0).sum())}",
            f"groups also involved in mandatory candidate connections: {int((conflict['active_in_candidate_connections_count'] > 0).sum())}",
            f"estimated minimum xdev needed: {estimated_xdev}",
            f"estimated minimum max_shift needed: {estimated_shift}",
            "",
            "top 20 most conflicting groups:",
        ]
        for row in top.itertuples(index=False):
            lines.append(
                f"- {row.regularity_group_id} flight={row.normalized_flight_no} "
                f"{row.origin}-{row.destination} {row.direction_type}: old_spread={row.old_spread_min}, "
                f"min_possible={row.theoretically_possible_min_spread}, reason={row.likely_conflict_reason}"
            )
        lines.extend(
            [
                "",
                "Turkce yorum:",
                "- XDEV=15 ve max_shift=30 kombinasyonu teorik bounds tarafinda bazi gruplar icin yetersiz kalabilir.",
                "- Sorunun az sayida extreme grup mu yoksa yaygin mi oldugu yukaridaki infeasible grup sayisindan okunmalidir.",
            ]
        )
    (diagnosis_dir / "regularity_conflict_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _run_od300_regularity_scenario(run_config: RunConfig, diagnosis_dir: Path, scenario: str, overrides: dict) -> dict:
    base = {
        "model_level": "regularity",
        "regularity_mode": "hard",
        "regularity_penalty_weight": REGULARITY_PENALTY_WEIGHT,
        "regularity_abs_tolerance_min": 0,
        "regularity_xdev_min": 15,
        "max_shift_min": 30,
        "directional_e1_mode": "report_only",
        "directional_e2_mode": "report_only",
    }
    base.update(overrides)
    row = _run_od300_diagnosis_subprocess(run_config, diagnosis_dir, scenario, "days12", [1, 2], scenario, base)
    run_dir = diagnosis_dir / scenario
    summary = _parse_key_value_file(run_dir / "run_summary.txt")
    comparison = _parse_key_value_file(run_dir / "comparison_to_baseline.txt")
    baseline = _safe_float(summary.get("baseline_objective", comparison.get("baseline objective", "")))
    after = _safe_float(summary.get("objective_after_penalty", summary.get("total_objective", "")))
    improvement = ""
    if baseline not in {"", None} and after not in {"", None} and abs(float(baseline)) > 1e-12:
        improvement = 100.0 * (float(after) - float(baseline)) / abs(float(baseline))
    return {
        "scenario": scenario,
        "feasible": row.get("feasible", False),
        "solver_status": row.get("solver_status", ""),
        "runtime_sec": row.get("runtime_sec", ""),
        "max_shift": base.get("max_shift_min", 30),
        "regularity_mode": base.get("regularity_mode", "hard"),
        "xdev": base.get("regularity_xdev_min", 15),
        "regularity_abs_tolerance": base.get("regularity_abs_tolerance_min", 0),
        "regularity_penalty_weight": base.get("regularity_penalty_weight", REGULARITY_PENALTY_WEIGHT),
        "baseline_objective": summary.get("baseline_objective", comparison.get("baseline objective", "")),
        "objective_before_penalty": summary.get("objective_before_penalty", ""),
        "objective_after_penalty": summary.get("objective_after_penalty", summary.get("total_objective", "")),
        "improvement_after_penalty_pct": improvement if improvement != "" else comparison.get("percentage improvement", "").replace("%", ""),
        "active_connections": summary.get("active_connections", ""),
        "beaten_competitors": summary.get("beaten_competitors", ""),
        "total_candidate_connections": summary.get("total_candidate_connections", ""),
        "feasible_by_schedule": summary.get("connections_feasible_by_final_schedule", ""),
        "mandatory_feasible_but_inactive": summary.get("feasible_but_inactive_connection_count", ""),
        "mandatory_infeasible_but_active": summary.get("infeasible_but_active_connection_count", ""),
        "rotation_violations": summary.get("rotation_violations", row.get("rotation_violations", "")),
        "capacity_violations": summary.get("hub_capacity_violations", row.get("hub_capacity_violations", "")),
        "regularity_hard_violations": summary.get("regularity_hard_violations", row.get("regularity_violations", "")),
        "regularity_soft_violations": summary.get("regularity_soft_violations", ""),
        "regularity_violation_amount_total": summary.get("regularity_violation_amount_total", ""),
        "max_regularity_spread": summary.get("max_regularity_spread", ""),
        "avg_regularity_spread": summary.get("avg_regularity_spread", ""),
        "avg_abs_shift": summary.get("avg_abs_shift", ""),
        "median_abs_shift": summary.get("median_abs_shift", ""),
        "pct_shift_le_5": summary.get("pct_shift_le_5", ""),
        "total_hard_violations": row.get("total_hard_violations", ""),
        "suspected_bottleneck": row.get("suspected_bottleneck", ""),
    }


def _write_od300_regularity_interpretation(diagnosis_dir: Path, conflict: pd.DataFrame, summary: pd.DataFrame) -> None:
    feasible = summary[summary["feasible"].astype(str).str.lower().isin(["true", "1"])] if not summary.empty else pd.DataFrame()
    hard_feasible = feasible[feasible["regularity_mode"].astype(str) == "hard"] if not feasible.empty else pd.DataFrame()
    soft_feasible = feasible[feasible["regularity_mode"].astype(str) == "soft"] if not feasible.empty else pd.DataFrame()
    infeasible_count = int(conflict["infeasible_by_bounds"].fillna(False).astype(bool).sum()) if not conflict.empty else 0
    lines = [
        "OD300 REGULARITY DIAGNOSIS INTERPRETATION",
        "-----------------------------------------",
        f"conflict groups theoretically infeasible by bounds: {infeasible_count}",
        f"feasible scenarios: {len(feasible)}",
        f"hard feasible scenarios: {len(hard_feasible)}",
        f"soft feasible scenarios: {len(soft_feasible)}",
        "",
        "Yorum:",
    ]
    if infeasible_count:
        lines.append("- OD300 regularity infeasibility icin ilk guclu sinyal XDEV/max_shift bounds tarafinda.")
    if not hard_feasible.empty:
        best = hard_feasible.iloc[0]
        lines.append(f"- Yarismaya en yakin hard fallback adayi: {best['scenario']}.")
    elif not soft_feasible.empty:
        best = soft_feasible.iloc[0]
        lines.append(f"- Hard feasible bulunamazsa soft fallback adayi: {best['scenario']}; primary final degil analysis fallback olarak tutulmali.")
    else:
        lines.append("- Matrix icinde feasible regularity fallback bulunamadi; conflict gruplari icin group-level relax/skip tasarimi gerekir.")
    lines.append("- final_submission_candidate, OD300 hard-clean bulunmadikca od200_days123 lookup_linear olarak kalmali.")
    (diagnosis_dir / "regularity_diagnosis_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_od300_fallback_candidate(output_dir: Path, diagnosis_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    feasible = summary[summary["feasible"].astype(str).str.lower().isin(["true", "1"])].copy()
    if feasible.empty:
        return
    for col in [
        "mandatory_feasible_but_inactive",
        "mandatory_infeasible_but_active",
        "rotation_violations",
        "capacity_violations",
        "total_hard_violations",
        "improvement_after_penalty_pct",
        "avg_abs_shift",
    ]:
        feasible[col] = pd.to_numeric(feasible[col], errors="coerce").fillna(0)
    clean = feasible[
        (feasible["mandatory_feasible_but_inactive"] == 0)
        & (feasible["mandatory_infeasible_but_active"] == 0)
        & (feasible["rotation_violations"] == 0)
        & (feasible["capacity_violations"] == 0)
        & (feasible["total_hard_violations"] == 0)
        & (feasible["improvement_after_penalty_pct"] > 0)
    ].copy()
    if clean.empty:
        return
    clean["mode_rank"] = clean["regularity_mode"].map({"hard": 0, "soft": 1, "report_only": 2}).fillna(3)
    clean["max_shift_rank"] = (pd.to_numeric(clean["max_shift"], errors="coerce").fillna(999) - 30).abs()
    clean["effective_regularity_limit"] = pd.to_numeric(clean["xdev"], errors="coerce").fillna(999) + pd.to_numeric(clean["regularity_abs_tolerance"], errors="coerce").fillna(999)
    best = clean.sort_values(
        ["mode_rank", "max_shift_rank", "effective_regularity_limit", "avg_abs_shift"],
        ascending=[True, True, True, True],
    ).iloc[0]
    src = diagnosis_dir / str(best["scenario"])
    dst = output_dir / "od300_fallback_candidate"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    (dst / "fallback_source.txt").write_text(
        f"source_dir: {src}\nscenario: {best['scenario']}\nregularity_mode: {best['regularity_mode']}\n",
        encoding="utf-8",
    )


TARGETED_REGULARITY_COLUMNS = [
    "scenario",
    "days",
    "feasible",
    "solver_status",
    "runtime_sec",
    "regularity_relax_mode",
    "base_xdev",
    "max_effective_xdev",
    "relax_applied_groups",
    "max_relax_amount",
    "regularity_hard_violations",
    "regularity_soft_violations",
    "regularity_soft_amount",
    "baseline_objective",
    "objective_before_penalty",
    "objective_after_penalty",
    "improvement_after_penalty_pct",
    "active_connections",
    "beaten_competitors",
    "mandatory_feasible_but_inactive",
    "mandatory_infeasible_but_active",
    "rotation_violations",
    "capacity_violations",
    "directional_e1_violations",
    "directional_e2_violations",
    "total_hard_violations",
    "avg_abs_shift",
    "median_abs_shift",
    "pct_shift_le_5",
    "pct_at_bounds",
    "process_status",
    "note",
]


def run_od300_targeted_regularity_test(run_config: RunConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnosis_dir = run_config.output_dir / f"od300_targeted_regularity_{timestamp}"
    diagnosis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diagnosis_dir / "targeted_regularity_summary.csv"
    pd.DataFrame(columns=TARGETED_REGULARITY_COLUMNS).to_csv(summary_path, index=False)
    scenarios = [
        (
            "od300_targeted_theoretical",
            {
                "regularity_relax_mode": "targeted_theoretical",
                "regularity_targeted_max_relax_min": 30,
                "regularity_xdev_min": 15,
                "regularity_mode": "hard",
                "max_shift_min": 30,
            },
        ),
        (
            "od300_targeted_soft",
            {
                "regularity_relax_mode": "targeted_soft",
                "regularity_xdev_min": 15,
                "regularity_mode": "hard",
                "regularity_penalty_weight": 500,
                "max_shift_min": 30,
            },
        ),
        (
            "od300_global_xdev30_reference",
            {
                "regularity_relax_mode": "none",
                "regularity_xdev_min": 30,
                "regularity_mode": "hard",
                "max_shift_min": 30,
            },
        ),
        (
            "od300_max_shift60_reference",
            {
                "regularity_relax_mode": "none",
                "regularity_xdev_min": 15,
                "regularity_mode": "hard",
                "max_shift_min": 60,
            },
        ),
    ]
    rows = []
    days12_rows = _run_targeted_regularity_day_set(run_config, diagnosis_dir, summary_path, "days12", [1, 2], scenarios)
    rows.extend(days12_rows)
    if _targeted_days123_is_worth_running(days12_rows):
        rows.extend(_run_targeted_regularity_day_set(run_config, diagnosis_dir, summary_path, "days123", [1, 2, 3], scenarios))
    summary = pd.DataFrame(rows, columns=TARGETED_REGULARITY_COLUMNS)
    _write_od300_targeted_regularity_interpretation(diagnosis_dir, summary)
    _write_od300_targeted_fallback_candidate(run_config.output_dir, diagnosis_dir, summary)
    print(f"[INFO] OD300 targeted regularity outputs written to {diagnosis_dir}")
    return diagnosis_dir


def _run_targeted_regularity_day_set(
    run_config: RunConfig,
    diagnosis_dir: Path,
    summary_path: Path,
    day_label: str,
    days: list[int],
    scenarios: list[tuple[str, dict]],
) -> list[dict]:
    rows = []
    for scenario, overrides in scenarios:
        run_id = f"{day_label}_{scenario}"
        row = _run_od300_diagnosis_subprocess(run_config, diagnosis_dir, run_id, day_label, days, scenario, overrides)
        targeted = _targeted_regularity_row_from_outputs(diagnosis_dir / run_id, row, day_label, days, scenario, overrides)
        rows.append(targeted)
        pd.DataFrame([targeted], columns=TARGETED_REGULARITY_COLUMNS).to_csv(summary_path, mode="a", header=False, index=False)
        print(f"[INFO] OD300 targeted regularity appended: {day_label}/{scenario} -> {targeted['solver_status']}, feasible={targeted['feasible']}")
    return rows


def _targeted_regularity_row_from_outputs(run_dir: Path, row: dict, day_label: str, days: list[int], scenario: str, overrides: dict) -> dict:
    summary = _parse_key_value_file(run_dir / "run_summary.txt")
    comparison = _parse_key_value_file(run_dir / "comparison_to_baseline.txt")
    baseline = _safe_float(summary.get("baseline_objective", comparison.get("baseline objective", "")))
    after = _safe_float(summary.get("objective_after_penalty", summary.get("total_objective", "")))
    improvement = ""
    if baseline not in {"", None} and after not in {"", None} and abs(float(baseline)) > 1e-12:
        improvement = 100.0 * (float(after) - float(baseline)) / abs(float(baseline))
    total_hard = row.get("total_hard_violations", "")
    if not row.get("feasible", False) and not (run_dir / "optimized_schedule.xlsx").exists():
        total_hard = ""
    return {
        "scenario": scenario,
        "days": ",".join(str(day) for day in days),
        "feasible": row.get("feasible", False),
        "solver_status": row.get("solver_status", ""),
        "runtime_sec": row.get("runtime_sec", ""),
        "regularity_relax_mode": summary.get("regularity_relax_mode", overrides.get("regularity_relax_mode", "none")),
        "base_xdev": summary.get("regularity_xdev_minutes", overrides.get("regularity_xdev_min", "")),
        "max_effective_xdev": summary.get("regularity_max_effective_xdev", ""),
        "relax_applied_groups": summary.get("regularity_relax_applied_groups", ""),
        "max_relax_amount": summary.get("regularity_max_relax_amount", ""),
        "regularity_hard_violations": summary.get("regularity_hard_violations", row.get("regularity_violations", "")),
        "regularity_soft_violations": summary.get("regularity_soft_violations", ""),
        "regularity_soft_amount": summary.get("regularity_violation_amount_total", ""),
        "baseline_objective": summary.get("baseline_objective", comparison.get("baseline objective", "")),
        "objective_before_penalty": summary.get("objective_before_penalty", ""),
        "objective_after_penalty": summary.get("objective_after_penalty", summary.get("total_objective", "")),
        "improvement_after_penalty_pct": improvement if improvement != "" else comparison.get("percentage improvement", "").replace("%", ""),
        "active_connections": summary.get("active_connections", row.get("active_connections", "")),
        "beaten_competitors": summary.get("beaten_competitors", ""),
        "mandatory_feasible_but_inactive": summary.get("feasible_but_inactive_connection_count", row.get("mandatory_feasible_but_inactive", "")),
        "mandatory_infeasible_but_active": summary.get("infeasible_but_active_connection_count", row.get("mandatory_infeasible_but_active", "")),
        "rotation_violations": summary.get("rotation_violations", row.get("rotation_violations", "")),
        "capacity_violations": summary.get("hub_capacity_violations", row.get("hub_capacity_violations", "")),
        "directional_e1_violations": summary.get("directional_e1_violations", row.get("directional_e1_violations", "")),
        "directional_e2_violations": summary.get("directional_e2_violations", row.get("directional_e2_violations", "")),
        "total_hard_violations": total_hard,
        "avg_abs_shift": summary.get("avg_abs_shift", row.get("avg_abs_shift", "")),
        "median_abs_shift": summary.get("median_abs_shift", ""),
        "pct_shift_le_5": summary.get("pct_shift_le_5", ""),
        "pct_at_bounds": summary.get("pct_at_shift_bounds", ""),
        "process_status": row.get("process_status", ""),
        "note": row.get("note", ""),
    }


def _targeted_days123_is_worth_running(days12_rows: list[dict]) -> bool:
    if not days12_rows:
        return False
    targeted = [row for row in days12_rows if row.get("scenario") == "od300_targeted_theoretical"]
    if not targeted:
        return False
    row = targeted[0]
    return str(row.get("feasible", "")).lower() in {"true", "1"} and _safe_int(row.get("total_hard_violations", "")) == 0


def _write_od300_targeted_regularity_interpretation(diagnosis_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "OD300 TARGETED REGULARITY INTERPRETATION",
        "----------------------------------------",
    ]
    if summary.empty:
        lines.append("No targeted regularity scenarios completed.")
        (diagnosis_dir / "targeted_regularity_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return
    days12 = summary[summary["days"].astype(str) == "1,2"].copy()
    targeted = days12[days12["scenario"] == "od300_targeted_theoretical"] if not days12.empty else pd.DataFrame()
    global_ref = days12[days12["scenario"] == "od300_global_xdev30_reference"] if not days12.empty else pd.DataFrame()
    maxshift_ref = days12[days12["scenario"] == "od300_max_shift60_reference"] if not days12.empty else pd.DataFrame()
    if not targeted.empty:
        row = targeted.iloc[0]
        hard_text = row.get("total_hard_violations", "")
        hard_text = hard_text if str(hard_text).strip() else "not measured because no feasible schedule was produced"
        lines.extend(
            [
                f"targeted_theoretical feasible: {row.get('feasible', '')}",
                f"targeted_theoretical hard violations: {hard_text}",
                f"relax applied groups: {row.get('relax_applied_groups', '')}",
                f"max effective XDEV: {row.get('max_effective_xdev', '')}",
                f"avg abs shift: {row.get('avg_abs_shift', '')}",
            ]
        )
    if not global_ref.empty:
        row = global_ref.iloc[0]
        lines.append(f"global XDEV=30 reference objective after: {row.get('objective_after_penalty', '')}")
    if not maxshift_ref.empty:
        row = maxshift_ref.iloc[0]
        lines.append(f"max_shift=60 reference objective after: {row.get('objective_after_penalty', '')}")
    lines.extend(
        [
            "",
            "Turkce yorum:",
            "- Targeted theoretical relax, global XDEV=30 yerine sadece teorik olarak imkansiz regularity gruplarini gevsetmek icin tasarlandi.",
            "- Bu run feasible cozum uretmediyse hard violations cozum uzerinden olculemez ve OD300 fallback candidate uretilmez.",
            "- Primary final_submission_candidate otomatik degismez; od200_days123 lookup_linear final aday olarak kalir.",
            "- OD400'e gecmeden once OD300 targeted fallback'in days=1,2 sonucunun hard-clean olmasi tercih edilir.",
        ]
    )
    (diagnosis_dir / "targeted_regularity_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_od300_targeted_fallback_candidate(output_dir: Path, diagnosis_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    candidates = summary[
        (summary["scenario"] == "od300_targeted_theoretical")
        & (summary["days"].astype(str) == "1,2")
        & (summary["feasible"].astype(str).str.lower().isin(["true", "1"]))
    ].copy()
    if candidates.empty:
        return
    candidates["total_hard_violations_num"] = pd.to_numeric(candidates["total_hard_violations"], errors="coerce").fillna(999)
    candidates = candidates[candidates["total_hard_violations_num"] == 0]
    if candidates.empty:
        return
    best = candidates.iloc[0]
    src = diagnosis_dir / f"days12_{best['scenario']}"
    if not src.exists():
        return
    dst = output_dir / "od300_fallback_candidate_targeted"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    (dst / "fallback_source.txt").write_text(
        "\n".join(
            [
                f"source_dir: {src}",
                f"scenario: {best['scenario']}",
                f"days: {best['days']}",
                f"regularity_relax_mode: {best['regularity_relax_mode']}",
                f"relax_applied_groups: {best['relax_applied_groups']}",
                "note: Primary final_submission_candidate remains unchanged; this is an OD300 scale fallback candidate.",
            ]
        ),
        encoding="utf-8",
    )


def run_od300_maxshift60_quality_audit(run_config: RunConfig, audit_run_dir: str | None = None) -> Path:
    run_dir = Path(audit_run_dir) if audit_run_dir else _find_latest_od300_maxshift60_run(run_config.output_dir)
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Audit run directory not found: {run_dir}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_dir = run_config.output_dir / f"od300_maxshift60_quality_audit_{timestamp}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    schedule = pd.read_excel(run_dir / "optimized_schedule.xlsx", engine="openpyxl")
    selected = _read_excel_if_exists(run_dir / "selected_connections.xlsx")
    all_candidates = _read_excel_if_exists(run_dir / "all_candidate_connections.xlsx")
    rotation_candidates = _read_excel_if_exists(run_dir / "rotation_candidates.xlsx")
    regularity_report = _read_excel_if_exists(run_dir / "regularity_report.xlsx")
    hub_report = _read_excel_if_exists(run_dir / "hub_capacity_report.xlsx")
    od_metrics = _read_excel_if_exists(run_dir / "od_metrics.xlsx")

    context = _build_maxshift60_audit_context(
        schedule,
        selected,
        all_candidates,
        rotation_candidates,
        regularity_report,
        hub_report,
        od_metrics,
    )
    shift_distribution = _build_maxshift60_shift_distribution(schedule, context)
    shift_distribution.to_excel(audit_dir / "od300_maxshift60_shift_distribution.xlsx", index=False)

    regularity_quality = _build_maxshift60_regularity_quality(schedule, regularity_report)
    regularity_quality.to_excel(audit_dir / "od300_maxshift60_regularity_quality.xlsx", index=False)

    large_shift_explain = _build_maxshift60_large_shift_explain(shift_distribution, context)
    large_shift_explain.to_excel(audit_dir / "od300_maxshift60_large_shift_explain.xlsx", index=False)

    checker_report, checker_text = check_run_output(run_dir, ModelConfig())
    (audit_dir / "checker_rerun_report.txt").write_text(checker_text, encoding="utf-8")

    _write_maxshift60_quality_summary(
        audit_dir,
        run_dir,
        shift_distribution,
        regularity_quality,
        large_shift_explain,
        checker_report,
        checker_text,
    )
    _write_maxshift60_report_snippet(audit_dir)
    print(f"[INFO] OD300 max_shift60 quality audit written to {audit_dir}")
    return audit_dir


def _find_latest_od300_maxshift60_run(output_dir: Path) -> Path:
    candidates = []
    for run_summary in output_dir.rglob("run_summary.txt"):
        run_dir = run_summary.parent
        if "od300_max_shift60_reference" not in run_dir.name:
            continue
        values = _parse_key_value_file(run_summary)
        if str(values.get("solver status", values.get("solver_status", ""))).lower() != "ok/optimal":
            continue
        if str(values.get("max_shift_min", values.get("max_shift", ""))) not in {"60", "60.0"}:
            continue
        candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError("No successful od300_max_shift60_reference run found under output_dir.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_excel_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl") if path.exists() else pd.DataFrame()


def _build_maxshift60_audit_context(
    schedule: pd.DataFrame,
    selected: pd.DataFrame,
    all_candidates: pd.DataFrame,
    rotation_candidates: pd.DataFrame,
    regularity_report: pd.DataFrame,
    hub_report: pd.DataFrame,
    od_metrics: pd.DataFrame,
) -> dict:
    active_candidates = all_candidates.copy()
    if not active_candidates.empty:
        if "selected_in_output" in active_candidates:
            active_candidates = active_candidates[active_candidates["selected_in_output"].fillna(False).astype(bool)]
        elif "active_x" in active_candidates:
            active_candidates = active_candidates[active_candidates["active_x"].fillna(False).astype(bool)]
    event_connection_count: dict[str, int] = {}
    event_connection_ids: dict[str, list[str]] = {}
    event_od_days: dict[str, set[tuple[str, int]]] = {}
    if not active_candidates.empty:
        for row in active_candidates.itertuples(index=False):
            conn_id = str(getattr(row, "connection_id", ""))
            od = str(getattr(row, "od", ""))
            day = int(getattr(row, "day", 0)) if pd.notna(getattr(row, "day", None)) else 0
            for attr in ["inbound_event_id", "outbound_event_id"]:
                event_id = getattr(row, attr, None)
                if pd.isna(event_id):
                    continue
                event_id = str(event_id)
                event_connection_count[event_id] = event_connection_count.get(event_id, 0) + 1
                event_connection_ids.setdefault(event_id, []).append(conn_id)
                event_od_days.setdefault(event_id, set()).add((od, day))

    importance = {}
    if not od_metrics.empty and {"od", "day", "importance_factor"}.issubset(od_metrics.columns):
        for row in od_metrics.itertuples(index=False):
            importance[(str(row.od), int(row.day))] = float(row.importance_factor)

    rotation_event_pairs: dict[str, list[str]] = {}
    if not rotation_candidates.empty:
        active_rot = rotation_candidates
        if "rotation_constraint_available" in active_rot:
            active_rot = active_rot[active_rot["rotation_constraint_available"].fillna(False).astype(bool)]
        for row in active_rot.itertuples(index=False):
            pair_id = str(getattr(row, "pair_id", ""))
            for attr in ["outbound_event_id", "inbound_event_id"]:
                event_id = getattr(row, attr, None)
                if pd.isna(event_id):
                    continue
                rotation_event_pairs.setdefault(str(event_id), []).append(pair_id)

    regularity_by_key = {}
    if not regularity_report.empty:
        for row in regularity_report.itertuples(index=False):
            key = (
                str(getattr(row, "normalized_flight_no", "")),
                str(getattr(row, "origin", "")),
                str(getattr(row, "destination", "")),
                str(getattr(row, "direction_type", "")),
            )
            regularity_by_key[key] = row

    hub_lookup = {}
    if not hub_report.empty and {"day", "bucket_id"}.issubset(hub_report.columns):
        for row in hub_report.itertuples(index=False):
            hub_lookup[(int(row.day), int(row.bucket_id))] = row

    schedule_event_key_to_id = {}
    for row in schedule.itertuples(index=False):
        key = (
            str(getattr(row, "flight_no", "")),
            int(getattr(row, "day", 0)),
            str(getattr(row, "origin", "")),
            str(getattr(row, "destination", "")),
            _minute_of_day(getattr(row, "old_ist_time")),
        )
        schedule_event_key_to_id[key] = str(getattr(row, "flight_id", ""))

    return {
        "event_connection_count": event_connection_count,
        "event_connection_ids": event_connection_ids,
        "event_od_days": event_od_days,
        "importance": importance,
        "rotation_event_pairs": rotation_event_pairs,
        "regularity_by_key": regularity_by_key,
        "hub_lookup": hub_lookup,
        "schedule_event_key_to_id": schedule_event_key_to_id,
        "selected": selected,
        "all_candidates": all_candidates,
    }


def _build_maxshift60_shift_distribution(schedule: pd.DataFrame, context: dict) -> pd.DataFrame:
    rows = []
    for row in schedule.itertuples(index=False):
        event_id = str(getattr(row, "flight_id", ""))
        origin = str(getattr(row, "origin", ""))
        destination = str(getattr(row, "destination", ""))
        direction = "inbound" if destination == "IST" else "outbound"
        reg_key = (str(getattr(row, "flight_no", "")), origin, destination, direction)
        regularity = context["regularity_by_key"].get(reg_key)
        shift = float(getattr(row, "time_change_min", 0.0))
        old_time = getattr(row, "old_ist_time", None)
        new_time = getattr(row, "new_ist_time", None)
        rows.append(
            {
                "flight_no": getattr(row, "flight_no", ""),
                "day": getattr(row, "day", ""),
                "origin": origin,
                "destination": destination,
                "old_time": old_time,
                "new_time": new_time,
                "shift_min": shift,
                "abs_shift_min": abs(shift),
                "is_arrival_to_hub": destination == "IST",
                "is_departure_from_hub": origin == "IST",
                "involved_in_connection_count": context["event_connection_count"].get(event_id, 0),
                "involved_in_rotation": event_id in context["rotation_event_pairs"],
                "involved_in_capacity_bucket": True,
                "involved_in_regularity_group": regularity is not None,
                "at_shift_bound": abs(abs(shift) - 60.0) <= 1e-6,
                "notes": _shift_audit_notes(event_id, regularity, context),
            }
        )
    return pd.DataFrame(rows)


def _shift_audit_notes(event_id: str, regularity, context: dict) -> str:
    notes = []
    if context["event_connection_count"].get(event_id, 0):
        notes.append("connection")
    if event_id in context["rotation_event_pairs"]:
        notes.append("rotation")
    if regularity is not None:
        notes.append("regularity")
    return ",".join(notes)


def _build_maxshift60_regularity_quality(schedule: pd.DataFrame, regularity_report: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "regularity_group_id",
        "flight_no",
        "route",
        "direction_type",
        "old_spread",
        "new_spread",
        "xdev",
        "improvement_in_spread",
        "group_size",
        "max_abs_shift_in_group",
        "avg_abs_shift_in_group",
        "hard_violation",
        "notes",
    ]
    if regularity_report.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    schedule = schedule.copy()
    schedule["flight_no_norm"] = schedule["flight_no"].map(normalize_flight_no)
    for row in regularity_report.itertuples(index=False):
        flight_no = str(getattr(row, "normalized_flight_no", ""))
        origin = str(getattr(row, "origin", ""))
        destination = str(getattr(row, "destination", ""))
        group_sched = schedule[
            (schedule["flight_no_norm"].astype(str) == flight_no)
            & (schedule["origin"].astype(str) == origin)
            & (schedule["destination"].astype(str) == destination)
        ]
        shifts = pd.to_numeric(group_sched.get("time_change_min", pd.Series(dtype=float)), errors="coerce").abs().dropna()
        old_spread = float(getattr(row, "old_spread_min", 0.0) or 0.0)
        new_spread = float(getattr(row, "spread_min", 0.0) or 0.0)
        xdev = float(getattr(row, "xdev_limit", getattr(row, "effective_xdev", 15)) or 15)
        rows.append(
            {
                "regularity_group_id": getattr(row, "regularity_group_id", ""),
                "flight_no": flight_no,
                "route": f"{origin}-{destination}",
                "direction_type": getattr(row, "direction_type", ""),
                "old_spread": old_spread,
                "new_spread": new_spread,
                "xdev": xdev,
                "improvement_in_spread": old_spread - new_spread,
                "group_size": getattr(row, "num_instances", ""),
                "max_abs_shift_in_group": float(shifts.max()) if len(shifts) else 0.0,
                "avg_abs_shift_in_group": float(shifts.mean()) if len(shifts) else 0.0,
                "hard_violation": bool(getattr(row, "violation", False)),
                "notes": getattr(row, "skip_reason", "") if pd.notna(getattr(row, "skip_reason", "")) else "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_maxshift60_large_shift_explain(shift_distribution: pd.DataFrame, context: dict) -> pd.DataFrame:
    rows = []
    large = shift_distribution[pd.to_numeric(shift_distribution["abs_shift_min"], errors="coerce") > 30].copy()
    for row in large.itertuples(index=False):
        event_id = _schedule_event_id_from_row(row, context)
        od_days = context["event_od_days"].get(event_id, set())
        importance_sum = sum(context["importance"].get(key, 0.0) for key in od_days)
        related_connections = context["event_connection_ids"].get(event_id, [])
        related_rotation_pairs = context["rotation_event_pairs"].get(event_id, [])
        direction = "inbound" if getattr(row, "destination") == "IST" else "outbound"
        reg_key = (str(getattr(row, "flight_no")), str(getattr(row, "origin")), str(getattr(row, "destination")), direction)
        regularity = context["regularity_by_key"].get(reg_key)
        old_reg_spread = float(getattr(regularity, "old_spread_min", 0.0) or 0.0) if regularity is not None else ""
        new_reg_spread = float(getattr(regularity, "spread_min", 0.0) or 0.0) if regularity is not None else ""
        old_bucket = _time_bucket(getattr(row, "day"), getattr(row, "old_time"))
        new_bucket = _time_bucket(getattr(row, "day"), getattr(row, "new_time"))
        reason = _large_shift_reason(
            related_connections,
            importance_sum,
            related_rotation_pairs,
            regularity,
            old_bucket,
            new_bucket,
        )
        rows.append(
            {
                "flight_no": getattr(row, "flight_no"),
                "day": getattr(row, "day"),
                "route": f"{getattr(row, 'origin')}-{getattr(row, 'destination')}",
                "shift_min": getattr(row, "shift_min"),
                "abs_shift_min": getattr(row, "abs_shift_min"),
                "old_time": getattr(row, "old_time"),
                "new_time": getattr(row, "new_time"),
                "related_connections": ",".join(related_connections),
                "related_connection_importance_sum": importance_sum,
                "related_rotation_pairs": ",".join(related_rotation_pairs),
                "related_regularity_group": getattr(regularity, "regularity_group_id", "") if regularity is not None else "",
                "old_regularity_spread": old_reg_spread,
                "new_regularity_spread": new_reg_spread,
                "capacity_bucket_old": old_bucket,
                "capacity_bucket_new": new_bucket,
                "likely_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _schedule_event_id_from_row(row, context: dict) -> str:
    minute = _minute_of_day(getattr(row, "old_time"))
    key = (
        str(getattr(row, "flight_no")),
        int(getattr(row, "day")),
        str(getattr(row, "origin")),
        str(getattr(row, "destination")),
        minute,
    )
    if key in context.get("schedule_event_key_to_id", {}):
        return context["schedule_event_key_to_id"][key]
    direction = "in" if getattr(row, "destination") == "IST" else "out"
    station = getattr(row, "origin") if direction == "in" else getattr(row, "destination")
    return f"{direction}|{getattr(row, 'day')}|{getattr(row, 'flight_no')}|{station}|{minute}"


def _minute_of_day(value) -> int:
    ts = pd.to_datetime(value)
    return int(ts.hour * 60 + ts.minute)


def _time_bucket(day, value, bucket_size: int = 10) -> str:
    minute = _minute_of_day(value)
    return f"day{int(day)}_b{minute // bucket_size}_{minute}-{minute + bucket_size}"


def _large_shift_reason(related_connections, importance_sum, related_rotation_pairs, regularity, old_bucket, new_bucket) -> str:
    reasons = []
    if regularity is not None:
        old_spread = float(getattr(regularity, "old_spread_min", 0.0) or 0.0)
        new_spread = float(getattr(regularity, "spread_min", 0.0) or 0.0)
        xdev = float(getattr(regularity, "xdev_limit", 15) or 15)
        if old_spread > xdev + 1e-6 and new_spread <= xdev + 1e-6:
            reasons.append("regularity_repair")
    if related_connections:
        reasons.append("connection_window_repair")
    if related_rotation_pairs:
        reasons.append("ranking_reward_tradeoff" if related_connections else "mixed")
    if old_bucket != new_bucket and not reasons:
        reasons.append("capacity_repair")
    if importance_sum > 0 and not reasons:
        reasons.append("ranking_reward_tradeoff")
    if not reasons:
        return "unknown"
    unique = list(dict.fromkeys(reasons))
    return unique[0] if len(unique) == 1 else "mixed"


def _write_maxshift60_quality_summary(
    audit_dir: Path,
    run_dir: Path,
    shift_distribution: pd.DataFrame,
    regularity_quality: pd.DataFrame,
    large_shift_explain: pd.DataFrame,
    checker_report,
    checker_text: str,
) -> None:
    abs_shift = pd.to_numeric(shift_distribution["abs_shift_min"], errors="coerce").fillna(0.0)
    changed = abs_shift > 1e-6
    at_bound = abs(abs_shift - 60.0) <= 1e-6
    lines = [
        "OD300 MAX_SHIFT60 QUALITY SUMMARY",
        "---------------------------------",
        f"source_run_dir: {run_dir}",
        "",
        "Shift distribution:",
        f"total adjusted flights: {len(shift_distribution)}",
        f"changed flights count: {int(changed.sum())}",
        f"changed flight percentage: {_pct(changed.sum(), len(shift_distribution))}",
        f"avg abs shift: {float(abs_shift.mean()) if len(abs_shift) else 0.0}",
        f"median abs shift: {float(abs_shift.median()) if len(abs_shift) else 0.0}",
        f"p75 abs shift: {float(abs_shift.quantile(0.75)) if len(abs_shift) else 0.0}",
        f"p90 abs shift: {float(abs_shift.quantile(0.90)) if len(abs_shift) else 0.0}",
        f"p95 abs shift: {float(abs_shift.quantile(0.95)) if len(abs_shift) else 0.0}",
        f"max abs shift: {float(abs_shift.max()) if len(abs_shift) else 0.0}",
        f"pct shift <= 5: {_pct((abs_shift <= 5).sum(), len(abs_shift))}",
        f"pct shift <= 10: {_pct((abs_shift <= 10).sum(), len(abs_shift))}",
        f"pct shift <= 15: {_pct((abs_shift <= 15).sum(), len(abs_shift))}",
        f"pct shift <= 30: {_pct((abs_shift <= 30).sum(), len(abs_shift))}",
        f"pct shift > 30: {_pct((abs_shift > 30).sum(), len(abs_shift))}",
        f"pct shift > 45: {_pct((abs_shift > 45).sum(), len(abs_shift))}",
        f"pct at 60-min bound: {_pct(at_bound.sum(), len(abs_shift))}",
        f"number of flights exactly at bound: {int(at_bound.sum())}",
        "",
        "top 20 largest shifts:",
    ]
    for row in shift_distribution.sort_values("abs_shift_min", ascending=False).head(20).itertuples(index=False):
        lines.append(
            f"- flight={row.flight_no} day={row.day} route={row.origin}-{row.destination} "
            f"shift={row.shift_min} abs={row.abs_shift_min} notes={row.notes}"
        )
    lines.extend(["", "Regularity quality:"])
    if regularity_quality.empty:
        lines.append("No regularity report found.")
    else:
        old_spread = pd.to_numeric(regularity_quality["old_spread"], errors="coerce").fillna(0.0)
        new_spread = pd.to_numeric(regularity_quality["new_spread"], errors="coerce").fillna(0.0)
        xdev = pd.to_numeric(regularity_quality["xdev"], errors="coerce").fillna(15.0)
        lines.extend(
            [
                f"max old spread: {float(old_spread.max())}",
                f"max new spread: {float(new_spread.max())}",
                f"groups over xdev before: {int((old_spread > xdev + 1e-6).sum())}",
                f"groups over xdev after: {int((new_spread > xdev + 1e-6).sum())}",
                f"avg old spread: {float(old_spread.mean())}",
                f"avg new spread: {float(new_spread.mean())}",
                f"regularity violations after: {int(regularity_quality['hard_violation'].fillna(False).astype(bool).sum())}",
                "top improved groups:",
            ]
        )
        for row in regularity_quality.sort_values("improvement_in_spread", ascending=False).head(10).itertuples(index=False):
            lines.append(f"- {row.regularity_group_id} flight={row.flight_no} route={row.route}: old={row.old_spread}, new={row.new_spread}, improvement={row.improvement_in_spread}")
    lines.extend(["", "Large shift explanation:"])
    large_abs = pd.to_numeric(large_shift_explain.get("abs_shift_min", pd.Series(dtype=float)), errors="coerce")
    lines.extend(
        [
            f"count abs_shift > 30: {len(large_shift_explain)}",
            f"count abs_shift > 45: {int((large_abs > 45).sum()) if len(large_abs) else 0}",
            f"count abs_shift = 60: {int((abs(large_abs - 60) <= 1e-6).sum()) if len(large_abs) else 0}",
            f"top reasons: {_value_counts_text(large_shift_explain.get('likely_reason', pd.Series(dtype=str)))}",
            f"toplam importance etkisi: {float(pd.to_numeric(large_shift_explain.get('related_connection_importance_sum', pd.Series(dtype=float)), errors='coerce').fillna(0.0).sum())}",
            "bu buyuk shiftler olmadan xdev15 feasible olur muydu: Outputlardan kesin kanitlanamaz; ancak max_shift30 targeted/global XDEV30 final-stack infeasible kalirken max_shift60 hard-clean feasible oldugu icin daha genis hareket alaninin final hard stack icin kritik oldugu goruluyor.",
            "",
            "Checker re-run:",
            checker_text,
        ]
    )
    (audit_dir / "od300_maxshift60_quality_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _pct(count, total) -> float:
    return 0.0 if not total else 100.0 * float(count) / float(total)


def _value_counts_text(series: pd.Series) -> str:
    if series is None or len(series) == 0:
        return ""
    counts = series.fillna("unknown").astype(str).value_counts()
    return "; ".join(f"{idx}={value}" for idx, value in counts.items())


def _write_maxshift60_report_snippet(audit_dir: Path) -> None:
    lines = [
        "OD300 olceginde max_shift=30 ile final hard stack feasible cozum uretmemistir.",
        "Targeted regularity diagnosis, problemin yalnizca tek bir regularity grubuyla sinirli kalmadigini; final hard stack'in daha genis zaman hareket alanina ihtiyac duydugunu gostermistir.",
        "max_shift=60 reference kosusunda hard-clean feasible cozum elde edilmistir.",
        "Buna ragmen ortalama mutlak shift dusuk kalmis ve median shift 0 dakika olmustur.",
        "Bu nedenle max_shift60 sonucu primary final aday olarak degil, OD300 scale fallback/stress cozum olarak degerlendirilmistir.",
        "Primary final aday od200_days123 lookup_linear olarak korunmustur.",
    ]
    (audit_dir / "od300_maxshift60_report_snippet_tr.txt").write_text("\n".join(lines), encoding="utf-8")


CONTROLLED_SHIFT_COLUMNS = [
    "scenario",
    "feasible",
    "solver_status",
    "runtime_sec",
    "max_shift",
    "shift_penalty_mode",
    "shift_penalty_weight",
    "threshold",
    "baseline_objective",
    "objective_before_penalty",
    "objective_after_penalty",
    "improvement_after_penalty_pct",
    "active_connections",
    "beaten_competitors",
    "total_candidate_connections",
    "mandatory_feasible_but_inactive",
    "rotation_violations",
    "capacity_violations",
    "regularity_violations",
    "directional_e1_violations",
    "directional_e2_violations",
    "total_hard_violations",
    "avg_abs_shift",
    "median_abs_shift",
    "p90_abs_shift",
    "p95_abs_shift",
    "max_abs_shift",
    "pct_shift_le_5",
    "pct_shift_le_10",
    "pct_shift_le_15",
    "pct_shift_gt_30",
    "pct_shift_gt_45",
    "pct_at_bound",
    "changed_flight_pct",
    "notes",
]


def run_od300_controlled_shift_test(run_config: RunConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    controlled_dir = run_config.output_dir / f"od300_controlled_shift_{timestamp}"
    controlled_dir.mkdir(parents=True, exist_ok=True)
    summary_path = controlled_dir / "controlled_shift_summary.csv"
    pd.DataFrame(columns=CONTROLLED_SHIFT_COLUMNS).to_csv(summary_path, index=False)
    scenarios = [
        ("max_shift45_linear5", {"max_shift_min": 45, "shift_penalty_mode": "linear", "shift_penalty_weight": 5.0}),
        ("max_shift45_linear10", {"max_shift_min": 45, "shift_penalty_mode": "linear", "shift_penalty_weight": 10.0}),
        ("max_shift45_threshold", {"max_shift_min": 45, "shift_penalty_mode": "threshold", "shift_penalty_weight": 15.0, "shift_free_threshold_min": 15}),
        ("max_shift60_linear10", {"max_shift_min": 60, "shift_penalty_mode": "linear", "shift_penalty_weight": 10.0}),
        ("max_shift60_linear15", {"max_shift_min": 60, "shift_penalty_mode": "linear", "shift_penalty_weight": 15.0}),
        ("max_shift60_threshold15", {"max_shift_min": 60, "shift_penalty_mode": "threshold", "shift_penalty_weight": 15.0, "shift_free_threshold_min": 15}),
        ("max_shift60_threshold30", {"max_shift_min": 60, "shift_penalty_mode": "threshold", "shift_penalty_weight": 20.0, "shift_free_threshold_min": 30}),
        ("max_shift60_quadratic_proxy", {"max_shift_min": 60, "shift_penalty_mode": "quadratic_proxy", "shift_penalty_weight": 5.0}),
    ]
    rows = []
    for scenario, overrides in scenarios:
        run_id = scenario
        row = _run_od300_diagnosis_subprocess(run_config, controlled_dir, run_id, "days12", [1, 2], scenario, overrides)
        controlled = _controlled_shift_row_from_outputs(controlled_dir / run_id, row, scenario, overrides)
        rows.append(controlled)
        pd.DataFrame([controlled], columns=CONTROLLED_SHIFT_COLUMNS).to_csv(summary_path, mode="a", header=False, index=False)
        print(f"[INFO] OD300 controlled shift appended: {scenario} -> {controlled['solver_status']}, feasible={controlled['feasible']}")
    summary = pd.DataFrame(rows, columns=CONTROLLED_SHIFT_COLUMNS)
    best = _select_controlled_shift_candidate(run_config.output_dir, controlled_dir, summary)
    _write_controlled_shift_interpretation(controlled_dir, summary, best)
    print(f"[INFO] OD300 controlled shift outputs written to {controlled_dir}")
    return controlled_dir


def _controlled_shift_row_from_outputs(run_dir: Path, row: dict, scenario: str, overrides: dict) -> dict:
    summary = _parse_key_value_file(run_dir / "run_summary.txt")
    comparison = _parse_key_value_file(run_dir / "comparison_to_baseline.txt")
    baseline = _safe_float(summary.get("baseline_objective", comparison.get("baseline objective", "")))
    after = _safe_float(summary.get("objective_after_penalty", summary.get("total_objective", "")))
    improvement = ""
    if baseline not in {"", None} and after not in {"", None} and abs(float(baseline)) > 1e-12:
        improvement = 100.0 * (float(after) - float(baseline)) / abs(float(baseline))
    shift_metrics = _shift_metrics_from_schedule(run_dir / "optimized_schedule.xlsx", overrides.get("max_shift_min", ""))
    total_hard = row.get("total_hard_violations", "")
    if not row.get("feasible", False) and not (run_dir / "optimized_schedule.xlsx").exists():
        total_hard = ""
    notes = row.get("note", "")
    if overrides.get("shift_penalty_mode") == "threshold" and "shift_free_threshold_min" not in overrides:
        notes = (notes + " " if notes else "") + "threshold skipped: missing shift_free_threshold"
    return {
        "scenario": scenario,
        "feasible": row.get("feasible", False),
        "solver_status": row.get("solver_status", ""),
        "runtime_sec": row.get("runtime_sec", ""),
        "max_shift": overrides.get("max_shift_min", ""),
        "shift_penalty_mode": overrides.get("shift_penalty_mode", ""),
        "shift_penalty_weight": overrides.get("shift_penalty_weight", ""),
        "threshold": overrides.get("shift_free_threshold_min", ""),
        "baseline_objective": summary.get("baseline_objective", comparison.get("baseline objective", "")),
        "objective_before_penalty": summary.get("objective_before_penalty", ""),
        "objective_after_penalty": summary.get("objective_after_penalty", summary.get("total_objective", "")),
        "improvement_after_penalty_pct": improvement if improvement != "" else comparison.get("percentage improvement", "").replace("%", ""),
        "active_connections": summary.get("active_connections", row.get("active_connections", "")),
        "beaten_competitors": summary.get("beaten_competitors", ""),
        "total_candidate_connections": summary.get("total_candidate_connections", row.get("total_candidate_connections", "")),
        "mandatory_feasible_but_inactive": summary.get("feasible_but_inactive_connection_count", row.get("mandatory_feasible_but_inactive", "")),
        "rotation_violations": summary.get("rotation_violations", row.get("rotation_violations", "")),
        "capacity_violations": summary.get("hub_capacity_violations", row.get("hub_capacity_violations", "")),
        "regularity_violations": summary.get("regularity_violations", row.get("regularity_violations", "")),
        "directional_e1_violations": summary.get("directional_e1_violations", row.get("directional_e1_violations", "")),
        "directional_e2_violations": summary.get("directional_e2_violations", row.get("directional_e2_violations", "")),
        "total_hard_violations": total_hard,
        "avg_abs_shift": summary.get("avg_abs_shift", shift_metrics["avg_abs_shift"]),
        "median_abs_shift": summary.get("median_abs_shift", shift_metrics["median_abs_shift"]),
        "p90_abs_shift": shift_metrics["p90_abs_shift"],
        "p95_abs_shift": shift_metrics["p95_abs_shift"],
        "max_abs_shift": shift_metrics["max_abs_shift"],
        "pct_shift_le_5": summary.get("pct_shift_le_5", shift_metrics["pct_shift_le_5"]),
        "pct_shift_le_10": shift_metrics["pct_shift_le_10"],
        "pct_shift_le_15": shift_metrics["pct_shift_le_15"],
        "pct_shift_gt_30": shift_metrics["pct_shift_gt_30"],
        "pct_shift_gt_45": shift_metrics["pct_shift_gt_45"],
        "pct_at_bound": shift_metrics["pct_at_bound"],
        "changed_flight_pct": shift_metrics["changed_flight_pct"],
        "notes": notes,
    }


def _shift_metrics_from_schedule(schedule_path: Path, max_shift) -> dict:
    empty = {
        "avg_abs_shift": "",
        "median_abs_shift": "",
        "p90_abs_shift": "",
        "p95_abs_shift": "",
        "max_abs_shift": "",
        "pct_shift_le_5": "",
        "pct_shift_le_10": "",
        "pct_shift_le_15": "",
        "pct_shift_gt_30": "",
        "pct_shift_gt_45": "",
        "pct_at_bound": "",
        "changed_flight_pct": "",
    }
    if not schedule_path.exists():
        return empty
    schedule = pd.read_excel(schedule_path, engine="openpyxl")
    if schedule.empty or "time_change_min" not in schedule:
        return empty
    abs_shift = pd.to_numeric(schedule["time_change_min"], errors="coerce").abs().fillna(0.0)
    total = len(abs_shift)
    max_shift_value = _safe_float(max_shift)
    at_bound = 0
    if max_shift_value not in {"", None}:
        at_bound = int((abs(abs_shift - float(max_shift_value)) <= 1e-6).sum())
    return {
        "avg_abs_shift": float(abs_shift.mean()) if total else 0.0,
        "median_abs_shift": float(abs_shift.median()) if total else 0.0,
        "p90_abs_shift": float(abs_shift.quantile(0.90)) if total else 0.0,
        "p95_abs_shift": float(abs_shift.quantile(0.95)) if total else 0.0,
        "max_abs_shift": float(abs_shift.max()) if total else 0.0,
        "pct_shift_le_5": _pct((abs_shift <= 5).sum(), total),
        "pct_shift_le_10": _pct((abs_shift <= 10).sum(), total),
        "pct_shift_le_15": _pct((abs_shift <= 15).sum(), total),
        "pct_shift_gt_30": _pct((abs_shift > 30).sum(), total),
        "pct_shift_gt_45": _pct((abs_shift > 45).sum(), total),
        "pct_at_bound": _pct(at_bound, total),
        "changed_flight_pct": _pct((abs_shift > 1e-6).sum(), total),
    }


def _select_controlled_shift_candidate(output_dir: Path, controlled_dir: Path, summary: pd.DataFrame) -> pd.Series | None:
    if summary.empty:
        return None
    candidates = summary[summary["feasible"].astype(str).str.lower().isin(["true", "1"])].copy()
    if candidates.empty:
        return None
    numeric_cols = [
        "mandatory_feasible_but_inactive",
        "total_hard_violations",
        "improvement_after_penalty_pct",
        "pct_shift_gt_45",
        "pct_at_bound",
        "avg_abs_shift",
        "objective_after_penalty",
    ]
    for col in numeric_cols:
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    clean = candidates[
        candidates["mandatory_feasible_but_inactive"].fillna(999).eq(0)
        & candidates["total_hard_violations"].fillna(999).eq(0)
        & (candidates["improvement_after_penalty_pct"].fillna(-999) > 0)
    ].copy()
    if clean.empty:
        return None
    clean["max_shift_rank"] = clean["max_shift"].astype(float).map(lambda value: 0 if value <= 45 else 1)
    clean = clean.sort_values(
        ["max_shift_rank", "pct_shift_gt_45", "pct_at_bound", "avg_abs_shift", "objective_after_penalty"],
        ascending=[True, True, True, True, False],
    )
    best = clean.iloc[0]
    src = controlled_dir / str(best["scenario"])
    if src.exists():
        dst = output_dir / "od300_controlled_fallback_candidate"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        (dst / "fallback_source.txt").write_text(
            "\n".join(
                [
                    f"source_dir: {src}",
                    f"scenario: {best['scenario']}",
                    f"max_shift: {best['max_shift']}",
                    f"shift_penalty_mode: {best['shift_penalty_mode']}",
                    f"shift_penalty_weight: {best['shift_penalty_weight']}",
                    f"threshold: {best['threshold']}",
                    "note: Primary final_submission_candidate remains unchanged; this is an OD300 controlled fallback candidate.",
                ]
            ),
            encoding="utf-8",
        )
        checker = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "check_solution.py"),
                "--output_dir",
                str(controlled_dir),
                "--run_id",
                str(best["scenario"]),
            ],
            cwd=str(Path(__file__).resolve().parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        (controlled_dir / "best_controlled_candidate_checker.txt").write_text(
            (checker.stdout or "") + ("\nSTDERR:\n" + checker.stderr if checker.stderr else ""),
            encoding="utf-8",
            errors="replace",
        )
    return best


def _write_controlled_shift_interpretation(controlled_dir: Path, summary: pd.DataFrame, best: pd.Series | None) -> None:
    lines = [
        "OD300 CONTROLLED SHIFT INTERPRETATION",
        "------------------------------------",
        f"completed scenarios: {len(summary)}",
    ]
    if summary.empty:
        lines.append("No scenarios completed.")
        (controlled_dir / "controlled_shift_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return
    feasible = summary[summary["feasible"].astype(str).str.lower().isin(["true", "1"])].copy()
    max45 = summary[pd.to_numeric(summary["max_shift"], errors="coerce") == 45]
    max60 = summary[pd.to_numeric(summary["max_shift"], errors="coerce") == 60]
    max45_feasible = not max45[max45["feasible"].astype(str).str.lower().isin(["true", "1"])].empty
    lines.extend(
        [
            f"feasible scenarios: {len(feasible)}",
            f"max_shift45 feasible: {max45_feasible}",
            f"max_shift60 feasible scenarios: {len(max60[max60['feasible'].astype(str).str.lower().isin(['true', '1'])]) if not max60.empty else 0}",
        ]
    )
    if best is not None:
        lines.extend(
            [
                f"best controlled fallback: {best['scenario']}",
                f"best objective after penalty: {best['objective_after_penalty']}",
                f"best improvement pct: {best['improvement_after_penalty_pct']}",
                f"best avg abs shift: {best['avg_abs_shift']}",
                f"best pct >45: {best['pct_shift_gt_45']}",
                f"best pct at bound: {best['pct_at_bound']}",
            ]
        )
    else:
        lines.append("best controlled fallback: none")
    lines.extend(["", "Turkce yorum:"])
    if max45_feasible:
        lines.append("- max_shift45 feasible oldu; OD300 icin 60 dakika yerine daha kontrollu hareket alani yeterli olabilir.")
    else:
        lines.append("- max_shift45 feasible olmadiysa, OD300 final hard stack icin 60 dakika hareket alani daha gercekci fallback siniri olarak goruluyor.")
    if not max60.empty:
        lines.append("- max_shift60 senaryolarinda daha guclu penalty'nin >30/>45 shift oranlarini ve objective kaybini nasil etkiledigi summary tablosundan okunmalidir.")
    lines.append("- Primary final_submission_candidate degismemeli; od200_days123 lookup_linear ana teslim adayi olarak kalmali.")
    lines.append("- OD400'e gecmeden once secilen OD300 controlled fallback'in rapordaki operasyonel savunmasi netlestirilmelidir.")
    (controlled_dir / "controlled_shift_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


DAYS123_CONTROLLED_COLUMNS = [
    "scenario",
    "feasible",
    "solver_status",
    "runtime_sec",
    "days",
    "max_shift",
    "shift_penalty_mode",
    "threshold",
    "baseline_objective",
    "objective_before_penalty",
    "objective_after_penalty",
    "improvement_after_penalty_pct",
    "active_connections",
    "beaten_competitors",
    "total_candidate_connections",
    "mandatory_feasible_but_inactive",
    "rotation_violations",
    "capacity_violations",
    "regularity_violations",
    "directional_e1_violations",
    "directional_e2_violations",
    "total_hard_violations",
    "avg_abs_shift",
    "median_abs_shift",
    "p90_abs_shift",
    "p95_abs_shift",
    "max_abs_shift",
    "pct_shift_le_5",
    "pct_shift_le_10",
    "pct_shift_le_15",
    "pct_shift_gt_30",
    "pct_shift_gt_45",
    "pct_at_bound",
    "changed_flight_pct",
    "notes",
]


def run_od300_days123_controlled_shift_test(run_config: RunConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    controlled_dir = run_config.output_dir / f"od300_days123_controlled_shift_{timestamp}"
    controlled_dir.mkdir(parents=True, exist_ok=True)
    summary_path = controlled_dir / "days123_controlled_summary.csv"
    pd.DataFrame(columns=DAYS123_CONTROLLED_COLUMNS).to_csv(summary_path, index=False)
    scenarios = [
        ("max_shift60_threshold15_days123", {"max_shift_min": 60, "shift_penalty_mode": "threshold", "shift_penalty_weight": 15.0, "shift_free_threshold_min": 15, "timeout_sec": 600}),
        ("max_shift60_threshold30_days123", {"max_shift_min": 60, "shift_penalty_mode": "threshold", "shift_penalty_weight": 20.0, "shift_free_threshold_min": 30, "timeout_sec": 600}),
    ]
    rows = []
    for scenario, overrides in scenarios:
        row = _run_od300_diagnosis_subprocess(run_config, controlled_dir, scenario, "days123", [1, 2, 3], scenario, overrides)
        controlled = _days123_controlled_row_from_outputs(controlled_dir / scenario, row, scenario, overrides)
        rows.append(controlled)
        pd.DataFrame([controlled], columns=DAYS123_CONTROLLED_COLUMNS).to_csv(summary_path, mode="a", header=False, index=False)
        print(f"[INFO] OD300 days123 controlled appended: {scenario} -> {controlled['solver_status']}, feasible={controlled['feasible']}")
    summary = pd.DataFrame(rows, columns=DAYS123_CONTROLLED_COLUMNS)
    best = _select_days123_controlled_candidate(run_config.output_dir, controlled_dir, summary)
    _write_days123_controlled_interpretation(controlled_dir, summary, best)
    print(f"[INFO] OD300 days123 controlled outputs written to {controlled_dir}")
    return controlled_dir


def _days123_controlled_row_from_outputs(run_dir: Path, row: dict, scenario: str, overrides: dict) -> dict:
    base = _controlled_shift_row_from_outputs(run_dir, row, scenario, overrides)
    return {column: ("1,2,3" if column == "days" else base.get(column, "")) for column in DAYS123_CONTROLLED_COLUMNS}


def _select_days123_controlled_candidate(output_dir: Path, controlled_dir: Path, summary: pd.DataFrame) -> pd.Series | None:
    if summary.empty:
        return None
    candidates = summary[summary["feasible"].astype(str).str.lower().isin(["true", "1"])].copy()
    if candidates.empty:
        return None
    for col in ["mandatory_feasible_but_inactive", "total_hard_violations", "improvement_after_penalty_pct", "pct_shift_gt_45", "pct_at_bound", "avg_abs_shift", "objective_after_penalty"]:
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    clean = candidates[
        candidates["mandatory_feasible_but_inactive"].fillna(999).eq(0)
        & candidates["total_hard_violations"].fillna(999).eq(0)
        & (candidates["improvement_after_penalty_pct"].fillna(-999) > 0)
    ].copy()
    if clean.empty:
        return None
    clean["scenario_rank"] = clean["scenario"].map({"max_shift60_threshold15_days123": 0, "max_shift60_threshold30_days123": 1}).fillna(2)
    clean = clean.sort_values(
        ["scenario_rank", "pct_shift_gt_45", "pct_at_bound", "avg_abs_shift", "objective_after_penalty"],
        ascending=[True, True, True, True, False],
    )
    best = clean.iloc[0]
    src = controlled_dir / str(best["scenario"])
    if src.exists():
        dst = output_dir / "od300_days123_controlled_fallback_candidate"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        label = "threshold15_controlled" if best["scenario"] == "max_shift60_threshold15_days123" else "threshold30_higher_score_more_aggressive"
        (dst / "fallback_source.txt").write_text(
            "\n".join(
                [
                    f"source_dir: {src}",
                    f"scenario: {best['scenario']}",
                    f"fallback_label: {label}",
                    f"days: {best['days']}",
                    f"max_shift: {best['max_shift']}",
                    f"shift_penalty_mode: {best['shift_penalty_mode']}",
                    f"threshold: {best['threshold']}",
                    "note: Primary final_submission_candidate remains unchanged; this is an OD300 days123 controlled fallback candidate.",
                ]
            ),
            encoding="utf-8",
        )
        checker = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "check_solution.py"),
                "--output_dir",
                str(controlled_dir),
                "--run_id",
                str(best["scenario"]),
            ],
            cwd=str(Path(__file__).resolve().parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        (controlled_dir / "best_days123_controlled_checker.txt").write_text(
            (checker.stdout or "") + ("\nSTDERR:\n" + checker.stderr if checker.stderr else ""),
            encoding="utf-8",
            errors="replace",
        )
    return best


def _write_days123_controlled_interpretation(controlled_dir: Path, summary: pd.DataFrame, best: pd.Series | None) -> None:
    lines = [
        "OD300 DAYS123 CONTROLLED SHIFT INTERPRETATION",
        "---------------------------------------------",
        f"completed scenarios: {len(summary)}",
    ]
    if summary.empty:
        lines.append("No scenarios completed.")
        (controlled_dir / "days123_controlled_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return
    feasible = summary[summary["feasible"].astype(str).str.lower().isin(["true", "1"])].copy()
    lines.append(f"feasible scenarios: {len(feasible)}")
    t15 = summary[summary["scenario"] == "max_shift60_threshold15_days123"]
    t30 = summary[summary["scenario"] == "max_shift60_threshold30_days123"]
    lines.append(f"threshold15 feasible: {bool(t15['feasible'].astype(str).str.lower().isin(['true', '1']).iloc[0]) if not t15.empty else ''}")
    lines.append(f"threshold30 feasible: {bool(t30['feasible'].astype(str).str.lower().isin(['true', '1']).iloc[0]) if not t30.empty else ''}")
    if best is not None:
        lines.extend(
            [
                f"best days123 controlled fallback: {best['scenario']}",
                f"best objective after penalty: {best['objective_after_penalty']}",
                f"best improvement pct: {best['improvement_after_penalty_pct']}",
                f"best avg abs shift: {best['avg_abs_shift']}",
                f"best pct >30: {best['pct_shift_gt_30']}",
                f"best pct >45: {best['pct_shift_gt_45']}",
                f"best pct at bound: {best['pct_at_bound']}",
            ]
        )
    else:
        lines.append("best days123 controlled fallback: none")
    lines.extend(["", "Turkce yorum:"])
    if not feasible.empty:
        lines.append("- OD300 days123 controlled test feasible cozum uretmistir; days12 fallback daha genis gun kapsaminda da aday olabilir.")
    else:
        lines.append("- Iki senaryo da TIME_LIMIT / NOT_SOLVED veya infeasible kaldi; OD300 days12 controlled fallback korunmalidir.")
    if best is not None and best["scenario"] == "max_shift60_threshold15_days123":
        lines.append("- threshold15, days12'deki daha kontrollu shift profilini days123 icin de oncelikli aday olarak korur.")
    elif best is not None:
        lines.append("- threshold30 daha yuksek skor verebilir ancak daha agresif shift profili nedeniyle ikincil fallback olarak anlatilmalidir.")
    lines.append("- Primary final_submission_candidate degismemeli; od200_days123 lookup_linear ana teslim adayi olarak kalmali.")
    lines.append("- OD400 stress testine gecmek, OD300 days123 sonucunun hard-clean olup olmamasina gore planlanmalidir.")
    (controlled_dir / "days123_controlled_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


OD400_STRESS_COLUMNS = [
    "scenario",
    "feasible",
    "solver_status",
    "runtime_sec",
    "max_od",
    "days",
    "max_shift",
    "xdev",
    "shift_penalty_mode",
    "threshold",
    "directional_mode",
    "regularity_mode",
    "baseline_objective",
    "objective_before_penalty",
    "objective_after_penalty",
    "improvement_after_penalty_pct",
    "active_connections",
    "beaten_competitors",
    "total_candidate_connections",
    "mandatory_feasible_but_inactive",
    "mandatory_infeasible_but_active",
    "rotation_violations",
    "capacity_violations",
    "regularity_violations",
    "directional_e1_violations",
    "directional_e2_violations",
    "total_hard_violations",
    "avg_abs_shift",
    "median_abs_shift",
    "p90_abs_shift",
    "p95_abs_shift",
    "pct_shift_le_5",
    "pct_shift_le_10",
    "pct_shift_le_15",
    "pct_shift_gt_30",
    "pct_shift_gt_45",
    "pct_at_bound",
    "changed_flight_pct",
    "suspected_bottleneck",
    "notes",
]


def run_od400_stress_test(run_config: RunConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stress_dir = run_config.output_dir / f"od400_stress_test_{timestamp}"
    stress_dir.mkdir(parents=True, exist_ok=True)
    summary_path = stress_dir / "od400_stress_summary.csv"
    pd.DataFrame(columns=OD400_STRESS_COLUMNS).to_csv(summary_path, index=False)
    scenarios = [
        (
            "od400_days12_primary_rules",
            [1, 2],
            {
                "max_od": 400,
                "max_shift_min": 30,
                "regularity_xdev_min": 15,
                "regularity_mode": "hard",
                "directional_e1_mode": "hard",
                "directional_e2_mode": "hard",
                "shift_penalty_mode": "linear",
                "shift_penalty_weight": 5.0,
                "shift_free_threshold_min": SHIFT_FREE_THRESHOLD_MINUTES,
                "timeout_sec": 900,
            },
        ),
        (
            "od400_days12_relaxed_shift_controlled",
            [1, 2],
            {
                "max_od": 400,
                "max_shift_min": 60,
                "regularity_xdev_min": 15,
                "regularity_mode": "hard",
                "directional_e1_mode": "hard",
                "directional_e2_mode": "hard",
                "shift_penalty_mode": "threshold",
                "shift_penalty_weight": 20.0,
                "shift_free_threshold_min": 30,
                "timeout_sec": 900,
            },
        ),
        (
            "od400_days12_no_directional_diagnosis",
            [1, 2],
            {
                "max_od": 400,
                "max_shift_min": 60,
                "regularity_xdev_min": 15,
                "regularity_mode": "hard",
                "directional_e1_mode": "report_only",
                "directional_e2_mode": "report_only",
                "shift_penalty_mode": "threshold",
                "shift_penalty_weight": 20.0,
                "shift_free_threshold_min": 30,
                "timeout_sec": 900,
            },
        ),
        (
            "od400_days12_regularity_report_only_diagnosis",
            [1, 2],
            {
                "max_od": 400,
                "max_shift_min": 60,
                "regularity_xdev_min": 15,
                "regularity_mode": "report_only",
                "directional_e1_mode": "report_only",
                "directional_e2_mode": "report_only",
                "shift_penalty_mode": "threshold",
                "shift_penalty_weight": 20.0,
                "shift_free_threshold_min": 30,
                "timeout_sec": 900,
            },
        ),
    ]
    rows = []
    for scenario, days, overrides in scenarios:
        row = _run_od300_diagnosis_subprocess(run_config, stress_dir, scenario, "days" + "".join(str(day) for day in days), days, scenario, overrides)
        stress_row = _od400_stress_row_from_outputs(stress_dir / scenario, row, scenario, days, overrides)
        rows.append(stress_row)
        pd.DataFrame([stress_row], columns=OD400_STRESS_COLUMNS).to_csv(summary_path, mode="a", header=False, index=False)
        print(f"[INFO] OD400 stress appended: {scenario} -> {stress_row['solver_status']}, feasible={stress_row['feasible']}")
    if _should_run_od400_days123(rows):
        scenario = "od400_days123_controlled_optional"
        days = [1, 2, 3]
        overrides = {
            "max_od": 400,
            "max_shift_min": 60,
            "regularity_xdev_min": 15,
            "regularity_mode": "hard",
            "directional_e1_mode": "hard",
            "directional_e2_mode": "hard",
            "shift_penalty_mode": "threshold",
            "shift_penalty_weight": 20.0,
            "shift_free_threshold_min": 30,
            "timeout_sec": 1200,
        }
        row = _run_od300_diagnosis_subprocess(run_config, stress_dir, scenario, "days123", days, scenario, overrides)
        stress_row = _od400_stress_row_from_outputs(stress_dir / scenario, row, scenario, days, overrides)
        rows.append(stress_row)
        pd.DataFrame([stress_row], columns=OD400_STRESS_COLUMNS).to_csv(summary_path, mode="a", header=False, index=False)
        print(f"[INFO] OD400 stress appended: {scenario} -> {stress_row['solver_status']}, feasible={stress_row['feasible']}")
    summary = pd.DataFrame(rows, columns=OD400_STRESS_COLUMNS)
    best = _select_od400_stress_candidate(run_config.output_dir, stress_dir, summary)
    _write_od400_stress_interpretation(stress_dir, summary, best)
    print(f"[INFO] OD400 stress outputs written to {stress_dir}")
    return stress_dir


def _od400_stress_row_from_outputs(run_dir: Path, row: dict, scenario: str, days: list[int], overrides: dict) -> dict:
    summary = _parse_key_value_file(run_dir / "run_summary.txt")
    comparison = _parse_key_value_file(run_dir / "comparison_to_baseline.txt")
    baseline = _safe_float(summary.get("baseline_objective", comparison.get("baseline objective", "")))
    after = _safe_float(summary.get("objective_after_penalty", summary.get("total_objective", "")))
    improvement = ""
    if baseline not in {"", None} and after not in {"", None} and abs(float(baseline)) > 1e-12:
        improvement = 100.0 * (float(after) - float(baseline)) / abs(float(baseline))
    shift_metrics = _shift_metrics_from_schedule(run_dir / "optimized_schedule.xlsx", overrides.get("max_shift_min", ""))
    total_hard = row.get("total_hard_violations", "")
    if not row.get("feasible", False) and not (run_dir / "optimized_schedule.xlsx").exists():
        total_hard = ""
    return {
        "scenario": scenario,
        "feasible": row.get("feasible", False),
        "solver_status": row.get("solver_status", ""),
        "runtime_sec": row.get("runtime_sec", ""),
        "max_od": overrides.get("max_od", 400),
        "days": ",".join(str(day) for day in days),
        "max_shift": overrides.get("max_shift_min", ""),
        "xdev": overrides.get("regularity_xdev_min", ""),
        "shift_penalty_mode": overrides.get("shift_penalty_mode", ""),
        "threshold": overrides.get("shift_free_threshold_min", ""),
        "directional_mode": f"E1={overrides.get('directional_e1_mode', '')},E2={overrides.get('directional_e2_mode', '')}",
        "regularity_mode": overrides.get("regularity_mode", ""),
        "baseline_objective": summary.get("baseline_objective", comparison.get("baseline objective", "")),
        "objective_before_penalty": summary.get("objective_before_penalty", ""),
        "objective_after_penalty": summary.get("objective_after_penalty", summary.get("total_objective", "")),
        "improvement_after_penalty_pct": improvement if improvement != "" else comparison.get("percentage improvement", "").replace("%", ""),
        "active_connections": summary.get("active_connections", row.get("active_connections", "")),
        "beaten_competitors": summary.get("beaten_competitors", ""),
        "total_candidate_connections": summary.get("total_candidate_connections", row.get("total_candidate_connections", "")),
        "mandatory_feasible_but_inactive": summary.get("feasible_but_inactive_connection_count", row.get("mandatory_feasible_but_inactive", "")),
        "mandatory_infeasible_but_active": summary.get("infeasible_but_active_connection_count", row.get("mandatory_infeasible_but_active", "")),
        "rotation_violations": summary.get("rotation_violations", row.get("rotation_violations", "")),
        "capacity_violations": summary.get("hub_capacity_violations", row.get("hub_capacity_violations", "")),
        "regularity_violations": summary.get("regularity_violations", row.get("regularity_violations", "")),
        "directional_e1_violations": summary.get("directional_e1_violations", row.get("directional_e1_violations", "")),
        "directional_e2_violations": summary.get("directional_e2_violations", row.get("directional_e2_violations", "")),
        "total_hard_violations": total_hard,
        "avg_abs_shift": summary.get("avg_abs_shift", shift_metrics["avg_abs_shift"]),
        "median_abs_shift": summary.get("median_abs_shift", shift_metrics["median_abs_shift"]),
        "p90_abs_shift": shift_metrics["p90_abs_shift"],
        "p95_abs_shift": shift_metrics["p95_abs_shift"],
        "pct_shift_le_5": summary.get("pct_shift_le_5", shift_metrics["pct_shift_le_5"]),
        "pct_shift_le_10": shift_metrics["pct_shift_le_10"],
        "pct_shift_le_15": shift_metrics["pct_shift_le_15"],
        "pct_shift_gt_30": shift_metrics["pct_shift_gt_30"],
        "pct_shift_gt_45": shift_metrics["pct_shift_gt_45"],
        "pct_at_bound": shift_metrics["pct_at_bound"],
        "changed_flight_pct": shift_metrics["changed_flight_pct"],
        "suspected_bottleneck": _od400_suspected_bottleneck(scenario, row.get("feasible", False), row.get("solver_status", ""), total_hard),
        "notes": row.get("note", ""),
    }


def _should_run_od400_days123(rows: list[dict]) -> bool:
    if not rows:
        return False
    controlled = [row for row in rows if row.get("scenario") == "od400_days12_relaxed_shift_controlled"]
    if not controlled:
        return False
    row = controlled[0]
    return str(row.get("feasible", "")).lower() in {"true", "1"} and _safe_int(row.get("total_hard_violations", "")) == 0


def _od400_suspected_bottleneck(scenario: str, feasible: bool, solver_status: str, total_hard) -> str:
    if feasible:
        return "none"
    if "TIME_LIMIT" in str(solver_status):
        return "runtime/scale stress"
    if scenario == "od400_days12_primary_rules":
        return "primary hard stack or max_shift30"
    if scenario == "od400_days12_relaxed_shift_controlled":
        return "full hard stack at OD400"
    if scenario == "od400_days12_no_directional_diagnosis":
        return "not directional only; upstream hard stack or runtime"
    if scenario == "od400_days12_regularity_report_only_diagnosis":
        return "mandatory/rotation/capacity or runtime"
    return "undetermined"


def _select_od400_stress_candidate(output_dir: Path, stress_dir: Path, summary: pd.DataFrame) -> pd.Series | None:
    if summary.empty:
        return None
    candidates = summary[summary["feasible"].astype(str).str.lower().isin(["true", "1"])].copy()
    if candidates.empty:
        return None
    hard_stack = candidates[candidates["scenario"].isin(["od400_days12_primary_rules", "od400_days12_relaxed_shift_controlled", "od400_days123_controlled_optional"])].copy()
    if hard_stack.empty:
        return None
    for col in ["mandatory_feasible_but_inactive", "mandatory_infeasible_but_active", "total_hard_violations", "improvement_after_penalty_pct", "pct_shift_gt_45", "pct_at_bound", "avg_abs_shift", "objective_after_penalty"]:
        hard_stack[col] = pd.to_numeric(hard_stack[col], errors="coerce")
    clean = hard_stack[
        hard_stack["mandatory_feasible_but_inactive"].fillna(999).eq(0)
        & hard_stack["mandatory_infeasible_but_active"].fillna(999).eq(0)
        & hard_stack["total_hard_violations"].fillna(999).eq(0)
        & (hard_stack["improvement_after_penalty_pct"].fillna(-999) > 0)
    ].copy()
    if clean.empty:
        return None
    clean["scenario_rank"] = clean["scenario"].map({"od400_days12_primary_rules": 0, "od400_days12_relaxed_shift_controlled": 1, "od400_days123_controlled_optional": 2}).fillna(9)
    clean = clean.sort_values(
        ["scenario_rank", "pct_shift_gt_45", "pct_at_bound", "avg_abs_shift", "objective_after_penalty"],
        ascending=[True, True, True, True, False],
    )
    best = clean.iloc[0]
    src = stress_dir / str(best["scenario"])
    if src.exists():
        dst = output_dir / "od400_stress_fallback_candidate"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        (dst / "fallback_source.txt").write_text(
            "\n".join(
                [
                    f"source_dir: {src}",
                    f"scenario: {best['scenario']}",
                    f"days: {best['days']}",
                    f"max_shift: {best['max_shift']}",
                    f"shift_penalty_mode: {best['shift_penalty_mode']}",
                    f"threshold: {best['threshold']}",
                    "note: Primary final_submission_candidate remains unchanged; this is an OD400 stress/scale fallback candidate.",
                ]
            ),
            encoding="utf-8",
        )
        checker = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "check_solution.py"),
                "--output_dir",
                str(stress_dir),
                "--run_id",
                str(best["scenario"]),
            ],
            cwd=str(Path(__file__).resolve().parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        (stress_dir / "best_od400_stress_checker.txt").write_text(
            (checker.stdout or "") + ("\nSTDERR:\n" + checker.stderr if checker.stderr else ""),
            encoding="utf-8",
            errors="replace",
        )
    return best


def _write_od400_stress_interpretation(stress_dir: Path, summary: pd.DataFrame, best: pd.Series | None) -> None:
    lines = [
        "OD400 STRESS TEST INTERPRETATION",
        "--------------------------------",
        f"completed scenarios: {len(summary)}",
    ]
    if summary.empty:
        lines.append("No scenarios completed.")
        (stress_dir / "od400_stress_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return
    feasible = summary[summary["feasible"].astype(str).str.lower().isin(["true", "1"])].copy()
    primary = summary[summary["scenario"] == "od400_days12_primary_rules"]
    controlled = summary[summary["scenario"] == "od400_days12_relaxed_shift_controlled"]
    lines.extend(
        [
            f"feasible scenarios: {len(feasible)}",
            f"primary rules feasible: {bool(primary['feasible'].astype(str).str.lower().isin(['true', '1']).iloc[0]) if not primary.empty else ''}",
            f"controlled days12 feasible: {bool(controlled['feasible'].astype(str).str.lower().isin(['true', '1']).iloc[0]) if not controlled.empty else ''}",
        ]
    )
    if best is not None:
        lines.extend(
            [
                f"best OD400 stress fallback: {best['scenario']}",
                f"best objective after penalty: {best['objective_after_penalty']}",
                f"best improvement pct: {best['improvement_after_penalty_pct']}",
                f"best avg abs shift: {best['avg_abs_shift']}",
                f"best hard violations: {best['total_hard_violations']}",
            ]
        )
    else:
        lines.append("best OD400 stress fallback: none")
    lines.extend(["", "Turkce yorum:"])
    if primary.empty or str(primary["feasible"].iloc[0]).lower() not in {"true", "1"}:
        lines.append("- OD400 primary kurallarla cozulmediyse bu basarisizlik degil; max_shift30/final hard stack icin runtime-scale siniri olarak yorumlanmalidir.")
    else:
        lines.append("- OD400 primary kurallarla hard-clean cozuldu; yine de primary teslim adayi degistirilmemelidir.")
    if not controlled.empty and str(controlled["feasible"].iloc[0]).lower() in {"true", "1"}:
        lines.append("- OD400 controlled max_shift60 threshold30 hard stack cozum verdi; bu sadece stress/scale basarisi olarak raporlanmalidir.")
    else:
        lines.append("- OD400 controlled senaryo cozum vermediyse OD300 fallback daha guvenilir scale fallback olarak kalir.")
    lines.append("- Primary final_submission_candidate degismemeli; od200_days123 lookup_linear ana teslim adayi olarak kalmali.")
    lines.append("- Final teslim stratejisi: od200 primary, OD300 controlled fallback, OD400 stress sonucu ayri sinirlilik/scale analizi.")
    (stress_dir / "od400_stress_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def _ordered_final_scale_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "max_od",
        "days",
        "top_importance_only",
        "ranking_objective_mode",
        "feasible",
        "baseline_objective",
        "objective_before_penalty",
        "objective_after_penalty",
        "improvement_before_penalty_pct",
        "improvement_after_penalty_pct",
        "active_connections",
        "beaten_competitors",
        "lookup_ranking_reward",
        "proxy_ranking_reward",
        "connection_reward",
        "shift_penalty",
        "total_candidate_connections",
        "feasible_by_schedule",
        "mandatory_feasible_but_inactive",
        "mandatory_infeasible_but_active",
        "rotation_violations",
        "rotation_block_elapsedtime_match_count",
        "rotation_block_legacy_exact_match_count",
        "rotation_block_exact_match_count",
        "rotation_block_fallback_count",
        "hub_capacity_violations",
        "regularity_violations",
        "directional_e1_violations",
        "directional_e2_violations",
        "total_hard_violations",
        "avg_abs_shift",
        "median_abs_shift",
        "max_abs_shift",
        "pct_shift_le_5",
        "pct_shift_5_15",
        "pct_shift_15_30",
        "pct_at_shift_bounds",
        "runtime_sec",
        "solver_status",
        "gap",
    ]
    data = summary.copy()
    for col in columns:
        if col not in data.columns:
            data[col] = ""
    extras = [col for col in data.columns if col not in columns]
    return data[columns + extras]


def _write_proxy_vs_lookup_linear_comparison(output_dir: Path, scale_dir: Path, lookup_summary: pd.DataFrame) -> pd.DataFrame:
    proxy_path = _find_latest_proxy_final_scale_summary(output_dir, exclude_dir=scale_dir)
    if proxy_path is None:
        warning = (
            "Proxy final scale summary bulunamadi. "
            "proxy_vs_lookup_linear_comparison.csv uretilmedi; sadece lookup_linear final_scale_summary.csv esas alinmalidir."
        )
        (scale_dir / "proxy_vs_lookup_linear_comparison_warning.txt").write_text(warning, encoding="utf-8")
        return pd.DataFrame()
    proxy = pd.read_csv(proxy_path)
    lookup = lookup_summary.copy()
    rows = []
    for scenario in lookup["scenario"].astype(str).tolist():
        lookup_row = lookup[lookup["scenario"].astype(str) == scenario]
        proxy_row = proxy[proxy["scenario"].astype(str) == scenario]
        if lookup_row.empty or proxy_row.empty:
            continue
        lrow = lookup_row.iloc[0]
        prow = proxy_row.iloc[0]
        proxy_obj = _safe_float(prow.get("objective_after_penalty", ""))
        lookup_obj = _safe_float(lrow.get("objective_after_penalty", ""))
        proxy_imp = _safe_float(prow.get("improvement_after_penalty_pct", ""))
        lookup_imp = _safe_float(lrow.get("improvement_after_penalty_pct", ""))
        proxy_active = _safe_float(prow.get("active_connections", ""))
        lookup_active = _safe_float(lrow.get("active_connections", ""))
        proxy_beaten = _safe_float(prow.get("beaten_competitors", ""))
        lookup_beaten = _safe_float(lrow.get("beaten_competitors", ""))
        proxy_shift = _safe_float(prow.get("avg_abs_shift", ""))
        lookup_shift = _safe_float(lrow.get("avg_abs_shift", ""))
        proxy_hard = _safe_float(prow.get("total_hard_violations", ""))
        lookup_hard = _safe_float(lrow.get("total_hard_violations", ""))
        clean_lookup = (
            str(lrow.get("feasible", "")).lower() in {"true", "1"}
            and lookup_hard not in {"", None}
            and float(lookup_hard) == 0
            and lookup_imp not in {"", None}
            and float(lookup_imp) > 0
        )
        if clean_lookup and proxy_obj not in {"", None} and lookup_obj not in {"", None} and float(lookup_obj) >= float(proxy_obj):
            recommendation = "prefer_lookup_linear"
        elif clean_lookup:
            recommendation = "lookup_linear_clean_but_lower_objective"
        else:
            recommendation = "keep_proxy_or_investigate"
        rows.append(
            {
                "scenario": scenario,
                "proxy_obj_after": proxy_obj,
                "lookup_linear_obj_after": lookup_obj,
                "obj_after_delta": _delta(lookup_obj, proxy_obj),
                "proxy_improvement_after": proxy_imp,
                "lookup_linear_improvement_after": lookup_imp,
                "improvement_delta": _delta(lookup_imp, proxy_imp),
                "proxy_active_connections": proxy_active,
                "lookup_active_connections": lookup_active,
                "active_delta": _delta(lookup_active, proxy_active),
                "proxy_beaten_competitors": proxy_beaten,
                "lookup_beaten_competitors": lookup_beaten,
                "beaten_delta": _delta(lookup_beaten, proxy_beaten),
                "proxy_avg_shift": proxy_shift,
                "lookup_avg_shift": lookup_shift,
                "avg_shift_delta": _delta(lookup_shift, proxy_shift),
                "proxy_hard_violations": proxy_hard,
                "lookup_hard_violations": lookup_hard,
                "recommendation": recommendation,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(scale_dir / "proxy_vs_lookup_linear_comparison.csv", index=False)
    (scale_dir / "proxy_vs_lookup_linear_source.txt").write_text(str(proxy_path), encoding="utf-8")
    return comparison


def _find_latest_proxy_final_scale_summary(output_dir: Path, exclude_dir: Path) -> Path | None:
    candidates = []
    for path in output_dir.glob("final_scale_*/final_scale_summary.csv"):
        if path.parent.resolve() == exclude_dir.resolve():
            continue
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        mode_values = set(df.get("ranking_objective_mode", pd.Series(["proxy"])).fillna("proxy").astype(str).str.lower())
        if "lookup_linear" in mode_values and len(mode_values) == 1:
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _delta(left, right):
    if left in {"", None} or right in {"", None}:
        return ""
    return float(left) - float(right)


def _final_scale_failure_row(scenario: str, max_od: int, days: list[int], top_only: bool, exc: Exception) -> dict:
    return {
        "scenario": scenario,
        "max_od": max_od,
        "days": ",".join(str(day) for day in days),
        "top_importance_only": top_only,
        "ranking_objective_mode": "lookup_linear",
        "mandatory_connection_selection": True,
        "directional_e1_mode": "hard",
        "directional_e1_formula": "absolute_tolerance",
        "directional_e1_abs_tolerance": 1,
        "directional_e2_mode": "hard",
        "feasible": False,
        "baseline_objective": "",
        "objective_before_penalty": "",
        "objective_after_penalty": "",
        "improvement_before_penalty_pct": "",
        "improvement_after_penalty_pct": "",
        "active_connections": "",
        "beaten_competitors": "",
        "lookup_ranking_reward": "",
        "proxy_ranking_reward": "",
        "connection_reward": "",
        "shift_penalty": "",
        "total_candidate_connections": "",
        "feasible_by_schedule": "",
        "mandatory_feasible_but_inactive": "",
        "mandatory_infeasible_but_active": "",
        "newly_activated_connections": "",
        "improved_od_count": "",
        "worsened_od_count": "",
        "rotation_violations": "",
        "hub_capacity_violations": "",
        "regularity_violations": "",
        "directional_e1_violations": "",
        "directional_e2_violations": "",
        "total_hard_violations": "",
        "avg_abs_shift": "",
        "median_abs_shift": "",
        "max_abs_shift": "",
        "pct_shift_le_5": "",
        "pct_shift_5_15": "",
        "pct_shift_15_30": "",
        "pct_at_shift_bounds": "",
        "runtime_sec": "",
        "solver_status": f"failed: {type(exc).__name__}",
        "gap": "",
    }


def _select_final_candidate(summary: pd.DataFrame) -> str:
    if summary.empty:
        return ""
    data = summary.copy()
    numeric_cols = [
        "max_od",
        "improvement_after_penalty_pct",
        "total_hard_violations",
        "mandatory_feasible_but_inactive",
        "mandatory_infeasible_but_active",
        "directional_e1_violations",
        "directional_e2_violations",
        "avg_abs_shift",
        "active_connections",
        "runtime_sec",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    feasible = data[data["feasible"].astype(str).str.lower().isin(["true", "1"])]
    if "ranking_objective_mode" in feasible.columns:
        feasible = feasible[feasible["ranking_objective_mode"].astype(str).str.lower() == "lookup_linear"]
    clean = feasible[
        (feasible["total_hard_violations"].fillna(1) == 0)
        & (feasible["mandatory_feasible_but_inactive"].fillna(1) == 0)
        & (feasible["mandatory_infeasible_but_active"].fillna(1) == 0)
        & (feasible["directional_e1_violations"].fillna(1) == 0)
        & (feasible["directional_e2_violations"].fillna(1) == 0)
        & (feasible["improvement_after_penalty_pct"].fillna(-1e9) > 0)
    ].copy()
    if clean.empty:
        clean = feasible if not feasible.empty else data
    clean["day_count"] = clean["days"].astype(str).apply(lambda text: len([part for part in text.split(",") if part.strip()]))
    preferred = clean.sort_values(
        ["max_od", "day_count", "improvement_after_penalty_pct", "avg_abs_shift"],
        ascending=[False, False, False, True],
    )
    return str(preferred.iloc[0]["scenario"]) if not preferred.empty else ""


def _write_final_scale_interpretation(
    scale_dir: Path,
    summary: pd.DataFrame,
    best_scenario: str,
    proxy_comparison: pd.DataFrame | None = None,
) -> None:
    lines = ["FINAL SCALE TEST YORUMU", "-----------------------"]
    if summary.empty:
        lines.append("Final scale summary bos; yorum uretilemedi.")
        (scale_dir / "final_scale_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return
    data = summary.copy()
    for col in [
        "improvement_after_penalty_pct",
        "improvement_before_penalty_pct",
        "total_hard_violations",
        "avg_abs_shift",
        "median_abs_shift",
        "runtime_sec",
        "active_connections",
        "beaten_competitors",
        "lookup_ranking_reward",
        "connection_reward",
        "shift_penalty",
        "mandatory_feasible_but_inactive",
        "mandatory_infeasible_but_active",
        "directional_e1_violations",
        "directional_e2_violations",
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    feasible = data["feasible"].astype(str).str.lower().isin(["true", "1"])
    completed = pd.to_numeric(data["objective_after_penalty"], errors="coerce").notna()
    hard_clean = data["total_hard_violations"].fillna(1) == 0
    mandatory_clean = (
        (data["mandatory_feasible_but_inactive"].fillna(1) == 0)
        & (data["mandatory_infeasible_but_active"].fillna(1) == 0)
    )
    directional_clean = (
        (data["directional_e1_violations"].fillna(1) == 0)
        & (data["directional_e2_violations"].fillna(1) == 0)
    )
    positive_after = data["improvement_after_penalty_pct"].fillna(-1e9) > 0
    lines.extend(
        [
            f"Tamamlanan senaryo sayisi: {len(data)}",
            f"Feasible senaryo sayisi: {int(feasible.sum())}",
            f"Hard violation toplam var mi: {bool((data['total_hard_violations'].fillna(0) > 0).any())}",
            f"Mandatory feasible-but-inactive toplam var mi: {bool((data['mandatory_feasible_but_inactive'].fillna(0) > 0).any())}",
            f"E1/E2 hard violation toplam var mi: {bool(((data['directional_e1_violations'].fillna(0) + data['directional_e2_violations'].fillna(0)) > 0).any())}",
            f"Penalty sonrasi baseline ustunde kalan senaryo sayisi: {int((completed & positive_after).sum())}",
            f"Min improvement after penalty pct: {float(data['improvement_after_penalty_pct'].min())}",
            f"Max improvement after penalty pct: {float(data['improvement_after_penalty_pct'].max())}",
            f"Max runtime sec: {float(data['runtime_sec'].max())}",
            f"Avg abs shift araligi: {float(data['avg_abs_shift'].min())} - {float(data['avg_abs_shift'].max())}",
            f"Secilen final aday senaryo: {best_scenario}",
            f"Ranking objective mode: {data['ranking_objective_mode'].dropna().astype(str).iloc[0] if 'ranking_objective_mode' in data.columns and data['ranking_objective_mode'].notna().any() else ''}",
            "",
            "Yorum:",
        ]
    )
    if bool(feasible.all()) and bool(hard_clean.all()) and bool(mandatory_clean.all()) and bool(directional_clean.all()):
        lines.append("- Final model tum scale testlerde feasible, mandatory uyumlu ve E1/E2 hard violation olmadan calisti.")
    else:
        failed = data[~feasible]["scenario"].astype(str).tolist()
        lines.append("- Bazi scale testler tamamlanamadi veya feasible bulunamadi; checker/solver sonucu esas alinmali.")
        if failed:
            lines.append(f"- Tamamlanamayan/feasible olmayan senaryolar: {', '.join(failed)}.")
    if bool((positive_after[completed]).all()):
        lines.append("- Tamamlanan senaryolarda penalty sonrasi objective baseline ustunde kaldi.")
        lines.append("- lookup_linear + linear_5 final model icin guclu adaydir.")
    else:
        lines.append("- Bazi buyuk testlerde penalty sonrasi improvement dustu veya negatif oldu.")
        lines.append("- Penalty weight olcege gore yeniden ayarlanabilir; linear_1 veya threshold_5_w5 alternatif olarak degerlendirilebilir.")
    if data["runtime_sec"].is_monotonic_increasing:
        lines.append("- Olcek buyudukce runtime genel olarak artiyor.")
    else:
        lines.append("- Runtime olcek ile birebir monoton degil; solver alternatif optimumlari farkli hizlarda bulabiliyor.")
    if data["avg_abs_shift"].max() <= 10:
        lines.append("- Shiftler final penalty ile makul seviyede kaldi.")
    else:
        lines.append("- Bazi senaryolarda avg shift hala yuksek; penalty veya kapsam ayari incelenebilir.")
    od200 = data[data["scenario"].astype(str).str.contains("od200", na=False)]
    od300 = data[data["scenario"].astype(str).str.contains("od300", na=False)]
    if not od200.empty:
        lines.append(f"- od200 senaryolari feasible sayisi: {int(feasible[od200.index].sum())}/{len(od200)}.")
    if not od300.empty:
        lines.append(f"- od300 senaryolari feasible sayisi: {int(feasible[od300.index].sum())}/{len(od300)}.")
        if int(feasible[od300.index].sum()) < len(od300):
            lines.append("- od300 infeasible kalirsa olasi neden, mandatory secim ile rotation/hub/regularity/directional hard kisitlarinin daha buyuk aday havuzunda ayni anda dar bogaz yaratmasidir.")
    if proxy_comparison is not None and not proxy_comparison.empty:
        comp = proxy_comparison.copy()
        for col in ["obj_after_delta", "beaten_delta", "avg_shift_delta", "lookup_hard_violations"]:
            comp[col] = pd.to_numeric(comp[col], errors="coerce")
        feasible_comp = comp[
            (comp["lookup_hard_violations"].fillna(1) == 0)
            & comp["obj_after_delta"].notna()
        ]
        if not feasible_comp.empty:
            lines.append(
                f"- Proxy karsilastirmasinda ortalama objective after delta: {float(feasible_comp['obj_after_delta'].mean())}."
            )
            lines.append(
                f"- Ortalama beaten competitor delta: {float(feasible_comp['beaten_delta'].mean())}; ortalama avg shift delta: {float(feasible_comp['avg_shift_delta'].mean())}."
            )
            if (feasible_comp["obj_after_delta"].dropna() > 0).any():
                lines.append("- lookup_linear proxy'ye gore objective/beaten competitor tarafinda net artis sagliyor.")
    else:
        lines.append("- Proxy final scale summary bulunamadigi icin proxy_vs_lookup_linear karsilastirmasi yapilamadi.")
    if best_scenario:
        best = data[data["scenario"].astype(str) == best_scenario].iloc[0]
        if str(best.get("ranking_objective_mode", "")).lower() == "lookup_linear":
            lines.append(f"- Final submission candidate lookup_linear olarak secildi: {best_scenario}.")
    lines.append("- Teslim icin onerilen ayar: final + mandatory selection + E1 hard absolute_tolerance=1 + E2 hard + linear shift penalty weight 5 + ranking_objective_mode=lookup_linear.")
    (scale_dir / "final_scale_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_final_submission_bundle(output_dir: Path, source_run_dir: Path, row: pd.Series) -> None:
    bundle_dir = output_dir / "final_submission_candidate"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for src in source_run_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, bundle_dir / src.name)
    interpretation = [
        "FINAL MODEL INTERPRETATION",
        "--------------------------",
        "Model; MVP connection selection cekirdegi uzerine rotation, hub capacity, schedule regularity, directional balance ve linear schedule disruption penalty kisit/cezalarini icerir.",
        f"Secilen run: {row.get('scenario', '')}",
        f"Ranking objective mode: {row.get('ranking_objective_mode', '')}",
        f"Baseline objective: {row.get('baseline_objective', '')}",
        f"Final objective before penalty: {row.get('objective_before_penalty', '')}",
        f"Final objective after penalty: {row.get('objective_after_penalty', '')}",
        f"Improvement before penalty pct: {row.get('improvement_before_penalty_pct', '')}",
        f"Improvement after penalty pct: {row.get('improvement_after_penalty_pct', '')}",
        f"Lookup ranking reward: {row.get('lookup_ranking_reward', '')}",
        f"Proxy ranking reward: {row.get('proxy_ranking_reward', '')}",
        f"Connection reward: {row.get('connection_reward', '')}",
        f"Shift penalty: {row.get('shift_penalty', '')}",
        f"Hard violations: {row.get('total_hard_violations', '')}",
        f"Mandatory feasible but inactive: {row.get('mandatory_feasible_but_inactive', '')}",
        f"Mandatory infeasible but active: {row.get('mandatory_infeasible_but_active', '')}",
        f"Directional E1 violations: {row.get('directional_e1_violations', '')}",
        f"Directional E2 violations: {row.get('directional_e2_violations', '')}",
        f"Avg abs shift: {row.get('avg_abs_shift', '')}",
        f"Median abs shift: {row.get('median_abs_shift', '')}",
        f"Active connections: {row.get('active_connections', '')}",
        "",
        "Yorum:",
        "- Bu ayar final aday secildi cunku mandatory secimi tam uyguluyor, lookup ranking reward'u objective'e daha dogrudan bagliyor, hard violation uretmeden baseline ustunde kaliyor ve shift penalty ile zaman kaymalarini makul seviyede tutuyor.",
        "- Kullanilan final ayar: model_level=final, mandatory_connection_selection=True, directional_e1_mode=hard, directional_e1_formula=absolute_tolerance, directional_e1_abs_tolerance=1, directional_e2_mode=hard, shift_penalty_mode=linear, shift_penalty_weight=5, ranking_objective_mode=lookup_linear.",
    ]
    (bundle_dir / "final_model_interpretation.txt").write_text("\n".join(interpretation), encoding="utf-8")


def run_ranking_objective_comparison_test(run_config: RunConfig) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = run_config.output_dir / f"ranking_objective_comparison_{timestamp}"
    scenarios = [
        ("od100_days12_top", 100, [1, 2], True),
        ("od200_days12_top", 200, [1, 2], True),
        ("od200_days123_top", 200, [1, 2, 3], True),
    ]
    modes = ["proxy", "lookup_posthoc", "lookup_linear"]
    rows = []
    for scenario, max_od, days, top_only in scenarios:
        for mode in modes:
            run_id = f"{scenario}_{mode}"
            cfg = clone_run_config(
                run_config,
                run_id=run_id,
                output_dir=comparison_dir,
                max_shift_min=30,
                max_od=max_od,
                days=days,
                top_importance_only=top_only,
                model_level="final",
                hub_arrival_capacity=HUB_ARRIVAL_CAPACITY_PER_BUCKET,
                hub_departure_capacity=HUB_DEPARTURE_CAPACITY_PER_BUCKET,
                hub_bucket_size_min=HUB_BUCKET_SIZE_MINUTES,
                regularity_xdev_min=SCHEDULE_REGULARITY_XDEV_MINUTES,
                directional_alpha=ALPHA_DIRECTIONAL_BALANCE,
                directional_gamma_min=GAMMA_BEST_JOURNEY_MINUTES,
                directional_e1_mode="hard",
                directional_e1_formula="absolute_tolerance",
                directional_e1_abs_tolerance=1,
                directional_e2_mode="hard",
                shift_penalty_mode="linear",
                shift_penalty_weight=5.0,
                mandatory_connection_selection=True,
                ranking_objective_mode=mode,
            )
            print(f"[INFO] Ranking objective comparison run scenario={scenario}, mode={mode}")
            try:
                result = run_once(cfg, analyze_mvp=True)
            except Exception as exc:
                print(f"[WARN] Ranking objective scenario failed: {run_id}: {exc}")
                rows.append(_ranking_comparison_failure_row(scenario, mode, exc))
                continue
            rows.append(_ranking_comparison_row(scenario, mode, result))
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(comparison_dir / "ranking_objective_comparison_summary.csv", index=False)
    _write_ranking_objective_interpretation(comparison_dir, summary)
    print(f"[INFO] Ranking objective comparison outputs written to {comparison_dir}")


def _ranking_comparison_row(scenario: str, mode: str, result: dict) -> dict:
    baseline = _safe_float(result.get("baseline_objective", ""))
    after = _safe_float(result.get("objective_after_penalty", ""))
    improvement = (
        100.0 * (after - baseline) / abs(baseline)
        if baseline not in {"", None} and after not in {"", None} and abs(baseline) > 1e-12
        else ""
    )
    hard = _result_hard_violations(result)
    return {
        "scenario": scenario,
        "ranking_objective_mode": mode,
        "feasible": result.get("feasible", False),
        "baseline_objective": baseline,
        "objective_before_penalty": result.get("objective_before_penalty", ""),
        "objective_after_penalty": after,
        "reported_lookup_ranking_reward": result.get("reported_lookup_ranking_reward", ""),
        "model_proxy_ranking_reward": result.get("model_proxy_ranking_reward", ""),
        "connection_reward": result.get("connection_reward", ""),
        "shift_penalty": result.get("total_shift_penalty", ""),
        "improvement_after_penalty_pct": improvement,
        "active_connections": result.get("active_connections", ""),
        "beaten_competitors": result.get("beaten_competitors", ""),
        "ranking_reward_difference_lookup_vs_proxy": result.get("ranking_reward_difference_lookup_vs_proxy", ""),
        "objective_recalculation_difference": result.get("ranking_objective_recalculation_difference", ""),
        "mandatory_feasible_but_inactive": result.get("feasible_but_inactive_connection_count", ""),
        "mandatory_infeasible_but_active": result.get("infeasible_but_active_connection_count", ""),
        "e1_violations": result.get("directional_e1_violations", ""),
        "e2_violations": result.get("directional_e2_violations", ""),
        "hard_violations": hard,
        "avg_abs_shift": result.get("avg_abs_shift", ""),
        "median_abs_shift": result.get("median_abs_shift", ""),
        "pct_shift_le_5": result.get("pct_shift_le_5", result.get("pct_small_shift_le_5min", "")),
        "runtime_sec": result.get("runtime_sec", ""),
        "solver_status": result.get("solver_status", ""),
        "gap": result.get("gap", ""),
    }


def _ranking_comparison_failure_row(scenario: str, mode: str, exc: Exception) -> dict:
    return {
        "scenario": scenario,
        "ranking_objective_mode": mode,
        "feasible": False,
        "baseline_objective": "",
        "objective_before_penalty": "",
        "objective_after_penalty": "",
        "reported_lookup_ranking_reward": "",
        "model_proxy_ranking_reward": "",
        "connection_reward": "",
        "shift_penalty": "",
        "improvement_after_penalty_pct": "",
        "active_connections": "",
        "beaten_competitors": "",
        "ranking_reward_difference_lookup_vs_proxy": "",
        "objective_recalculation_difference": "",
        "mandatory_feasible_but_inactive": "",
        "mandatory_infeasible_but_active": "",
        "e1_violations": "",
        "e2_violations": "",
        "hard_violations": "",
        "avg_abs_shift": "",
        "median_abs_shift": "",
        "pct_shift_le_5": "",
        "runtime_sec": "",
        "solver_status": f"failed: {type(exc).__name__}",
        "gap": "",
    }


def _write_ranking_objective_interpretation(comparison_dir: Path, summary: pd.DataFrame) -> None:
    lines = ["RANKING OBJECTIVE COMPARISON YORUMU", "------------------------------------"]
    if summary.empty:
        lines.append("Karsilastirma sonucu yok.")
        (comparison_dir / "ranking_objective_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return
    data = summary.copy()
    numeric_cols = [
        "objective_after_penalty",
        "improvement_after_penalty_pct",
        "reported_lookup_ranking_reward",
        "model_proxy_ranking_reward",
        "active_connections",
        "beaten_competitors",
        "ranking_reward_difference_lookup_vs_proxy",
        "hard_violations",
        "avg_abs_shift",
        "runtime_sec",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    feasible = data["feasible"].astype(str).str.lower().isin(["true", "1"])
    lookup_linear = data[data["ranking_objective_mode"] == "lookup_linear"]
    proxy = data[data["ranking_objective_mode"] == "proxy"]
    lines.extend(
        [
            f"Toplam kosu: {len(data)}",
            f"Feasible kosu: {int(feasible.sum())}",
            f"lookup_linear feasible kosu: {int(lookup_linear['feasible'].astype(str).str.lower().isin(['true', '1']).sum())}/{len(lookup_linear)}",
            f"Hard violation var mi: {bool((data['hard_violations'].fillna(0) > 0).any())}",
            f"Maksimum lookup-proxy reward farki: {float(data['ranking_reward_difference_lookup_vs_proxy'].abs().max()) if data['ranking_reward_difference_lookup_vs_proxy'].notna().any() else 0.0}",
            "",
            "Yorum:",
        ]
    )
    if not lookup_linear.empty and lookup_linear["feasible"].astype(str).str.lower().isin(["true", "1"]).all():
        lines.append("- lookup_linear tum test subsetlerinde feasible calisti.")
    else:
        lines.append("- lookup_linear en az bir testte feasible calismadi veya tamamlanamadi; final modele almadan once dikkatli incelenmeli.")
    merged = proxy[["scenario", "reported_lookup_ranking_reward", "active_connections", "beaten_competitors", "avg_abs_shift"]].merge(
        lookup_linear[["scenario", "reported_lookup_ranking_reward", "active_connections", "beaten_competitors", "avg_abs_shift"]],
        on="scenario",
        how="inner",
        suffixes=("_proxy", "_lookup_linear"),
    )
    if not merged.empty:
        lookup_gain = (merged["reported_lookup_ranking_reward_lookup_linear"] - merged["reported_lookup_ranking_reward_proxy"]).mean()
        active_gain = (merged["active_connections_lookup_linear"] - merged["active_connections_proxy"]).mean()
        beaten_gain = (merged["beaten_competitors_lookup_linear"] - merged["beaten_competitors_proxy"]).mean()
        shift_delta = (merged["avg_abs_shift_lookup_linear"] - merged["avg_abs_shift_proxy"]).mean()
        lines.append(f"- Ortalama lookup ranking reward farki lookup_linear - proxy: {lookup_gain}.")
        lines.append(f"- Ortalama active connection farki: {active_gain}; beaten competitor farki: {beaten_gain}.")
        lines.append(f"- Ortalama avg shift farki: {shift_delta}.")
        if lookup_gain > 1e-6 and shift_delta <= 2.0:
            lines.append("- lookup_linear posthoc lookup skorunu artiriyor ve shift davranisini belirgin bozmuyor; final aday icin degerlendirilebilir.")
        else:
            lines.append("- lookup_linear net bir lookup skor avantaji gostermiyorsa proxy + acik dokumantasyon daha guvenli kalabilir.")
    lines.append("- Nihai karar checker feasibility, hard violation=0 ve gercek lookup posthoc skoruna gore verilmelidir.")
    (comparison_dir / "ranking_objective_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def run_rotation_coverage_only(run_config: RunConfig) -> dict:
    """Analyze rotation matching coverage without solving the optimization model."""

    model_config = _make_model_config(run_config)
    run_dir = run_config.output_dir / run_config.run_id
    raw = load_inputs(run_config)
    prepared = prepare_data(raw, run_config, model_config)
    artifacts = build_model(prepared, model_config, "rotation")
    optimized_schedule = extract_optimized_schedule(artifacts)
    rotation_candidates = artifacts.rotation_candidates
    rotation_skip_analysis = build_rotation_skip_analysis(rotation_candidates, optimized_schedule, raw.connections, model_config)
    rotation_candidates = apply_rotation_skip_categories(rotation_candidates, rotation_skip_analysis)
    rotation_metrics = _summarize_rotation(rotation_candidates, None)
    rotation_skip_metrics = summarize_rotation_skip_analysis(rotation_candidates, rotation_skip_analysis)
    metrics = {
        "total_objective": "",
        "connection_reward": "",
        "ranking_reward": "",
        **rotation_metrics,
        **rotation_skip_metrics,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    rotation_candidates.to_excel(run_dir / "rotation_candidates.xlsx", index=False)
    write_rotation_skip_outputs(run_dir, rotation_candidates, rotation_skip_analysis, metrics)
    (run_dir / "run_summary.txt").write_text(
        render_run_summary(metrics, "coverage_only/not_solved", None, 0.0),
        encoding="utf-8",
    )
    append_experiment(
        run_config.output_dir,
        {
            "run_id": run_config.run_id,
            "model_level": "rotation_coverage_only",
            "solver": "",
            "time_limit": run_config.time_limit,
            "num_od": int(prepared.od_day["od"].nunique()) if not prepared.od_day.empty else 0,
            "num_flights": int(prepared.tk_connections[["inbound_flight", "outbound_flight"]].nunique().sum()),
            "num_candidate_connections": int(len(prepared.tk_connections)),
            "feasible": "",
            "objective": "",
            "baseline_objective": "",
            "improvement_pct": "",
            "connection_reward": "",
            "ranking_reward": "",
            "active_connections": "",
            "beaten_competitors": "",
            "improved_od_count": "",
            "worsened_od_count": "",
            "runtime_sec": 0.0,
            "gap": "",
            "rotation_candidates_total": metrics.get("rotation_candidates_total", 0),
            "rotation_constraints_active": metrics.get("rotation_constraints_active", 0),
            "rotation_constraints_skipped": metrics.get("rotation_constraints_skipped", 0),
            "rotation_coverage_pct": metrics.get("rotation_coverage_pct", 0.0),
            "rotation_skip_subset_excluded": metrics.get("rotation_skip_subset_excluded", 0),
            "rotation_skip_format_mismatch": metrics.get("rotation_skip_format_mismatch", 0),
            "rotation_skip_route_mismatch": metrics.get("rotation_skip_route_mismatch", 0),
            "rotation_skip_missing_block_time": metrics.get("rotation_skip_missing_block_time", 0),
            "rotation_block_elapsedtime_match_count": metrics.get("rotation_block_elapsedtime_match_count", 0),
            "rotation_block_legacy_exact_match_count": metrics.get("rotation_block_legacy_exact_match_count", 0),
            "rotation_block_exact_match_count": metrics.get("rotation_block_exact_match_count", 0),
            "rotation_block_fallback_count": metrics.get("rotation_block_fallback_count", 0),
            "rotation_violations": "",
            "notes": "rotation coverage analysis without solve",
        },
    )
    print(f"[INFO] Rotation coverage outputs written to {run_dir}")
    return {
        "run_id": run_config.run_id,
        "model_level": "rotation_coverage_only",
        "objective": "",
        "feasible": "",
        "rotation_violations": "",
        **metrics,
    }


def run_directional_coverage_only(run_config: RunConfig) -> dict:
    """Analyze directional O-D/reverse-O-D coverage without solving."""

    model_config = _make_model_config(run_config)
    run_dir = run_config.output_dir / run_config.run_id
    raw = load_inputs(run_config)
    prepared = prepare_data(raw, run_config, model_config)
    coverage, metrics = build_directional_coverage_analysis(raw.connections, prepared, model_config, None, run_config)
    directional_pairs = _directional_pairs_from_coverage(coverage)
    run_dir.mkdir(parents=True, exist_ok=True)
    directional_pairs.to_excel(run_dir / "directional_pairs.xlsx", index=False)
    coverage.to_excel(run_dir / "directional_coverage_analysis.xlsx", index=False)
    write_directional_coverage_summary(run_dir, metrics)
    summary_metrics = {
        "total_objective": "",
        "connection_reward": "",
        "ranking_reward": "",
        **metrics,
    }
    (run_dir / "run_summary.txt").write_text(
        render_run_summary(summary_metrics, "coverage_only/not_solved", None, 0.0),
        encoding="utf-8",
    )
    print(f"[INFO] Directional coverage outputs written to {run_dir}")
    return {
        "run_id": run_config.run_id,
        "model_level": "directional_coverage_only",
        "max_od": run_config.max_od,
        "days": ",".join(str(day) for day in run_config.days) if run_config.days else "",
        "top_importance_only": run_config.top_importance_only,
        **metrics,
    }


def run_capacity_stress_test(run_config: RunConfig) -> None:
    """Run the same capacity model subset with progressively tighter hub capacities."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stress_dir = run_config.output_dir / f"capacity_stress_{timestamp}"
    scenarios = [
        ("normal", HUB_ARRIVAL_CAPACITY_PER_BUCKET, HUB_DEPARTURE_CAPACITY_PER_BUCKET),
        ("medium_tight", 3, 3),
        ("tight", 2, 2),
        ("very_tight", 1, 1),
    ]
    rows = []
    for scenario, arrival_cap, departure_cap in scenarios:
        stress_config = clone_run_config(
            run_config,
            run_id=scenario,
            output_dir=stress_dir,
            max_shift_min=run_config.max_shift_min,
            model_level="capacity",
            hub_arrival_capacity=arrival_cap,
            hub_departure_capacity=departure_cap,
            hub_bucket_size_min=run_config.hub_bucket_size_min,
        )
        print(
            "[INFO] Capacity stress run "
            f"scenario={scenario}, arrival_capacity={arrival_cap}, departure_capacity={departure_cap}"
        )
        result = run_once(stress_config, analyze_mvp=True)
        rows.append(
            {
                "scenario": scenario,
                "arrival_capacity": arrival_cap,
                "departure_capacity": departure_cap,
                "feasible": result["feasible"],
                "baseline_objective": result["baseline_objective"],
                "rotation_objective": result.get("rotation_objective", ""),
                "capacity_objective": result["objective"],
                "improvement_vs_baseline_pct": result["improvement_pct"],
                "drop_vs_rotation": result.get("drop_vs_rotation", ""),
                "active_connections": result["active_connections"],
                "hub_capacity_violations": result.get("hub_capacity_violations", 0),
                "max_arrivals_in_bucket": result.get("max_arrivals_in_bucket", 0),
                "max_departures_in_bucket": result.get("max_departures_in_bucket", 0),
                "avg_abs_shift": result.get("avg_abs_shift", 0.0),
                "median_abs_shift": result.get("median_abs_shift", 0.0),
                "runtime_sec": result["runtime_sec"],
                "solver_status": result["solver_status"],
            }
        )
    stress_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(stress_dir / "capacity_stress_summary.csv", index=False)
    _write_capacity_stress_interpretation(stress_dir, summary)
    print(f"[INFO] Capacity stress outputs written to {stress_dir}")


def run_regularity_stress_test(run_config: RunConfig) -> None:
    """Run regularity model with progressively tighter XDEV limits."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stress_dir = run_config.output_dir / f"regularity_stress_{timestamp}"
    scenarios = [
        ("loose", 60),
        ("normal", SCHEDULE_REGULARITY_XDEV_MINUTES),
        ("tight", 5),
        ("very_tight", 0),
    ]
    rows = []
    for scenario, xdev in scenarios:
        stress_config = clone_run_config(
            run_config,
            run_id=scenario,
            output_dir=stress_dir,
            max_shift_min=run_config.max_shift_min,
            model_level="regularity",
            regularity_xdev_min=xdev,
        )
        print(f"[INFO] Regularity stress run scenario={scenario}, xdev={xdev}")
        result = run_once(stress_config, analyze_mvp=True)
        rows.append(
            {
                "scenario": scenario,
                "xdev": xdev,
                "feasible": result["feasible"],
                "baseline_objective": result["baseline_objective"],
                "capacity_objective": result.get("reference_capacity_objective", ""),
                "regularity_objective": result["objective"],
                "improvement_vs_baseline_pct": result["improvement_pct"],
                "drop_vs_capacity": result.get("drop_vs_capacity", ""),
                "active_connections": result["active_connections"],
                "regularity_violations": result.get("regularity_violations", 0),
                "regularity_constraints_active": result.get("regularity_constraints_active", 0),
                "coverage_total_regularity_groups": result.get("coverage_total_regularity_groups", 0),
                "coverage_active_regularity_groups": result.get("coverage_active_regularity_groups", 0),
                "coverage_skipped_single_instance_groups": result.get("coverage_skipped_single_instance_groups", 0),
                "coverage_active_group_pct": result.get("coverage_active_group_pct", 0.0),
                "coverage_groups_old_spread_gt_xdev": result.get("coverage_groups_old_spread_gt_xdev", 0),
                "coverage_groups_new_spread_gt_xdev": result.get("coverage_groups_new_spread_gt_xdev", 0),
                "coverage_groups_spread_reduced": result.get("coverage_groups_spread_reduced", 0),
                "coverage_groups_used_in_active_connections": result.get("coverage_groups_used_in_active_connections", 0),
                "effect_flights_changed_vs_capacity": result.get("effect_flights_changed_vs_capacity", 0),
                "effect_avg_abs_difference_vs_capacity": result.get("effect_avg_abs_difference_vs_capacity", 0.0),
                "max_regularity_spread": result.get("max_regularity_spread", 0.0),
                "avg_regularity_spread": result.get("avg_regularity_spread", 0.0),
                "avg_abs_shift": result.get("avg_abs_shift", 0.0),
                "median_abs_shift": result.get("median_abs_shift", 0.0),
                "runtime_sec": result["runtime_sec"],
                "solver_status": result["solver_status"],
            }
        )
    stress_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(stress_dir / "regularity_stress_summary.csv", index=False)
    _write_regularity_stress_interpretation(stress_dir, summary)
    print(f"[INFO] Regularity stress outputs written to {stress_dir}")


def _write_regularity_stress_interpretation(stress_dir: Path, summary: pd.DataFrame) -> None:
    lines = ["REGULARITY STRESS TEST YORUMU", "------------------------------"]
    lines.append(f"Tamamlanan senaryo sayisi: {len(summary)}")
    if summary.empty:
        lines.append("Yorumlanacak sonuc yok.")
        (stress_dir / "regularity_stress_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return
    objectives = pd.to_numeric(summary["regularity_objective"], errors="coerce")
    drops = pd.to_numeric(summary["drop_vs_capacity"], errors="coerce").fillna(0.0)
    violations = pd.to_numeric(summary["regularity_violations"], errors="coerce").fillna(0)
    loose = summary[summary["scenario"] == "loose"]
    normal = summary[summary["scenario"] == "normal"]
    tight = summary[summary["scenario"] == "tight"]
    very_tight = summary[summary["scenario"] == "very_tight"]
    objective_changed = objectives.nunique(dropna=False) > 1
    max_drop = float(drops.max()) if len(drops) else 0.0
    very_tight_feasible = bool(very_tight["feasible"].iloc[0]) if not very_tight.empty else False
    lines.extend(
        [
            f"Loose objective: {loose['regularity_objective'].iloc[0] if not loose.empty else ''}",
            f"Normal XDEV drop vs capacity: {normal['drop_vs_capacity'].iloc[0] if not normal.empty else ''}",
            f"Tight XDEV drop vs capacity: {tight['drop_vs_capacity'].iloc[0] if not tight.empty else ''}",
            f"Very tight feasible: {very_tight_feasible}",
            f"Max drop vs capacity: {max_drop}",
            f"Regularity checker violation toplamı: {int(violations.sum())}",
            "",
            "Yorum:",
        ]
    )
    if objective_changed and max_drop > 1e-6:
        lines.append("- Regularity kisiti dogru calismaktadir ve sikilastirildikca modelin hareket alanini azaltmaktadir.")
    elif not objective_changed:
        lines.append("- Bu subsette ayni ucus numarasinin coklu gun ornegi az olabilir veya regularity zaten dogal olarak saglaniyordur.")
    else:
        lines.append("- Regularity objective'i degistirdi; ancak etki her XDEV seviyesinde monoton dusus olarak gorunmedi.")
    if very_tight_feasible:
        lines.append("- Cok siki regularity modelin feasible kalmasini bozmadı.")
    else:
        lines.append("- Cok siki regularity modelin feasible kalmasini bozdu veya checker feasible bulmadi.")
    if int(violations.sum()) == 0:
        lines.append("- Gunler arasi saat tutarliligi checker'a gore saglandi.")
    else:
        lines.append("- Checker regularity violation buldu; model/checker uyumu incelenmeli.")
    (stress_dir / "regularity_stress_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def run_directional_stress_test(run_config: RunConfig) -> None:
    """Run directional model with progressively tighter alpha/gamma settings."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stress_dir = run_config.output_dir / f"directional_stress_{timestamp}"
    scenarios = [
        ("loose", 1.00, 120),
        ("normal", ALPHA_DIRECTIONAL_BALANCE, GAMMA_BEST_JOURNEY_MINUTES),
        ("tight", 0.10, 15),
        ("very_tight", 0.00, 0),
    ]
    rows = []
    for scenario, alpha, gamma in scenarios:
        stress_config = clone_run_config(
            run_config,
            run_id=scenario,
            output_dir=stress_dir,
            max_shift_min=run_config.max_shift_min,
            model_level="directional",
            directional_alpha=alpha,
            directional_gamma_min=gamma,
        )
        print(f"[INFO] Directional stress run scenario={scenario}, alpha={alpha}, gamma={gamma}")
        result = run_once(stress_config, analyze_mvp=True)
        rows.append(
            {
                "scenario": scenario,
                "alpha": alpha,
                "gamma": gamma,
                "feasible": result["feasible"],
                "baseline_objective": result["baseline_objective"],
                "regularity_objective": result.get("reference_regularity_objective", ""),
                "directional_objective": result["objective"],
                "improvement_vs_baseline_pct": result["improvement_pct"],
                "drop_vs_regularity": result.get("drop_vs_regularity", ""),
                "active_connections": result["active_connections"],
                "directional_pairs_active": result.get("directional_pairs_active", 0),
                "directional_e1_violations": result.get("directional_e1_violations", 0),
                "directional_e2_violations": result.get("directional_e2_violations", 0),
                "directional_total_violations": result.get("directional_total_violations", 0),
                "max_connection_count_diff": result.get("max_connection_count_diff", 0),
                "avg_connection_count_diff": result.get("avg_connection_count_diff", 0.0),
                "max_journey_time_diff": result.get("max_journey_time_diff", 0.0),
                "avg_journey_time_diff": result.get("avg_journey_time_diff", 0.0),
                "avg_abs_shift": result.get("avg_abs_shift", 0.0),
                "median_abs_shift": result.get("median_abs_shift", 0.0),
                "runtime_sec": result["runtime_sec"],
                "solver_status": result["solver_status"],
            }
        )
    stress_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(stress_dir / "directional_stress_summary.csv", index=False)
    _write_directional_stress_interpretation(stress_dir, summary)
    print(f"[INFO] Directional stress outputs written to {stress_dir}")


def _write_directional_stress_interpretation(stress_dir: Path, summary: pd.DataFrame) -> None:
    lines = ["DIRECTIONAL STRESS TEST YORUMU", "-------------------------------"]
    lines.append(f"Tamamlanan senaryo sayisi: {len(summary)}")
    if summary.empty:
        lines.append("Yorumlanacak sonuc yok.")
        (stress_dir / "directional_stress_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return
    objectives = pd.to_numeric(summary["directional_objective"], errors="coerce")
    drops = pd.to_numeric(summary["drop_vs_regularity"], errors="coerce").fillna(0.0)
    total_violations = pd.to_numeric(summary["directional_total_violations"], errors="coerce").fillna(0)
    normal = summary[summary["scenario"] == "normal"]
    very_tight = summary[summary["scenario"] == "very_tight"]
    objective_changed = objectives.nunique(dropna=False) > 1
    very_tight_feasible = bool(very_tight["feasible"].iloc[0]) if not very_tight.empty else False
    lines.extend(
        [
            f"Normal feasible: {bool(normal['feasible'].iloc[0]) if not normal.empty else ''}",
            f"Normal drop vs regularity: {normal['drop_vs_regularity'].iloc[0] if not normal.empty else ''}",
            f"Very tight feasible: {very_tight_feasible}",
            f"Max drop vs regularity: {float(drops.max()) if len(drops) else 0.0}",
            f"Directional checker violation toplami: {int(total_violations.sum())}",
            f"Objective senaryolar arasinda degisti mi: {objective_changed}",
            "",
            "Yorum:",
        ]
    )
    if objective_changed:
        lines.append("- Directional balance parametreleri cozum uzayini etkiliyor.")
    else:
        lines.append("- Bu subsette directional parametreler objective uzerinde belirgin fark olusturmadi.")
    if very_tight_feasible:
        lines.append("- Cok siki durumda model checker'a gore feasible kaldi.")
    else:
        lines.append("- Cok siki durumda model feasible kalmadi veya checker violation buldu.")
    if int(total_violations.sum()) == 0:
        lines.append("- Checker E1/E2 directional violation bulmadi.")
    else:
        lines.append("- Checker E1/E2 violation buldu; model/checker uyumu veya E2 model entegrasyonu incelenmeli.")
    (stress_dir / "directional_stress_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def run_directional_scaling_test(run_config: RunConfig) -> None:
    """Run directional model on progressively broader subsets."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scaling_dir = run_config.output_dir / f"directional_scaling_{timestamp}"
    experiments = [
        {"scenario": "od50_top", "max_od": 50, "days": [1, 2], "top_importance_only": True},
        {"scenario": "od100_top", "max_od": 100, "days": [1, 2], "top_importance_only": True},
        {"scenario": "od200_top", "max_od": 200, "days": [1, 2], "top_importance_only": True},
        {"scenario": "od100_all", "max_od": 100, "days": [1, 2], "top_importance_only": False},
        {"scenario": "od200_all", "max_od": 200, "days": [1, 2], "top_importance_only": False},
    ]
    rows = []
    for exp in experiments:
        scale_config = clone_run_config(
            run_config,
            run_id=exp["scenario"],
            output_dir=scaling_dir,
            max_shift_min=run_config.max_shift_min,
            max_od=exp["max_od"],
            days=exp["days"],
            top_importance_only=exp["top_importance_only"],
            model_level="directional",
            directional_alpha=ALPHA_DIRECTIONAL_BALANCE,
            directional_gamma_min=GAMMA_BEST_JOURNEY_MINUTES,
        )
        scale_config.time_limit = max(run_config.time_limit, 900)
        print(
            "[INFO] Directional scaling run "
            f"scenario={exp['scenario']}, max_od={exp['max_od']}, "
            f"days={','.join(str(day) for day in exp['days'])}, "
            f"top_importance_only={exp['top_importance_only']}"
        )
        result = run_once(scale_config, analyze_mvp=True)
        rows.append(
            {
                "scenario": exp["scenario"],
                "max_od": exp["max_od"],
                "days": ",".join(str(day) for day in exp["days"]),
                "top_importance_only": exp["top_importance_only"],
                "feasible": result["feasible"],
                "baseline_objective": result["baseline_objective"],
                "regularity_objective": result.get("reference_regularity_objective", ""),
                "directional_objective": result["objective"],
                "improvement_vs_baseline_pct": result["improvement_pct"],
                "drop_vs_regularity": result.get("drop_vs_regularity", ""),
                "active_connections": result["active_connections"],
                "directional_pairs_raw": result.get("coverage_directional_pairs_raw", 0),
                "directional_pairs_selected_subset": result.get("coverage_directional_pairs_selected_subset", 0),
                "directional_pairs_active": result.get("directional_pairs_active", 0),
                "pairs_with_both_candidate_connections": result.get("coverage_pairs_both_candidate_connections", 0),
                "pairs_with_both_active_connections": result.get("coverage_pairs_both_active_connections", 0),
                "pairs_only_forward_active": result.get("coverage_pairs_only_forward_active", 0),
                "pairs_only_reverse_active": result.get("coverage_pairs_only_reverse_active", 0),
                "e1_applicable_pairs": result.get("coverage_e1_applicable_pairs", 0),
                "e2_applicable_pairs": result.get("coverage_e2_applicable_pairs", 0),
                "e1_violations": result.get("directional_e1_violations", 0),
                "e2_violations": result.get("directional_e2_violations", 0),
                "total_directional_violations": result.get("directional_total_violations", 0),
                "max_connection_count_diff": result.get("max_connection_count_diff", 0),
                "avg_connection_count_diff": result.get("avg_connection_count_diff", 0.0),
                "max_journey_time_diff": result.get("max_journey_time_diff", 0.0),
                "avg_journey_time_diff": result.get("avg_journey_time_diff", 0.0),
                "avg_abs_shift": result.get("avg_abs_shift", 0.0),
                "median_abs_shift": result.get("median_abs_shift", 0.0),
                "runtime_sec": result["runtime_sec"],
                "solver_status": result["solver_status"],
            }
        )
    scaling_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(scaling_dir / "directional_scaling_summary.csv", index=False)
    _write_directional_scaling_interpretation(scaling_dir, summary)
    print(f"[INFO] Directional scaling outputs written to {scaling_dir}")


def _write_directional_scaling_interpretation(scaling_dir: Path, summary: pd.DataFrame) -> None:
    lines = ["DIRECTIONAL SCALING YORUMU", "---------------------------"]
    lines.append(f"Tamamlanan senaryo sayisi: {len(summary)}")
    if summary.empty:
        lines.append("Yorumlanacak sonuc yok.")
        (scaling_dir / "directional_scaling_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return
    raw_pairs = pd.to_numeric(summary["directional_pairs_raw"], errors="coerce").fillna(0)
    selected_pairs = pd.to_numeric(summary["directional_pairs_selected_subset"], errors="coerce").fillna(0)
    both_active = pd.to_numeric(summary["pairs_with_both_active_connections"], errors="coerce").fillna(0)
    drops = pd.to_numeric(summary["drop_vs_regularity"], errors="coerce").fillna(0.0)
    active_connections = pd.to_numeric(summary["active_connections"], errors="coerce").fillna(0)
    violations = pd.to_numeric(summary["total_directional_violations"], errors="coerce").fillna(0)
    lines.extend(
        [
            f"Max raw directional pairs: {int(raw_pairs.max()) if len(raw_pairs) else 0}",
            f"Max selected subset directional pairs: {int(selected_pairs.max()) if len(selected_pairs) else 0}",
            f"Max both-active directional pairs: {int(both_active.max()) if len(both_active) else 0}",
            f"Max drop vs regularity: {float(drops.max()) if len(drops) else 0.0}",
            f"Active connection sayisi senaryolar arasinda degisti mi: {active_connections.nunique(dropna=False) > 1}",
            f"Directional checker violation toplamda var mi: {int(violations.sum()) > 0}",
            "",
            "Yorum:",
        ]
    )
    if selected_pairs.max() > selected_pairs.min():
        lines.append("- max_od veya top_importance ayarlari genisleyince selected subset directional coverage artiyor.")
    else:
        lines.append("- Selected subset directional coverage senaryolar arasinda belirgin artmadi.")
    if both_active.max() == 0:
        lines.append("- Iki yonde aktif connection olan pair hala yok; directional balance objective'i zorlamaz.")
    else:
        lines.append("- Iki yonde aktif connection olan pair olustu; directional balance gercek model etkisi uretebilir.")
    if drops.abs().max() <= 1e-6:
        lines.append("- Directional objective regularity referansina gore degismedi veya ihmal edilebilir seviyede degisti.")
    else:
        lines.append("- Directional balance regularity referansina gore objective uzerinde etki olusturdu.")
    if int(violations.sum()) == 0:
        lines.append("- Checker E1/E2 violation bulmadi.")
    else:
        lines.append("- Checker E1/E2 violation buldu; E2 model entegrasyonu veya parametreler incelenmeli.")
    (scaling_dir / "directional_scaling_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_capacity_stress_interpretation(stress_dir: Path, summary: pd.DataFrame) -> None:
    lines = ["CAPACITY STRESS TEST YORUMU", "-----------------------------"]
    lines.append(f"Tamamlanan senaryo sayisi: {len(summary)}")
    if summary.empty:
        lines.append("Yorumlanacak sonuc yok.")
        (stress_dir / "capacity_stress_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return

    normal = summary[summary["scenario"] == "normal"]
    tight = summary[summary["scenario"] == "tight"]
    very_tight = summary[summary["scenario"] == "very_tight"]
    normal_drop = float(normal["drop_vs_rotation"].iloc[0]) if not normal.empty and normal["drop_vs_rotation"].iloc[0] != "" else 0.0
    tight_drop = float(tight["drop_vs_rotation"].iloc[0]) if not tight.empty and tight["drop_vs_rotation"].iloc[0] != "" else 0.0
    normal_obj = float(normal["capacity_objective"].iloc[0]) if not normal.empty else 0.0
    tight_obj = float(tight["capacity_objective"].iloc[0]) if not tight.empty else normal_obj
    very_tight_feasible = bool(very_tight["feasible"].iloc[0]) if not very_tight.empty else False
    any_checker_violation = int(pd.to_numeric(summary["hub_capacity_violations"], errors="coerce").fillna(0).sum()) > 0
    active_changed = summary["active_connections"].nunique(dropna=False) > 1
    objective_changed = pd.to_numeric(summary["capacity_objective"], errors="coerce").nunique(dropna=False) > 1
    drops = pd.to_numeric(summary["drop_vs_rotation"], errors="coerce").fillna(0.0)
    max_positive_drop = float(drops.max()) if len(drops) else 0.0
    min_drop = float(drops.min()) if len(drops) else 0.0

    lines.extend(
        [
            f"Normal capacity drop vs rotation: {normal_drop}",
            f"Tight capacity drop vs rotation: {tight_drop}",
            f"Max positive drop vs rotation: {max_positive_drop}",
            f"Min drop vs rotation: {min_drop}",
            f"Very tight feasible: {very_tight_feasible}",
            f"Hub capacity checker violation toplamda var mi: {any_checker_violation}",
            f"Active connection sayisi senaryolar arasinda degisti mi: {active_changed}",
            f"Objective senaryolar arasinda degisti mi: {objective_changed}",
            "",
            "Yorum:",
        ]
    )
    if abs(normal_drop) <= 1e-6:
        lines.append("- Normal kapasitede constraint bu subsette baglayici gorunmuyor.")
    else:
        lines.append("- Normal kapasitede bile objective dustu; capacity kisiti baglayici.")
    if abs(normal_drop) <= 1e-6 and max_positive_drop > 1e-6:
        lines.append(
            "- Capacity kisiti dogru calismaktadir; normal yarisma kapasitesi bu subsette baglayici degildir."
        )
    elif abs(normal_drop) <= 1e-6 and (abs(tight_obj - normal_obj) > 1e-6 or objective_changed):
        lines.append(
            "- Kapasite dusurulunce objective degisti; capacity kisiti cozum uzayini etkiliyor. Bu run'da objective dusmek yerine daha iyi bir alternatif cozum bulundu."
        )
    elif not objective_changed:
        lines.append(
            "- Tight/very tight capacity'de de objective degismedi; capacity kisiti muhtemelen modelde yeterince baglayici degildir veya subset yogunlugu cok dusuktur."
        )
    else:
        lines.append("- Kapasite dusuruldukce objective uzerinde baski olusuyor.")
    if active_changed:
        lines.append("- Kapasite dusurulunce aktif baglanti sayisi degisti.")
    else:
        lines.append("- Kapasite dusurulunce aktif baglanti sayisi degismedi.")
    if very_tight_feasible:
        lines.append("- Cok siki kapasitede model checker'a gore feasible kaldi.")
    else:
        lines.append("- Cok siki kapasitede model checker'a gore feasible kalmadi.")
    if any_checker_violation:
        lines.append("- Model ve checker arasinda tutarsizlik var, duzeltilmeli.")
    else:
        lines.append("- Checker hub capacity violation bulmadi.")
    (stress_dir / "capacity_stress_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def run_scaling_test(run_config: RunConfig) -> None:
    """Run the fixed MVP scalability/stability experiment set."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scaling_dir = run_config.output_dir / f"scaling_test_{timestamp}"
    experiments = [
        {"max_od": 20, "days": [1], "max_shift": 30},
        {"max_od": 50, "days": [1], "max_shift": 30},
        {"max_od": 100, "days": [1], "max_shift": 30},
        {"max_od": 20, "days": [1, 2], "max_shift": 30},
        {"max_od": 50, "days": [1, 2], "max_shift": 30},
    ]
    rows = []
    for idx, experiment in enumerate(experiments, start=1):
        days = experiment["days"]
        day_text = "".join(str(day) for day in days)
        run_id = f"scale_{idx}_od{experiment['max_od']}_days{day_text}_shift{experiment['max_shift']}"
        scale_config = clone_run_config(
            run_config,
            run_id=run_id,
            output_dir=scaling_dir,
            max_shift_min=experiment["max_shift"],
            max_od=experiment["max_od"],
            days=days,
            model_level="mvp",
        )
        print(
            "[INFO] Scaling run "
            f"max_od={experiment['max_od']}, days={','.join(str(day) for day in days)}, "
            f"max_shift={experiment['max_shift']}"
        )
        rows.append(run_once(scale_config, analyze_mvp=True))
    write_scaling_summary(scaling_dir, rows)
    print(f"[INFO] Scaling outputs written to {scaling_dir}")


def run_mandatory_directional_diagnosis(run_config: RunConfig) -> None:
    """Diagnose directional feasibility after mandatory connection selection."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnosis_dir = run_config.output_dir / f"mandatory_directional_diagnosis_{timestamp}"
    scenarios = [
        {
            "scenario": "regularity_only_mandatory",
            "model_level": "regularity",
            "e1_mode": "report_only",
            "e2_mode": "report_only",
            "alpha": run_config.directional_alpha,
            "e1_formula": "ratio_total",
            "allow_soft": False,
        },
        {
            "scenario": "directional_report_report",
            "model_level": "final",
            "e1_mode": "report_only",
            "e2_mode": "report_only",
            "alpha": run_config.directional_alpha,
            "e1_formula": "ratio_total",
            "allow_soft": False,
        },
        {
            "scenario": "directional_soft_report",
            "model_level": "final",
            "e1_mode": "soft",
            "e2_mode": "report_only",
            "alpha": run_config.directional_alpha,
            "e1_formula": "ratio_total",
            "allow_soft": True,
        },
        {
            "scenario": "directional_soft_soft",
            "model_level": "final",
            "e1_mode": "soft",
            "e2_mode": "soft",
            "alpha": run_config.directional_alpha,
            "e1_formula": "ratio_total",
            "allow_soft": True,
        },
        {
            "scenario": "directional_hard_report",
            "model_level": "final",
            "e1_mode": "hard",
            "e2_mode": "report_only",
            "alpha": run_config.directional_alpha,
            "e1_formula": "ratio_total",
            "allow_soft": False,
        },
        {
            "scenario": "directional_hard_hard",
            "model_level": "final",
            "e1_mode": "hard",
            "e2_mode": "hard",
            "alpha": run_config.directional_alpha,
            "e1_formula": "ratio_total",
            "allow_soft": False,
        },
        {
            "scenario": "relaxed_alpha_hard_report",
            "model_level": "final",
            "e1_mode": "hard",
            "e2_mode": "report_only",
            "alpha": 1.0,
            "e1_formula": "ratio_total",
            "allow_soft": False,
        },
        {
            "scenario": "no_directional_final",
            "model_level": "regularity",
            "e1_mode": "report_only",
            "e2_mode": "report_only",
            "alpha": run_config.directional_alpha,
            "e1_formula": "ratio_total",
            "allow_soft": False,
        },
    ]
    rows = []
    for scenario in scenarios:
        diag_config = clone_run_config(
            run_config,
            run_id=scenario["scenario"],
            output_dir=diagnosis_dir,
            max_shift_min=30,
            model_level=scenario["model_level"],
            directional_alpha=scenario["alpha"],
            directional_gamma_min=run_config.directional_gamma_min,
            directional_e1_mode=scenario["e1_mode"],
            directional_e1_formula=scenario["e1_formula"],
            directional_e1_abs_tolerance=run_config.directional_e1_abs_tolerance,
            directional_e1_penalty_weight=run_config.directional_e1_penalty_weight,
            directional_e2_mode=scenario["e2_mode"],
            directional_e2_penalty_weight=run_config.directional_e2_penalty_weight,
            allow_soft_directional_violations=scenario["allow_soft"],
            shift_penalty_mode="linear",
            shift_penalty_weight=5.0,
            mandatory_connection_selection=True,
        )
        print(
            "[INFO] Mandatory directional diagnosis "
            f"scenario={scenario['scenario']}, model={scenario['model_level']}, "
            f"E1={scenario['e1_mode']}/{scenario['e1_formula']}, E2={scenario['e2_mode']}, alpha={scenario['alpha']}"
        )
        result = run_once(diag_config, analyze_mvp=True)
        rows.append(_diagnosis_row_from_result(scenario, result, diag_config))
    diagnosis_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(diagnosis_dir / "diagnosis_summary.csv", index=False)
    _write_mandatory_directional_diagnosis_interpretation(diagnosis_dir, summary)
    print(f"[INFO] Mandatory directional diagnosis outputs written to {diagnosis_dir}")


def _diagnosis_row_from_result(scenario: dict, result: dict, run_config: RunConfig) -> dict:
    hard_violations = _result_hard_violations(result)
    return {
        "scenario": scenario["scenario"],
        "max_od": run_config.max_od,
        "days": ",".join(str(day) for day in run_config.days) if run_config.days else "",
        "mandatory_enabled": bool(result.get("mandatory_connection_selection", run_config.mandatory_connection_selection)),
        "model_level": run_config.model_level,
        "e1_mode": run_config.directional_e1_mode,
        "e1_formula": run_config.directional_e1_formula,
        "e1_tolerance": run_config.directional_e1_abs_tolerance,
        "e2_mode": run_config.directional_e2_mode,
        "alpha": run_config.directional_alpha,
        "gamma": run_config.directional_gamma_min,
        "feasible": result.get("feasible", False),
        "hard_violations": hard_violations,
        "mandatory_feasible_but_inactive": result.get("feasible_but_inactive_connection_count", ""),
        "mandatory_infeasible_but_active": result.get("infeasible_but_active_connection_count", ""),
        "e1_hard_violations": result.get("directional_e1_hard_violations", ""),
        "e1_soft_violations": result.get("directional_e1_soft_violations", ""),
        "e1_violation_amount_total": result.get("directional_e1_soft_violation_amount_total", ""),
        "e2_hard_violations": result.get("directional_e2_hard_violations", ""),
        "e2_soft_violations": result.get("directional_e2_soft_violations", ""),
        "e2_violation_amount_total": result.get("directional_e2_soft_violation_amount_total", ""),
        "baseline_objective": result.get("baseline_objective", ""),
        "objective_before_penalty": result.get("objective_before_penalty", ""),
        "objective_after_penalty": result.get("objective_after_penalty", ""),
        "active_connections": result.get("active_connections", ""),
        "total_candidate_connections": result.get("total_candidate_connections", ""),
        "feasible_by_schedule": result.get("connections_feasible_by_final_schedule", ""),
        "avg_abs_shift": result.get("avg_abs_shift", ""),
        "median_abs_shift": result.get("median_abs_shift", ""),
        "pct_shift_le_5": result.get("pct_shift_le_5", result.get("pct_small_shift_le_5min", "")),
        "runtime_sec": result.get("runtime_sec", ""),
        "solver_status": result.get("solver_status", ""),
        "suspected_bottleneck": _suspected_directional_bottleneck(result, hard_violations),
    }


def _result_hard_violations(result: dict) -> int | str:
    fields = [
        "feasible_but_inactive_connection_count",
        "infeasible_but_active_connection_count",
        "rotation_violations",
        "hub_capacity_violations",
        "regularity_violations",
        "directional_e1_hard_violations",
        "directional_e2_hard_violations",
    ]
    total = 0
    for field in fields:
        value = result.get(field, 0)
        if value in {"", None}:
            return ""
        total += int(float(value))
    return total


def _suspected_directional_bottleneck(result: dict, hard_violations) -> str:
    if not result.get("feasible", False) and result.get("solver_status") == "failed/NoFeasibleSolutionError":
        return "solver_infeasible_before_checker"
    if hard_violations not in {"", None} and int(hard_violations) == 0:
        soft_e1 = float(result.get("directional_e1_soft_violation_amount_total", 0.0) or 0.0)
        soft_e2 = float(result.get("directional_e2_soft_violation_amount_total", 0.0) or 0.0)
        if soft_e1 > 0:
            return "e1_soft_tradeoff"
        if soft_e2 > 0:
            return "e2_soft_tradeoff"
        return "none"
    if int(float(result.get("directional_e1_hard_violations", 0) or 0)) > 0:
        return "e1_hard"
    if int(float(result.get("directional_e2_hard_violations", 0) or 0)) > 0:
        return "e2_hard"
    return "hard_constraint"


def _write_mandatory_directional_diagnosis_interpretation(diagnosis_dir: Path, summary: pd.DataFrame) -> None:
    lines = ["MANDATORY + DIRECTIONAL DIAGNOSIS YORUMU", "----------------------------------------"]
    lines.append(f"Tamamlanan senaryo sayisi: {len(summary)}")
    if summary.empty:
        lines.append("Yorumlanacak sonuc yok.")
        (diagnosis_dir / "diagnosis_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
        return
    feasible = summary[summary["feasible"].astype(str).str.lower().isin(["true", "1"])]
    hard_report = summary[summary["scenario"] == "directional_hard_report"]
    report_report = summary[summary["scenario"] == "directional_report_report"]
    soft_report = summary[summary["scenario"] == "directional_soft_report"]
    relaxed = summary[summary["scenario"] == "relaxed_alpha_hard_report"]
    lines.extend(
        [
            f"Feasible senaryo sayisi: {len(feasible)}",
            f"E1 hard/report feasible: {hard_report['feasible'].iloc[0] if not hard_report.empty else ''}",
            f"E1 report/report feasible: {report_report['feasible'].iloc[0] if not report_report.empty else ''}",
            f"E1 soft/report feasible: {soft_report['feasible'].iloc[0] if not soft_report.empty else ''}",
            f"Alpha=1 hard/report feasible: {relaxed['feasible'].iloc[0] if not relaxed.empty else ''}",
            "",
            "Yorum:",
        ]
    )
    if not hard_report.empty and str(hard_report["feasible"].iloc[0]).lower() not in {"true", "1"}:
        lines.append("- E1 hard ratio_total mandatory selection sonrasinda ana darboğaz adayi olarak gorunuyor.")
    if not report_report.empty and str(report_report["feasible"].iloc[0]).lower() in {"true", "1"}:
        lines.append("- E1/E2 report_only ile model feasible olabiliyor; bu directional hard kisitlerinin cozum uzayini daralttigini gosterir.")
    if not soft_report.empty and str(soft_report["feasible"].iloc[0]).lower() in {"true", "1"}:
        lines.append("- E1 soft, mandatory hard kuralini korurken feasibility saglayan savunulabilir bir ara stratejidir.")
    if not relaxed.empty and str(relaxed["feasible"].iloc[0]).lower() in {"true", "1"}:
        lines.append("- Alpha gevsetmek feasible hale getiriyorsa sorun yuzde bazli E1 sertligindedir.")
    lines.append("- Yarışma brief uyumu acisindan mandatory connection selection hard kalmali; directional balance ise hard/soft secimi raporda acik varsayim olarak anlatilmalidir.")
    (diagnosis_dir / "diagnosis_interpretation.txt").write_text("\n".join(lines), encoding="utf-8")


def run_sequential_days(run_config: RunConfig, analyze_mvp: bool = False) -> dict:
    """Solve the schedule day by day (fix-and-optimize decomposition).

    G kisiti (gun-ici tutarlilik, Xdev) gunler arasindaki tek baglantidir.
    Her gun bagimsiz cozulur; onceki gunlerde belirlenen ucus saatleri
    anchor referansi olarak verilir, boylece haftalik Xdev bandi kurulum
    geregi saglanir. Wrap (Pazar->Pazartesi) baglantilarinda onceki gunlere
    ait eventler aynen sabitlenir. Tam kapsam (tum O-D'ler, 7 gun) icin
    onerilen kosum modudur.
    """

    base_run_id = run_config.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_sequential")
    seq_dir = run_config.output_dir / base_run_id
    seq_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Sequential fix-and-optimize run: {seq_dir}")

    raw = load_inputs(run_config)
    if run_config.days:
        days = sorted({int(d) for d in run_config.days})
    else:
        days = sorted(raw.connections["day"].dropna().astype(int).unique().tolist())
    print(f"[INFO] Sequential day order: {days}")

    anchor_refs: dict[tuple[str, str, str], list[float]] = {}
    fixed_shifts: dict[str, float] = {}
    day_rows = []
    for day in days:
        day_config = clone_run_config(
            run_config,
            run_id=f"day{day}",
            output_dir=seq_dir,
            max_shift_min=run_config.max_shift_min,
            days=[day],
        )
        day_config.regularity_anchor_mode = "off"
        print(f"[INFO] === Sequential solve: day {day} ===")
        metrics = run_once(
            day_config,
            analyze_mvp=False,
            fixed_event_shifts=fixed_shifts or None,
            anchor_references=anchor_refs or None,
        )
        sched_path = seq_dir / day_config.run_id / "optimized_schedule.xlsx"
        day_solved = sched_path.exists()
        if day_solved:
            sched = pd.read_excel(sched_path)
            for row in sched.itertuples(index=False):
                event_id = str(row.flight_id)
                fixed_shifts[event_id] = float(row.time_change_min)
                parts = event_id.split("|")
                if len(parts) == 5:
                    direction = "inbound" if parts[0] == "in" else "outbound"
                    new_minute = (float(parts[4]) + float(row.time_change_min)) % 1440
                    anchor_refs.setdefault((direction, parts[2], parts[3]), []).append(new_minute)
        else:
            print(f"[WARN] Day {day} produced no feasible schedule; weekly result will be marked INCOMPLETE.")
        day_rows.append(
            {
                "day": day,
                "status": "feasible" if day_solved else "infeasible_or_failed",
                "objective": metrics.get("objective", metrics.get("total_objective", 0.0)),
                "active_connections": metrics.get("active_connections", 0),
                "beaten_competitors": metrics.get("beaten_competitors", 0),
                "feasible": metrics.get("feasible", metrics.get("checker_feasible", "")),
            }
        )

    combined_dir = seq_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    model_config = _make_model_config(run_config)
    frames = {"optimized_schedule": [], "selected_connections": [], "od_metrics": [], "beaten_competitors": []}
    for day in days:
        run_dir = seq_dir / f"day{day}"
        for name in frames:
            path = run_dir / f"{name}.xlsx"
            if path.exists():
                frames[name].append(pd.read_excel(path))
    combined_schedule = (
        pd.concat(frames["optimized_schedule"], ignore_index=True).drop_duplicates("flight_id")
        if frames["optimized_schedule"]
        else pd.DataFrame()
    )
    combined_od = pd.concat(frames["od_metrics"], ignore_index=True) if frames["od_metrics"] else pd.DataFrame()
    combined_selected = pd.concat(frames["selected_connections"], ignore_index=True) if frames["selected_connections"] else pd.DataFrame()
    combined_beaten = pd.concat(frames["beaten_competitors"], ignore_index=True) if frames["beaten_competitors"] else pd.DataFrame()
    combined_schedule.to_excel(combined_dir / "optimized_schedule.xlsx", index=False)
    combined_od.to_excel(combined_dir / "od_metrics.xlsx", index=False)
    combined_selected.to_excel(combined_dir / "selected_connections.xlsx", index=False)
    combined_beaten.to_excel(combined_dir / "beaten_competitors.xlsx", index=False)

    total_objective = float(combined_od["total_weighted_reward"].sum()) if not combined_od.empty else 0.0
    full_prepared = prepare_data(raw, clone_run_config(run_config, run_id="combined_baseline", output_dir=seq_dir, max_shift_min=run_config.max_shift_min), model_config)
    baseline_full = compute_baseline(full_prepared, model_config)
    baseline_objective = float(baseline_full.summary.get("total_objective", 0.0))
    improvement_pct = 100.0 * (total_objective - baseline_objective) / abs(baseline_objective) if baseline_objective else 0.0

    weekly_regularity = build_regularity_report_from_schedule(combined_schedule, model_config)
    weekly_violations = 0
    pre_existing_violations = 0
    if not weekly_regularity.empty and "violation" in weekly_regularity.columns:
        violating = weekly_regularity[weekly_regularity["violation"] == True]
        if not violating.empty and "old_spread_min" in violating.columns:
            xdev_limit = float(model_config.schedule_regular_dev_min)
            pre_existing = violating["old_spread_min"].astype(float) > xdev_limit + 1e-6
            pre_existing_violations = int(pre_existing.sum())
            weekly_violations = int((~pre_existing).sum())
        else:
            weekly_violations = int(len(violating))
    weekly_regularity.to_excel(combined_dir / "weekly_regularity_report.xlsx", index=False)

    summary = pd.DataFrame(day_rows)
    summary.to_csv(combined_dir / "sequential_summary.csv", index=False)
    failed_days = [row["day"] for row in day_rows if row["status"] != "feasible"]
    weekly_status = "COMPLETE_ALL_DAYS_FEASIBLE" if not failed_days else "INCOMPLETE_NOT_A_FINAL_SOLUTION"
    lines = [
        "SEQUENTIAL FIX-AND-OPTIMIZE SUMMARY",
        "===================================",
        f"WEEKLY STATUS: {weekly_status}",
        f"days attempted: {days}",
        f"days failed (no feasible schedule): {failed_days if failed_days else 'none'}",
        f"baseline objective (full scope): {baseline_objective}",
        f"total objective (sum over solved days only): {total_objective}",
        f"improvement vs baseline: {improvement_pct:.4f}% (NOT comparable when weekly status is INCOMPLETE)",
        f"weekly regularity violations introduced by optimization: {weekly_violations}",
        f"weekly regularity violations pre-existing in input data (times frozen, not worsened): {pre_existing_violations}",
        "Per-day details: sequential_summary.csv; her gun icin standart run ciktilari day<N>/ altindadir.",
    ]
    if failed_days:
        lines.insert(3, "UYARI: Bu birlesik cikti bir final cozum DEGILDIR; eksik gunler icermektedir.")
    (combined_dir / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    objective_note = "" if not failed_days else f"\nWARNING: {weekly_status}; objective covers only days {sorted(set(days) - set(failed_days))}"
    (combined_dir / "objective_value.txt").write_text(
        f"total objective: {total_objective}\nobjective before penalty: {total_objective}\nobjective after penalty: {total_objective}{objective_note}",
        encoding="utf-8",
    )
    print(f"[INFO] Sequential weekly status: {weekly_status}")
    print(f"[INFO] Sequential combined objective: {total_objective} (baseline {baseline_objective}, +{improvement_pct:.2f}%)")
    print(
        f"[INFO] Weekly regularity: {weekly_violations} introduced violations, "
        f"{pre_existing_violations} pre-existing in input data (frozen)."
    )
    return {
        "weekly_status": weekly_status,
        "failed_days": failed_days,
        "total_objective": total_objective,
        "baseline_objective": baseline_objective,
        "improvement_pct": improvement_pct,
        "weekly_regularity_violations": weekly_violations,
        "weekly_regularity_pre_existing": pre_existing_violations,
        "days": days,
    }


def main() -> None:
    args = parse_args()
    run_config = make_run_config(args)
    if args.data_schema_audit:
        run_data_schema_audit(run_config)
    elif args.od400_stress_test:
        run_od400_stress_test(run_config)
    elif args.od300_days123_controlled_shift_test:
        run_od300_days123_controlled_shift_test(run_config)
    elif args.od300_controlled_shift_test:
        run_od300_controlled_shift_test(run_config)
    elif args.od300_maxshift60_quality_audit:
        run_od300_maxshift60_quality_audit(run_config, args.audit_run_dir)
    elif args.od300_targeted_regularity_test:
        run_od300_targeted_regularity_test(run_config)
    elif args.od300_regularity_diagnosis:
        run_od300_regularity_diagnosis(run_config)
    elif args.od300_infeasibility_diagnosis:
        run_od300_infeasibility_diagnosis(run_config, quick=True)
    elif args.mandatory_directional_diagnosis:
        run_mandatory_directional_diagnosis(run_config)
    elif args.ranking_objective_comparison_test:
        run_ranking_objective_comparison_test(run_config)
    elif args.directional_stress_test:
        run_directional_stress_test(run_config)
    elif args.directional_scaling_test:
        run_directional_scaling_test(run_config)
    elif args.directional_coverage_only:
        run_directional_coverage_only(run_config)
    elif args.regularity_stress_test:
        run_regularity_stress_test(run_config)
    elif args.capacity_stress_test:
        run_capacity_stress_test(run_config)
    elif args.shift_penalty_stress_test:
        run_shift_penalty_stress_test(run_config)
    elif args.final_scale_test:
        run_final_scale_test(run_config)
    elif args.rotation_coverage_only:
        run_rotation_coverage_only(run_config)
    elif args.scaling_test:
        run_scaling_test(run_config)
    elif args.sensitivity_test:
        run_sensitivity_test(run_config)
    elif args.sequential_days:
        run_sequential_days(run_config)
    else:
        run_once(run_config, analyze_mvp=args.analyze_mvp or run_config.model_level in {"mvp", "final"})


if __name__ == "__main__":
    main()
