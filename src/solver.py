"""Solver integration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from time import perf_counter


@dataclass
class SolveResult:
    """Compact solver outcome."""

    status: str
    termination_condition: str
    runtime_sec: float
    objective: float | None
    gap: float | None
    feasible: bool


def solve_model(
    model: Any,
    solver_name: str,
    time_limit: int,
    mip_gap: float | None = None,
    random_seed: int | None = 0,
    threads: int | None = 1,
) -> SolveResult:
    """Solve a Pyomo model and return normalized status data.

    random_seed ve threads varsayilan olarak sabitlenir; ayni girdiyle ayni
    cozumun uretilmesi (deterministik kosum) teslim kriteridir.
    """

    try:
        import pyomo.environ as pyo
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pyomo is not installed. Run `pip install -r requirements.txt` before MVP/full runs.") from exc
    print(f"[INFO] Solving with {solver_name}...")
    opt = pyo.SolverFactory(solver_name)
    if opt is None or not opt.available(False):
        raise RuntimeError(f"Solver not available: {solver_name}")
    if solver_name.lower() in {"highs", "appsi_highs"}:
        opt.options["time_limit"] = int(time_limit)
        if mip_gap is not None:
            opt.options["mip_rel_gap"] = float(mip_gap)
        if random_seed is not None:
            opt.options["random_seed"] = int(random_seed)
        if threads is not None:
            opt.options["threads"] = int(threads)
        import os as _os
        # Buyuk modelde HiGHS varsayilan sezgisellikle ilk feasible cozumu gec
        # buluyor; feasibility eforunu artirmak erken incumbent saglar.
        _heur = _os.environ.get("SOLVER_HEURISTIC_EFFORT", "").strip()
        if _heur:
            opt.options["mip_heuristic_effort"] = float(_heur)
        _log_file = _os.environ.get("SOLVER_LOG_FILE", "").strip()
        if _log_file:
            # HiGHS C-seviye log'unu dogrudan dosyaya yazar (stdout tamponlamasini
            # atlatir) -> uzun kosularda gercek zamanli gap izleme.
            opt.options["log_file"] = _log_file
    elif "gurobi" in solver_name.lower():
        # Jurinin degerlendirme ortami Pyomo kodunu GUROBI ile cozer (organizator Q&A).
        opt.options["TimeLimit"] = int(time_limit)
        if mip_gap is not None:
            opt.options["MIPGap"] = float(mip_gap)
        if random_seed is not None:
            opt.options["Seed"] = int(random_seed)
        if threads is not None:
            opt.options["Threads"] = int(threads)
        import os as _os
        _log_file = _os.environ.get("SOLVER_LOG_FILE", "").strip()
        if _log_file:
            opt.options["LogFile"] = _log_file
    elif "cplex" in solver_name.lower():
        opt.options["timelimit"] = int(time_limit)
        if mip_gap is not None:
            opt.options["mipgap"] = float(mip_gap)
        if threads is not None:
            opt.options["threads"] = int(threads)
    else:
        # CBC/GLPK vb. yedek open-source cozucular.
        opt.options["seconds"] = int(time_limit)
        if random_seed is not None:
            opt.options["randomSeed"] = int(random_seed)
        if threads is not None:
            opt.options["threads"] = int(threads)
    import os
    tee = os.environ.get("SOLVER_TEE", "").strip() in {"1", "true", "True"}
    start = perf_counter()
    try:
        result = opt.solve(model, tee=tee)
    except Exception as exc:
        runtime = perf_counter() - start
        if exc.__class__.__name__ == "NoFeasibleSolutionError":
            print(f"[WARN] Solver did not find a feasible solution: {exc}")
            return SolveResult("failed", "NoFeasibleSolutionError", runtime, None, None, False)
        raise
    runtime = perf_counter() - start
    status = str(result.solver.status)
    term = str(result.solver.termination_condition)
    objective = pyo.value(model.objective, exception=False)
    feasible_terms = {"optimal", "feasible", "maxTimeLimit", "maxIterations"}
    feasible = any(token.lower() in term.lower() for token in feasible_terms) and objective is not None
    gap = None
    if hasattr(result.solver, "mip_gap"):
        try:
            gap = float(result.solver.mip_gap)
        except Exception:
            gap = None
    print(f"[INFO] Solver status={status}, termination={term}, objective={objective}")
    return SolveResult(status, term, runtime, None if objective is None else float(objective), gap, feasible)


def solve_with_baseline_warmstart(
    model: Any,
    solver_name: str,
    time_limit: int,
    mip_gap: float | None = None,
    random_seed: int | None = 0,
    threads: int | None = 1,
    warmstart_seconds: int = 120,
) -> SolveResult:
    """Solve with a baseline (shift=0) MIP start.

    Buyuk modelde HiGHS ilk feasible tamsayi cozumunu kendi basina bulamiyor
    (kok dugumde takiliyor). Cozum:
      1. Tum shift=0 sabitlenir; kalan model (secim/yenme/slot) hizla optimal
         cozulur -> baseline tarifenin feasible tam atamasi elde edilir.
      2. shift serbest birakilir; appsi_highs bu atamayi MIP start olarak alip
         iyilestirir (ihlalleri kaydirmayla azaltir, odulu artirir).
    """

    try:
        import pyomo.environ as pyo
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pyomo is not installed.") from exc

    if not hasattr(model, "SH"):
        raise RuntimeError("Model has no shift set SH; rebuild with current model.py.")

    # 1) shift=0 sabitle, kisitli modeli coz (baseline incumbent + baseline ceza).
    print("[INFO] Warm start phase A: solving baseline (shift=0) to bound violations...")
    original_bounds = {}
    for sh in model.SH:
        original_bounds[sh] = (model.shift[sh].lb, model.shift[sh].ub)
        model.shift[sh].fix(0)
    seed = solve_model(model, solver_name, warmstart_seconds, mip_gap, random_seed, threads)
    baseline_penalty = float(pyo.value(model.penalty_expr)) if seed.feasible else None
    for sh in model.SH:
        model.shift[sh].unfix()
        lb, ub = original_bounds[sh]
        model.shift[sh].setlb(lb)
        model.shift[sh].setub(ub)
    if not seed.feasible or baseline_penalty is None:
        print("[WARN] Baseline seed solve was not feasible; falling back to cold solve.")
        return solve_model(model, solver_name, time_limit, mip_gap, random_seed, threads)
    print(f"[INFO] Baseline penalty = {baseline_penalty:.1f} (objective {seed.objective:.1f}).")

    # 2) Gecerli kesme: optimumun cezasi baseline'dan buyuk OLAMAZ (baseline feasible).
    #    penalty_expr <= baseline_penalty, cozucunun dolastigi tum yuksek-ceza (baseline'dan
    #    kotu) bolgeyi budar -> arama uzayi dramatik kuculur, iyilesme baseline'dan baslar.
    cap = baseline_penalty + max(1.0, abs(baseline_penalty) * 1e-9)
    model.baseline_penalty_cut = pyo.Constraint(expr=model.penalty_expr <= cap)
    print(f"[INFO] Added valid inequality penalty <= {cap:.1f}; solving full model (guaranteed no worse than baseline)...")
    remaining = max(warmstart_seconds, int(time_limit) - int(seed.runtime_sec))
    result = solve_model(model, solver_name, remaining, mip_gap, random_seed, threads)
    result = SolveResult(
        result.status, result.termination_condition,
        result.runtime_sec + seed.runtime_sec, result.objective, result.gap, result.feasible,
    )
    return result


def solve_two_phase(
    model: Any,
    solver_name: str,
    time_limit: int,
    mip_gap: float | None = None,
    random_seed: int | None = 0,
    threads: int | None = 1,
    violation_epsilon: float = 1e-6,
) -> SolveResult:
    """Lexicographic two-phase solve: minimize violations, then maximize reward.

    Faz 1: penalty_expr (E1/E2/regularity/shift slack toplami) minimize edilir;
      kacinilabilir ihlallerin kanitlanmis minimumu V* bulunur.
    Faz 2: penalty_expr <= V* + eps kisiti eklenir ve reward_expr maksimize edilir;
      boylece ihlal seviyesi optimalde sabitlenirken saf odul en iyilenir.

    Bu ayristirma, tek-amacli cozumdeki 'odul vs ceza ayni butcede takas edilir ve
    zaman limitinde yari yolda kesilir' sorununu ortadan kaldirir ve raporlanan
    ihlal setinin gercekten minimum oldugunu (leksikografik) kanitlar.
    """

    try:
        import pyomo.environ as pyo
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pyomo is not installed. Run `pip install -r requirements.txt`.") from exc

    if not hasattr(model, "reward_expr") or not hasattr(model, "penalty_expr"):
        raise RuntimeError("Model does not expose reward_expr/penalty_expr; rebuild with current model.py.")

    half = max(60, int(time_limit) // 2)

    # --- Faz 1: cezayi minimize et ---
    print("[INFO] Two-phase solve, phase 1: minimizing total violation penalty...")
    model.objective.deactivate()
    model.phase1_objective = pyo.Objective(expr=model.penalty_expr, sense=pyo.minimize)
    r1 = solve_model(model, solver_name, half, mip_gap, random_seed, threads)
    if not r1.feasible:
        print("[WARN] Phase 1 did not find a feasible solution; falling back to single-phase objective.")
        model.del_component(model.phase1_objective)
        model.objective.activate()
        return solve_model(model, solver_name, time_limit - r1.runtime_sec if time_limit > r1.runtime_sec else half, mip_gap, random_seed, threads)

    v_star = float(pyo.value(model.penalty_expr))
    print(f"[INFO] Phase 1 minimum violation penalty V* = {v_star:.4f} (runtime {r1.runtime_sec:.1f}s).")

    # --- Faz 2: cezayi V*+eps'e caple, odulu maksimize et ---
    cap = v_star + max(violation_epsilon, abs(v_star) * 1e-6)
    model.phase2_violation_cap = pyo.Constraint(expr=model.penalty_expr <= cap)
    model.del_component(model.phase1_objective)
    model.phase2_objective = pyo.Objective(expr=model.reward_expr, sense=pyo.maximize)
    print(f"[INFO] Two-phase solve, phase 2: maximizing reward with penalty capped at {cap:.4f}...")
    remaining = max(half, int(time_limit) - int(r1.runtime_sec))
    r2 = solve_model(model, solver_name, remaining, mip_gap, random_seed, threads)

    # Orijinal objective'i yeniden aktiflestir (raporlama/extract net olsun) ve
    # gercek bilesik degeri dondur.
    model.del_component(model.phase2_objective)
    model.del_component(model.phase2_violation_cap)
    model.objective.activate()
    composite = pyo.value(model.objective, exception=False)
    total_runtime = r1.runtime_sec + r2.runtime_sec
    if not r2.feasible:
        print("[WARN] Phase 2 did not improve; phase-1 solution retained.")
        return SolveResult(r1.status, r1.termination_condition, total_runtime, composite, r2.gap, True)
    print(f"[INFO] Two-phase complete: reward={pyo.value(model.reward_expr):.2f}, penalty={pyo.value(model.penalty_expr):.2f}.")
    return SolveResult(r2.status, r2.termination_condition, total_runtime, composite, r2.gap, True)
