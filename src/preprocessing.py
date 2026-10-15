"""Preprocessing functions that derive model-ready tables."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ModelConfig, RunConfig
from .data_loader import RawData


@dataclass
class PreparedData:
    """Tables shared by baseline, model, output writer, and checker."""

    raw_connections: pd.DataFrame
    passengers: pd.DataFrame
    ranking: pd.DataFrame
    flight_pairs: pd.DataFrame
    tk_connections: pd.DataFrame
    competitor_connections: pd.DataFrame
    od_day: pd.DataFrame
    ranking_lookup: dict[tuple[int, int, int], float]
    # Filtre uygulanmamis tum baglanti satirlari: rezidüel hub kapasitesi ve
    # kapsam disi gun regularity capasi icin gereklidir.
    full_connections: pd.DataFrame | None = None
    # Tasiyici bazli agregasyon oncesi ham rakip satirlari (audit icin).
    competitor_connections_raw: pd.DataFrame | None = None


def prepare_data(raw: RawData, run_config: RunConfig, model_config: ModelConfig) -> PreparedData:
    """Filter and derive baseline/model tables."""

    print("[INFO] Preprocessing input tables...")
    full_connections = raw.connections.copy()
    connections = raw.connections.copy()
    if run_config.days:
        connections = connections[connections["day"].isin(run_config.days)].copy()

    passengers = raw.passengers.copy()
    passenger_map = passengers.set_index("od")["importance_factor"].to_dict()
    connections["importance_factor"] = connections["O&D"].map(passenger_map).fillna(model_config.default_importance)

    od_order = _select_od_order(connections, passengers, run_config)
    if run_config.max_od:
        selected = set(od_order[: run_config.max_od])
        connections = connections[connections["O&D"].isin(selected)].copy()

    tk_connections = connections[connections["is_tk"]].copy().reset_index(drop=True)
    competitor_connections = connections[~connections["is_tk"]].copy().reset_index(drop=True)

    competitor_connections_raw = competitor_connections.copy()
    if model_config.competitor_mode == "carrier":
        competitor_connections = _aggregate_competitors_by_carrier(competitor_connections)
        print(
            "[INFO] Competitor mode=carrier: "
            f"{len(competitor_connections_raw)} rows -> {len(competitor_connections)} carrier-level competitors."
        )

    if "candidate_source" not in tk_connections.columns:
        tk_connections["candidate_source"] = "table"
    if model_config.candidate_mode == "cross_product":
        # Evren tum O-D/gun anahtarlaridir (yalnizca rakip satiri olan ters yon
        # pazarlari dahil): boylece yonsel denge icin ters yon adaylari da uretilir.
        od_universe = connections[["O&D", "origin", "destination", "day", "importance_factor"]].drop_duplicates(
            ["O&D", "day"]
        )
        generated = _generate_cross_product_candidates(tk_connections, model_config, od_universe)
        if not generated.empty:
            tk_connections = pd.concat([tk_connections, generated], ignore_index=True)
            print(
                "[INFO] Candidate mode=cross_product: "
                f"added {len(generated)} generated inbound x outbound TK candidates."
            )

    od_day = _build_od_day_table(connections, tk_connections, competitor_connections, model_config)
    ranking_lookup = {
        (int(row.competitor_count), int(row.initial_rank), int(row.new_rank)): float(row.weight)
        for row in raw.ranking.itertuples(index=False)
    }

    print(f"[INFO] Prepared {od_day[['od', 'day']].drop_duplicates().shape[0]} OD/day records.")
    print(f"[INFO] TK candidate connections: {len(tk_connections)}")
    print(f"[INFO] Competitor connections: {len(competitor_connections)}")
    return PreparedData(
        raw_connections=connections.reset_index(drop=True),
        passengers=passengers,
        ranking=raw.ranking,
        flight_pairs=raw.flight_pairs,
        tk_connections=tk_connections,
        competitor_connections=competitor_connections,
        od_day=od_day,
        ranking_lookup=ranking_lookup,
        full_connections=full_connections,
        competitor_connections_raw=competitor_connections_raw,
    )


def _aggregate_competitors_by_carrier(competitor_connections: pd.DataFrame) -> pd.DataFrame:
    """Reduce competitor rows to one per (O&D, day, carrier) using the carrier's best time.

    Brief tanimindaki 'rakip k' bir tasiyicidir; bir tasiyiciyi yenmek, onun o
    O-D/gundeki en iyi (min journey time) alternatifini yenmek demektir.
    """

    if competitor_connections.empty:
        return competitor_connections
    out = competitor_connections.sort_values(
        ["O&D", "day", "Carrier Name", "journey_time_min"], na_position="last"
    ).drop_duplicates(subset=["O&D", "day", "Carrier Name"], keep="first").reset_index(drop=True)
    out["competitor_id"] = (
        out["Carrier Name"].astype(str).str.replace(r"\s+", "_", regex=True)
        + "|"
        + out["O&D"].astype(str)
        + "|"
        + out["day"].astype(str)
    )
    return out


def _generate_cross_product_candidates(
    tk_connections: pd.DataFrame,
    model_config: ModelConfig,
    od_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate same-day TK inbound x outbound pairs missing from the input table.

    Tablo, mevcut tarifeye gore makul gap'li kombinasyonlari listeler; saatler
    kaydiginda tabloda olmayan ciftler pencereye girebilir. Bu fonksiyon ayni
    O-D/gun icin tum kombinasyonlari uretir ve gap erisilebilirlik on filtresi
    uygular. Yolculuk suresi TIB + gap + TOB olarak bacak surelerinden kurulur.
    """

    if tk_connections.empty:
        return pd.DataFrame()
    reach = 2 * float(model_config.max_shift_min)
    lo = float(model_config.min_connection_min) - reach
    hi = float(model_config.max_connection_min) + reach

    inbound_cols = ["day", "origin", "inbound_flight", "arr_time", "arr_min", "elapsed_time_1_min"]
    outbound_cols = ["day", "destination", "outbound_flight", "dep_time", "dep_min", "elapsed_time_2_min"]
    inbound = (
        tk_connections[inbound_cols]
        .groupby(["day", "origin", "inbound_flight", "arr_min"], as_index=False)
        .agg(arr_time=("arr_time", "first"), inbound_block_min=("elapsed_time_1_min", "median"))
    )
    outbound = (
        tk_connections[outbound_cols]
        .groupby(["day", "destination", "outbound_flight", "dep_min"], as_index=False)
        .agg(dep_time=("dep_time", "first"), outbound_block_min=("elapsed_time_2_min", "median"))
    )
    inbound = inbound[inbound["inbound_block_min"].notna()]
    outbound = outbound[outbound["outbound_block_min"].notna()]
    if inbound.empty or outbound.empty:
        print("[WARN] Cross-product generation skipped: leg block times (ElapsedTime1/2) unavailable.")
        return pd.DataFrame()

    existing = set(
        zip(
            tk_connections["day"].astype(int),
            tk_connections["O&D"].astype(str),
            tk_connections["inbound_flight"].astype(str),
            tk_connections["arr_min"].astype(int),
            tk_connections["outbound_flight"].astype(str),
            tk_connections["dep_min"].astype(int),
        )
    )
    if od_universe is None:
        od_universe = tk_connections[["O&D", "origin", "destination", "day", "importance_factor"]].drop_duplicates(
            ["O&D", "day"]
        )

    pairs = od_universe.merge(inbound, on=["day", "origin"], how="inner").merge(
        outbound, on=["day", "destination"], how="inner"
    )
    if pairs.empty:
        return pd.DataFrame()
    pairs["connection_gap_min"] = pairs["dep_min"].astype(float) - pairs["arr_min"].astype(float)
    pairs = pairs[(pairs["connection_gap_min"] >= lo) & (pairs["connection_gap_min"] <= hi)].copy()
    if pairs.empty:
        return pd.DataFrame()
    key = list(
        zip(
            pairs["day"].astype(int),
            pairs["O&D"].astype(str),
            pairs["inbound_flight"].astype(str),
            pairs["arr_min"].astype(int),
            pairs["outbound_flight"].astype(str),
            pairs["dep_min"].astype(int),
        )
    )
    pairs = pairs[[k not in existing for k in key]].copy()
    if pairs.empty:
        return pd.DataFrame()

    pairs["journey_time_min"] = (
        pairs["inbound_block_min"].astype(float)
        + pairs["connection_gap_min"].astype(float)
        + pairs["outbound_block_min"].astype(float)
    )
    pairs["elapsed_time_1_min"] = pairs["inbound_block_min"]
    pairs["elapsed_time_2_min"] = pairs["outbound_block_min"]
    pairs["journey_time_source"] = "cross_product_leg_sum"
    pairs["candidate_source"] = "generated"
    pairs["is_tk"] = True
    pairs["Cr1"] = "TK"
    pairs["Cr2"] = "TK"
    pairs["Carrier Name"] = "Turkish Airlines"
    pairs["carrier_code"] = "TK"
    pairs["connection_gap_week_wrapped"] = False
    pairs = pairs.reset_index(drop=True)
    pairs["connection_id"] = [f"gen_{i:06d}" for i in range(len(pairs))]
    pairs["competitor_id"] = "TK_" + pairs["connection_id"]
    keep_cols = [
        "connection_id",
        "competitor_id",
        "O&D",
        "origin",
        "destination",
        "day",
        "importance_factor",
        "inbound_flight",
        "outbound_flight",
        "arr_time",
        "dep_time",
        "arr_min",
        "dep_min",
        "connection_gap_min",
        "connection_gap_week_wrapped",
        "journey_time_min",
        "journey_time_source",
        "elapsed_time_1_min",
        "elapsed_time_2_min",
        "candidate_source",
        "is_tk",
        "Cr1",
        "Cr2",
        "Carrier Name",
        "carrier_code",
    ]
    return pairs[keep_cols]


def connection_reward(count: int, max_slots: int, decay: float) -> float:
    """Diminishing connection-count reward."""

    slots = min(max(0, int(count)), max_slots)
    return sum(decay**idx for idx in range(slots))


def _select_od_order(connections: pd.DataFrame, passengers: pd.DataFrame, run_config: RunConfig) -> list[str]:
    if run_config.top_importance_only:
        ordered = passengers.sort_values("importance_factor", ascending=False)["od"].tolist()
        present = set(connections["O&D"].unique())
        return [od for od in ordered if od in present]
    return sorted(connections["O&D"].dropna().unique().tolist())


def _build_od_day_table(
    connections: pd.DataFrame,
    tk_connections: pd.DataFrame,
    competitor_connections: pd.DataFrame,
    model_config: ModelConfig,
) -> pd.DataFrame:
    keys = connections[["O&D", "origin", "destination", "day", "importance_factor"]].drop_duplicates()
    comp = (
        competitor_connections.groupby(["O&D", "day"])
        .agg(
            competitor_count=("connection_id", "count"),
            best_competitor_time_min=("journey_time_min", "min"),
        )
        .reset_index()
    )
    baseline_tk = tk_connections[
        (tk_connections["connection_gap_min"] >= model_config.min_connection_min)
        & (tk_connections["connection_gap_min"] <= model_config.max_connection_min)
    ]
    tk = (
        baseline_tk.groupby(["O&D", "day"])
        .agg(
            baseline_active_connections=("connection_id", "count"),
            old_best_journey_time=("journey_time_min", "min"),
            avg_connection_gap=("connection_gap_min", "mean"),
        )
        .reset_index()
    )
    od_day = keys.rename(columns={"O&D": "od"}).drop_duplicates(["od", "day"])
    od_day = od_day.merge(comp.rename(columns={"O&D": "od"}), on=["od", "day"], how="left")
    od_day = od_day.merge(tk.rename(columns={"O&D": "od"}), on=["od", "day"], how="left")
    od_day["competitor_count"] = od_day["competitor_count"].fillna(0).astype(int)
    od_day["baseline_active_connections"] = od_day["baseline_active_connections"].fillna(0).astype(int)
    od_day["old_best_journey_time"] = od_day["old_best_journey_time"].fillna(float("inf"))
    od_day["best_competitor_time_min"] = od_day["best_competitor_time_min"].fillna(float("inf"))
    od_day["avg_connection_gap"] = od_day["avg_connection_gap"].fillna(0.0)
    od_day["initial_beaten"] = 0
    for idx, row in od_day.iterrows():
        if row["baseline_active_connections"] > 0 and row["competitor_count"] > 0:
            mask = (competitor_connections["O&D"] == row["od"]) & (competitor_connections["day"] == row["day"])
            od_day.at[idx, "initial_beaten"] = int(
                (competitor_connections.loc[mask, "journey_time_min"] >= row["old_best_journey_time"]).sum()
            )
    od_day["initial_rank"] = (od_day["competitor_count"] - od_day["initial_beaten"] + 1).clip(lower=1)
    od_day["baseline_connection_reward"] = od_day["baseline_active_connections"].map(
        lambda c: connection_reward(c, model_config.max_connection_slots, model_config.connection_reward_decay)
    )
    return od_day.reset_index(drop=True)
