"""Central configuration and assumptions for the optimization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


MAX_SHIFT_MINUTES = 60
MIN_GROUND_TIME_MINUTES = 45
HUB_BUCKET_SIZE_MINUTES = 10
HUB_DEPARTURE_CAPACITY_PER_BUCKET = 10
HUB_ARRIVAL_CAPACITY_PER_BUCKET = 15
SCHEDULE_REGULARITY_XDEV_MINUTES = 15
XDEV_MINUTES = 15
REGULARITY_MODE = "hard"
REGULARITY_PENALTY_WEIGHT = 100.0
REGULARITY_ABS_TOLERANCE_MINUTES = 0
REGULARITY_RELAX_MODE = "none"
REGULARITY_TARGETED_MAX_RELAX_MINUTES = 30
ALPHA_DIRECTIONAL_BALANCE = 0.20
GAMMA_BEST_JOURNEY_MINUTES = 30
DIRECTIONAL_E1_MODE = "hard"
DIRECTIONAL_E1_PENALTY_WEIGHT = 100.0
DIRECTIONAL_E1_FORMULA = "ratio_total"
DIRECTIONAL_E1_ABS_TOLERANCE = 0
DIRECTIONAL_E2_MODE = "report_only"
DIRECTIONAL_E2_PENALTY_WEIGHT = 100.0
SCHEDULE_DISRUPTION_PENALTY_WEIGHT = 0.0
SHIFT_PENALTY_MODE = "none"
SHIFT_FREE_THRESHOLD_MINUTES = 5
MANDATORY_CONNECTION_SELECTION = False
RANKING_OBJECTIVE_MODE = "proxy"
# "carrier": her O-D/gun icin rakip = ayri tasiyici, Tcomp = tasiyicinin en iyi (min) suresi.
# "row": eski davranis; tablodaki her rakip satiri ayri rakip sayilir.
COMPETITOR_MODE = "carrier"
# "table": aday TK baglantilari yalnizca girdi tablosundaki satirlar.
# "cross_product": ayni O-D/gun icindeki tum TK inbound x outbound kombinasyonlari
# (gap erisilebilirlik on filtresiyle) aday olarak uretilir.
CANDIDATE_MODE = "cross_product"
# "auto": kismi gun kosularinda kapsam disi gunlerdeki sabit saatlere gore shift
# degiskenine +-Xdev cap sinirlari uygulanir (haftalik regularity guvencesi).
# "off": cap uygulanmaz (sequential fix-and-optimize kosulari kendi cozer).
REGULARITY_ANCHOR_MODE = "auto"
# "flight_week": ayni ucus no/yon/istasyon icin TUM haftaya tek shift degiskeni.
#   Tarife duzenliligi (G, Xdev) yapisal olarak garanti edilir; gunler-arasi
#   infeasibility olusamaz. Haftalik paterni zaten Xdev'den fazla dagilan
#   ucuslar event bazinda cozulur (duzeltilebilirse) veya dondurulur.
# "event": her ucus orneginin gunune ozel shift (eski davranis).
SHIFT_GRANULARITY = "flight_week"
# "enforce": hub kapasitesi kova-atama binary'leri + big-M kisitlariyla cozum
#   sirasinda aktif tutulur (tam/kesin, ama buyuk ve LP'yi bozan big-M'ler).
# "postcheck": kova makinesi cozum sirasinda kurulmaz (model cok daha kucuk/hizli),
#   kapasite bagimsiz checker ile dogrulanir. Kapasite pratikte neredeyse hic
#   baglamadigi icin (rezidüel bol) cozum F kisitini zaten saglar; checker ihlal
#   raporlarsa "enforce" ile yeniden cozulmelidir.
HUB_CAPACITY_MODE = "enforce"
SOLVER_RANDOM_SEED = 0
SOLVER_THREADS = 1


@dataclass(frozen=True)
class ModelConfig:
    """Business rules and default assumptions used by baseline and models."""

    hub: str = "IST"
    min_connection_min: int = 60
    max_connection_min: int = 300
    min_ground_time_min: int = MIN_GROUND_TIME_MINUTES
    max_shift_min: int = MAX_SHIFT_MINUTES
    schedule_regular_dev_min: int = SCHEDULE_REGULARITY_XDEV_MINUTES
    regularity_mode: str = REGULARITY_MODE
    regularity_penalty_weight: float = REGULARITY_PENALTY_WEIGHT
    regularity_abs_tolerance_min: int = REGULARITY_ABS_TOLERANCE_MINUTES
    regularity_relax_mode: str = REGULARITY_RELAX_MODE
    regularity_targeted_max_relax_min: int = REGULARITY_TARGETED_MAX_RELAX_MINUTES
    directional_alpha: float = ALPHA_DIRECTIONAL_BALANCE
    directional_gamma_min: int = GAMMA_BEST_JOURNEY_MINUTES
    directional_e1_mode: str = DIRECTIONAL_E1_MODE
    directional_e1_penalty_weight: float = DIRECTIONAL_E1_PENALTY_WEIGHT
    directional_e1_formula: str = DIRECTIONAL_E1_FORMULA
    directional_e1_abs_tolerance: int = DIRECTIONAL_E1_ABS_TOLERANCE
    directional_e2_mode: str = DIRECTIONAL_E2_MODE
    directional_e2_penalty_weight: float = DIRECTIONAL_E2_PENALTY_WEIGHT
    allow_soft_directional_violations: bool = False
    shift_penalty_mode: str = SHIFT_PENALTY_MODE
    shift_penalty_weight: float = SCHEDULE_DISRUPTION_PENALTY_WEIGHT
    shift_free_threshold_min: int = SHIFT_FREE_THRESHOLD_MINUTES
    mandatory_connection_selection: bool = MANDATORY_CONNECTION_SELECTION
    ranking_objective_mode: str = RANKING_OBJECTIVE_MODE
    competitor_mode: str = COMPETITOR_MODE
    candidate_mode: str = CANDIDATE_MODE
    regularity_anchor_mode: str = REGULARITY_ANCHOR_MODE
    shift_granularity: str = SHIFT_GRANULARITY
    hub_capacity_mode: str = HUB_CAPACITY_MODE
    solver_random_seed: int = SOLVER_RANDOM_SEED
    solver_threads: int = SOLVER_THREADS
    bucket_size_min: int = HUB_BUCKET_SIZE_MINUTES
    departure_bucket_capacity: int = HUB_DEPARTURE_CAPACITY_PER_BUCKET
    arrival_bucket_capacity: int = HUB_ARRIVAL_CAPACITY_PER_BUCKET
    # 64: pratikte O-D/gun basina aday sayisindan buyuk; slot_fill esitligi
    # mandatory selection ile hicbir zaman yapay tavana takilmaz (decay^k ~ 0).
    max_connection_slots: int = 64
    connection_reward_decay: float = 0.5
    competitor_beating_reward: float = 0.05
    default_importance: float = 1.0
    time_limit_sec: int = 600
    mip_gap: Optional[float] = 0.01


@dataclass(frozen=True)
class InputFiles:
    """Default input file names."""

    competitor_connections: str = "O&D Rakip Bağlantı Tablosu.xlsx"
    passenger_data: str = "Yolcu Verisi_masked.xlsx"
    ranking_input: str = "change_ranking_input.xlsx"
    flight_pairs: str = "Flight Pairs.xlsx"


@dataclass
class RunConfig:
    """Runtime configuration parsed from CLI."""

    data_dir: Path
    output_dir: Path
    model_level: str = "baseline"
    solver: str = "highs"
    time_limit: int = 600
    run_id: Optional[str] = None
    max_od: Optional[int] = None
    days: Optional[list[int]] = None
    top_importance_only: bool = False
    max_shift_min: int = MAX_SHIFT_MINUTES
    hub_arrival_capacity: int = HUB_ARRIVAL_CAPACITY_PER_BUCKET
    hub_departure_capacity: int = HUB_DEPARTURE_CAPACITY_PER_BUCKET
    hub_bucket_size_min: int = HUB_BUCKET_SIZE_MINUTES
    regularity_xdev_min: int = SCHEDULE_REGULARITY_XDEV_MINUTES
    regularity_mode: str = REGULARITY_MODE
    regularity_penalty_weight: float = REGULARITY_PENALTY_WEIGHT
    regularity_abs_tolerance_min: int = REGULARITY_ABS_TOLERANCE_MINUTES
    regularity_relax_mode: str = REGULARITY_RELAX_MODE
    regularity_targeted_max_relax_min: int = REGULARITY_TARGETED_MAX_RELAX_MINUTES
    directional_alpha: float = ALPHA_DIRECTIONAL_BALANCE
    directional_gamma_min: int = GAMMA_BEST_JOURNEY_MINUTES
    directional_e1_mode: str = DIRECTIONAL_E1_MODE
    directional_e1_penalty_weight: float = DIRECTIONAL_E1_PENALTY_WEIGHT
    directional_e1_formula: str = DIRECTIONAL_E1_FORMULA
    directional_e1_abs_tolerance: int = DIRECTIONAL_E1_ABS_TOLERANCE
    directional_e2_mode: str = DIRECTIONAL_E2_MODE
    directional_e2_penalty_weight: float = DIRECTIONAL_E2_PENALTY_WEIGHT
    allow_soft_directional_violations: bool = False
    shift_penalty_mode: str = SHIFT_PENALTY_MODE
    shift_penalty_weight: float = SCHEDULE_DISRUPTION_PENALTY_WEIGHT
    shift_free_threshold_min: int = SHIFT_FREE_THRESHOLD_MINUTES
    mandatory_connection_selection: bool = MANDATORY_CONNECTION_SELECTION
    ranking_objective_mode: str = RANKING_OBJECTIVE_MODE
    competitor_mode: str = COMPETITOR_MODE
    candidate_mode: str = CANDIDATE_MODE
    regularity_anchor_mode: str = REGULARITY_ANCHOR_MODE
    shift_granularity: str = SHIFT_GRANULARITY
    hub_capacity_mode: str = HUB_CAPACITY_MODE
    solver_random_seed: int = SOLVER_RANDOM_SEED
    solver_threads: int = SOLVER_THREADS
    two_phase: bool = False
    warmstart: bool = False
    warmstart_seconds: int = 120
    competitor_connections_file: str = InputFiles.competitor_connections
    passenger_data_file: str = InputFiles.passenger_data
    ranking_input_file: str = InputFiles.ranking_input
    flight_pairs_file: str = InputFiles.flight_pairs


REQUIRED_CONNECTION_COLUMNS = [
    "Cr1",
    "Carrier Name",
    "Dep1",
    "Arr1",
    "FlNo1",
    "Arr Time",
    "Cr2",
    "Dep2",
    "Arr2",
    "FlNo2",
    "Dep Time",
    "Gate-to-Gate Uçuş Süresi",
    "O&D",
    "Gün",
]

REQUIRED_PASSENGER_COLUMNS = [
    "Orig Airport Code",
    "Dest Airport Code",
    "OD importance factor",
]

REQUIRED_RANKING_COLUMNS = ["Rakip Sayısı", "İlk Rank", "Son Rank", "Weight"]

REQUIRED_FLIGHT_PAIR_COLUMNS = ["FlNo", "Orig", "Dest", "Pair"]
