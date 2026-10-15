"""Input loading and robust type normalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    REQUIRED_CONNECTION_COLUMNS,
    REQUIRED_FLIGHT_PAIR_COLUMNS,
    REQUIRED_PASSENGER_COLUMNS,
    REQUIRED_RANKING_COLUMNS,
    RunConfig,
)


class DataLoadError(RuntimeError):
    """Raised when an input file, sheet, or required column is invalid."""


@dataclass
class RawData:
    """Raw normalized input tables."""

    connections: pd.DataFrame
    passengers: pd.DataFrame
    ranking: pd.DataFrame
    flight_pairs: pd.DataFrame


def _read_excel_first_sheet(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise DataLoadError(f"Input file not found: {path}")
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        raise DataLoadError(f"Could not read Excel file {path}: {exc}") from exc
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise DataLoadError(f"Missing required columns in {path.name}: {missing}")
    return df


def _missing_input_message(data_dir: Path, expected_files: list[str], found_files: list[str], missing_files: list[str]) -> str:
    tree = "\n".join(f"  - {name}" for name in expected_files)
    found = "\n".join(f"  - {name}" for name in found_files) if found_files else "  - <none>"
    missing = "\n".join(f"  - {name}" for name in missing_files)
    return (
        "Input file(s) not found.\n"
        f"Used data_dir: {data_dir}\n"
        "Expected files:\n"
        f"{tree}\n"
        "Found files:\n"
        f"{found}\n"
        "Missing files:\n"
        f"{missing}\n"
        "Expected folder structure example:\n"
        "data/\n"
        "  - change_ranking_input.xlsx\n"
        "  - Flight Pairs.xlsx\n"
        "  - O&D Rakip Bağlantı Tablosu.xlsx\n"
        "  - Yolcu Verisi_masked.xlsx\n"
        "  - .gitkeep"
    )


# Dosya adi birebir bulunamazsa asagidaki anahtar kelimelerle esnek eslestirme yapilir
# (gizli test setinde dosya adlari farkli olabilir, orn. "W26 Flight Pair 20260227 (1).xlsx").
_FILE_PATTERNS: dict[str, list[str]] = {
    "competitor_connections_file": ["rakip", "baglanti", "bağlantı", "o&d", "od rakip"],
    "passenger_data_file": ["yolcu", "passenger"],
    "ranking_input_file": ["ranking", "rank"],
    "flight_pairs_file": ["pair"],
}


def _resolve_input_file(data_dir: Path, configured_name: str, keywords: list[str]) -> str | None:
    """Return the configured file name, or a unique fuzzy match on keywords."""

    if (data_dir / configured_name).exists():
        return configured_name
    if not data_dir.exists():
        return None
    candidates = [p.name for p in data_dir.iterdir() if p.suffix.lower() in {".xlsx", ".xlsm", ".xls"}]
    for keyword in keywords:
        matches = [name for name in candidates if keyword.casefold() in name.casefold()]
        if len(matches) == 1:
            print(f"[INFO] Input file '{configured_name}' not found; using fuzzy match '{matches[0]}'.")
            return matches[0]
    return None


def _validate_input_files(config: RunConfig) -> None:
    data_dir = Path(config.data_dir)
    for attr, keywords in _FILE_PATTERNS.items():
        resolved = _resolve_input_file(data_dir, getattr(config, attr), keywords)
        if resolved is not None:
            setattr(config, attr, resolved)
    expected = [
        config.ranking_input_file,
        config.flight_pairs_file,
        config.competitor_connections_file,
        config.passenger_data_file,
    ]
    found = sorted([path.name for path in data_dir.iterdir()]) if data_dir.exists() else []
    missing = [name for name in expected if not (data_dir / name).exists()]
    if missing:
        raise DataLoadError(_missing_input_message(data_dir, expected, found, missing))


def _excel_serial_to_datetime(value: float) -> datetime:
    return datetime(1899, 12, 30) + timedelta(days=float(value))


def parse_datetime(value) -> pd.Timestamp:
    """Parse Excel/Python/string datetimes robustly."""

    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return pd.Timestamp(_excel_serial_to_datetime(float(value)))
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise DataLoadError(f"Could not parse datetime value: {value!r}")
    return parsed


def parse_duration_minutes(value) -> float:
    """Convert Excel/Python/string duration values to minutes."""

    if pd.isna(value):
        return np.nan
    if isinstance(value, time):
        return float(value.hour * 60 + value.minute + value.second / 60)
    if isinstance(value, timedelta):
        return value.total_seconds() / 60
    if isinstance(value, pd.Timedelta):
        return value.total_seconds() / 60
    if isinstance(value, datetime):
        return float(value.hour * 60 + value.minute + value.second / 60)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if 0 <= numeric < 2:
            return numeric * 24 * 60
        return numeric
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) >= 2:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(float(parts[2])) if len(parts) > 2 else 0
        return float(hours * 60 + minutes + seconds / 60)
    parsed = pd.to_timedelta(text, errors="coerce")
    if pd.isna(parsed):
        raise DataLoadError(f"Could not parse duration value: {value!r}")
    return parsed.total_seconds() / 60


def parse_excel_duration_detail(value) -> dict:
    """Parse Excel duration-like values and keep method/warning metadata."""

    if pd.isna(value):
        return {
            "original_value": value,
            "parsed_minutes": np.nan,
            "parse_method": "missing",
            "parse_warning": "missing duration value",
        }
    warning = ""
    if isinstance(value, pd.Timestamp):
        minutes = float(value.hour * 60 + value.minute + value.second / 60)
        warning = "timestamp/time-of-day value may be wrapped if original duration exceeded 24h"
        return {"original_value": value, "parsed_minutes": minutes, "parse_method": "timestamp_time_of_day", "parse_warning": warning}
    if isinstance(value, time):
        minutes = float(value.hour * 60 + value.minute + value.second / 60)
        warning = "time-of-day value may be wrapped if original duration exceeded 24h"
        return {"original_value": value, "parsed_minutes": minutes, "parse_method": "time_of_day", "parse_warning": warning}
    if isinstance(value, datetime):
        minutes = float(value.hour * 60 + value.minute + value.second / 60)
        warning = "datetime time-of-day value may be wrapped if original duration exceeded 24h"
        return {"original_value": value, "parsed_minutes": minutes, "parse_method": "datetime_time_of_day", "parse_warning": warning}
    if isinstance(value, timedelta):
        return {"original_value": value, "parsed_minutes": value.total_seconds() / 60, "parse_method": "timedelta", "parse_warning": ""}
    if isinstance(value, pd.Timedelta):
        return {"original_value": value, "parsed_minutes": value.total_seconds() / 60, "parse_method": "pandas_timedelta", "parse_warning": ""}
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if 0 <= numeric < 2:
            warning = "numeric Excel time fraction; raw value preserves total only if cell value is not display-wrapped"
            minutes = numeric * 24 * 60
            method = "excel_day_fraction"
        else:
            minutes = numeric
            method = "numeric_minutes"
        return {"original_value": value, "parsed_minutes": minutes, "parse_method": method, "parse_warning": warning}
    text = str(value).strip()
    if not text:
        return {"original_value": value, "parsed_minutes": np.nan, "parse_method": "blank", "parse_warning": "blank duration value"}
    parsed_datetime = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if not pd.isna(parsed_datetime):
        minutes = float(parsed_datetime.hour * 60 + parsed_datetime.minute + parsed_datetime.second / 60)
        return {
            "original_value": value,
            "parsed_minutes": minutes,
            "parse_method": "datetime_string_time_of_day",
            "parse_warning": "date/time string parsed as time-of-day duration; value may be wrapped if original duration exceeded 24h",
        }
    parsed_delta = pd.to_timedelta(text, errors="coerce")
    if not pd.isna(parsed_delta):
        return {"original_value": value, "parsed_minutes": parsed_delta.total_seconds() / 60, "parse_method": "timedelta_string", "parse_warning": ""}
    parts = text.split(":")
    if len(parts) >= 2:
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(float(parts[2])) if len(parts) > 2 else 0
            return {
                "original_value": value,
                "parsed_minutes": float(hours * 60 + minutes + seconds / 60),
                "parse_method": "hh_mm_ss_string",
                "parse_warning": "",
            }
        except ValueError:
            pass
    return {
        "original_value": value,
        "parsed_minutes": np.nan,
        "parse_method": "parse_failed",
        "parse_warning": f"could not parse duration value: {value!r}",
    }


def parse_excel_duration_to_minutes(value) -> float:
    """Convert Excel duration-like values to minutes with wrap-aware metadata available via detail helper."""

    return float(parse_excel_duration_detail(value)["parsed_minutes"])


def minutes_of_week(ts: pd.Timestamp) -> int:
    """Return minute index in week using Monday as day 1."""

    if pd.isna(ts):
        return -1
    day = int(ts.weekday()) + 1
    return day * 1440 + int(ts.hour) * 60 + int(ts.minute)


def load_inputs(config: RunConfig) -> RawData:
    """Load and normalize all input workbooks."""

    print("[INFO] Loading input files...")
    data_dir = Path(config.data_dir)
    _validate_input_files(config)
    connections = _read_excel_first_sheet(data_dir / config.competitor_connections_file, REQUIRED_CONNECTION_COLUMNS)
    passengers = _read_excel_first_sheet(data_dir / config.passenger_data_file, REQUIRED_PASSENGER_COLUMNS)
    ranking = _read_excel_first_sheet(data_dir / config.ranking_input_file, REQUIRED_RANKING_COLUMNS)
    flight_pairs = _read_excel_first_sheet(data_dir / config.flight_pairs_file, REQUIRED_FLIGHT_PAIR_COLUMNS)

    connections = normalize_connections(connections)
    passengers = normalize_passengers(passengers)
    ranking = normalize_ranking(ranking)
    flight_pairs = normalize_flight_pairs(flight_pairs)

    print(f"[INFO] Loaded {len(connections)} OD connection rows.")
    print(f"[INFO] Loaded {len(passengers)} passenger OD rows.")
    print(f"[INFO] Loaded {len(ranking)} ranking rows.")
    print(f"[INFO] Loaded {len(flight_pairs)} flight-pair rows.")
    return RawData(connections, passengers, ranking, flight_pairs)


def normalize_connections(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize competitor/TK connection rows."""

    out = df.copy()
    out = out.dropna(how="all").reset_index(drop=True)
    out["connection_id"] = [f"conn_{i:06d}" for i in range(len(out))]
    out["Cr1"] = out["Cr1"].astype(str).str.strip()
    out["Cr2"] = out["Cr2"].astype(str).str.strip()
    out["Carrier Name"] = out["Carrier Name"].astype(str).str.strip()
    out["Dep1"] = out["Dep1"].astype(str).str.strip()
    out["Arr1"] = out["Arr1"].astype(str).str.strip()
    out["Dep2"] = out["Dep2"].astype(str).str.strip()
    out["Arr2"] = out["Arr2"].astype(str).str.strip()
    out["O&D"] = out["O&D"].astype(str).str.strip()
    od_parts = out["O&D"].str.split("-", n=1, expand=True)
    out["origin"] = od_parts[0].str.strip()
    out["destination"] = od_parts[1].str.strip()
    out["arr_time"] = out["Arr Time"].map(parse_datetime)
    out["dep_time"] = out["Dep Time"].map(parse_datetime)
    out["arr_min"] = out["arr_time"].map(minutes_of_week)
    out["dep_min"] = out["dep_time"].map(minutes_of_week)
    out["day"] = pd.to_numeric(out["Gün"], errors="coerce").fillna(out["dep_time"].dt.weekday + 1).astype(int)
    out["is_tk"] = (out["Cr1"].str.upper() == "TK") | (out["Cr2"].str.upper() == "TK")
    out["carrier_code"] = out["Cr1"]
    out["competitor_id"] = out["carrier_code"] + "_" + out["connection_id"]
    out["inbound_flight"] = out["FlNo1"].astype(str).str.strip()
    out["outbound_flight"] = out["FlNo2"].astype(str).str.strip()
    # Hafta sarmasi duzeltmesi: dep_min/arr_min haftanin dakikasi oldugundan
    # Pazar gecesi varip Pazartesi devam eden baglantilarda ham fark ~ -10080+gap cikar.
    # Derin negatif farklar (yarim haftadan buyuk) bir hafta ileri sarilir; kucuk
    # negatifler (ayni gun icinde outbound'un once kalktigi gercek durumlar) korunur.
    raw_gap = out["dep_min"] - out["arr_min"]
    week_minutes = 7 * 1440
    wrapped = raw_gap < -(week_minutes / 2)
    out["connection_gap_min"] = raw_gap.where(~wrapped, raw_gap + week_minutes)
    out["connection_gap_week_wrapped"] = wrapped
    if int(wrapped.sum()):
        print(f"[INFO] Week-wrap corrected connection gaps: {int(wrapped.sum())} rows.")
    elapsed1 = out["ElapsedTime1"].map(parse_excel_duration_detail) if "ElapsedTime1" in out.columns else None
    elapsed2 = out["ElapsedTime2"].map(parse_excel_duration_detail) if "ElapsedTime2" in out.columns else None
    gate = out["Gate-to-Gate Uçuş Süresi"].map(parse_excel_duration_detail)
    out["gate_to_gate_journey_time_min"] = gate.map(lambda item: item["parsed_minutes"])
    out["gate_to_gate_parse_method"] = gate.map(lambda item: item["parse_method"])
    out["gate_to_gate_parse_warning"] = gate.map(lambda item: item["parse_warning"])
    if elapsed1 is not None:
        out["elapsed_time_1_min"] = elapsed1.map(lambda item: item["parsed_minutes"])
        out["elapsed_time_1_parse_method"] = elapsed1.map(lambda item: item["parse_method"])
        out["elapsed_time_1_parse_warning"] = elapsed1.map(lambda item: item["parse_warning"])
    else:
        out["elapsed_time_1_min"] = np.nan
        out["elapsed_time_1_parse_method"] = ""
        out["elapsed_time_1_parse_warning"] = "ElapsedTime1 column missing"
    if elapsed2 is not None:
        out["elapsed_time_2_min"] = elapsed2.map(lambda item: item["parsed_minutes"])
        out["elapsed_time_2_parse_method"] = elapsed2.map(lambda item: item["parse_method"])
        out["elapsed_time_2_parse_warning"] = elapsed2.map(lambda item: item["parse_warning"])
    else:
        out["elapsed_time_2_min"] = np.nan
        out["elapsed_time_2_parse_method"] = ""
        out["elapsed_time_2_parse_warning"] = "ElapsedTime2 column missing"
    both_elapsed = out["elapsed_time_1_min"].notna() & out["elapsed_time_2_min"].notna()
    computed = out["elapsed_time_1_min"] + out["connection_gap_min"] + out["elapsed_time_2_min"]
    out["computed_journey_time_from_elapsed_min"] = computed
    out["journey_time_min"] = out["gate_to_gate_journey_time_min"]
    out["journey_time_source"] = "gate_to_gate"
    valid_computed = both_elapsed & computed.notna() & (computed > 0)
    out.loc[valid_computed, "journey_time_min"] = computed.loc[valid_computed]
    out.loc[valid_computed, "journey_time_source"] = "elapsedtime_sum"
    return out


def normalize_passengers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize O-D importance rows and combine duplicates by sum."""

    out = df.copy().dropna(how="all")
    out["origin"] = out["Orig Airport Code"].astype(str).str.strip()
    out["destination"] = out["Dest Airport Code"].astype(str).str.strip()
    out["od"] = out["origin"] + "-" + out["destination"]
    out["importance_factor"] = pd.to_numeric(out["OD importance factor"], errors="coerce").fillna(0.0)
    return out.groupby(["od", "origin", "destination"], as_index=False)["importance_factor"].sum()


def normalize_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ranking reward lookup."""

    out = df.copy().dropna(how="all")
    out["competitor_count"] = pd.to_numeric(out["Rakip Sayısı"], errors="raise").astype(int)
    out["initial_rank"] = pd.to_numeric(out["İlk Rank"], errors="raise").astype(int)
    out["new_rank"] = pd.to_numeric(out["Son Rank"], errors="raise").astype(int)
    out["weight"] = pd.to_numeric(out["Weight"], errors="coerce").fillna(0.0)
    return out[["competitor_count", "initial_rank", "new_rank", "weight"]]


def normalize_flight_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize flight-pair records."""

    out = df.copy().dropna(how="all")
    out["flight_no"] = out["FlNo"].astype(str).str.strip().str.zfill(3)
    out["origin"] = out["Orig"].astype(str).str.strip()
    out["destination"] = out["Dest"].astype(str).str.strip()
    out["pair"] = out["Pair"].astype(str).str.strip()
    return out[["flight_no", "origin", "destination", "pair"]]
