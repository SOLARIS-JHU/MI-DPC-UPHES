from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def load_ablation_runs_csv(path: str | Path) -> list[dict[str, str]]:
    """Load the retained-study ablation run table as a list of row dicts."""
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return [
            {key: value.strip() if isinstance(value, str) else value for key, value in row.items()}
            for row in reader
        ]


def filter_runs(
    rows: Iterable[Mapping[str, str]],
    *,
    study: str,
    variant: str,
) -> list[dict[str, str]]:
    """Return rows matching the requested study and variant."""
    return [
        dict(row)
        for row in rows
        if row.get("study") == study and row.get("variant") == variant
    ]


def select_median_seed(
    rows: Iterable[Mapping[str, str]],
    *,
    study: str,
    variant: str,
) -> int:
    """Return the seed at the median ex-post profit for a study/variant slice."""
    matches = filter_runs(rows, study=study, variant=variant)
    if not matches:
        raise ValueError(f"No rows found for study={study!r}, variant={variant!r}")

    ranked = sorted(
        matches,
        key=lambda row: (_read_float(row, "mean_expost_profit"), _read_int(row, "seed")),
    )
    return _read_int(ranked[len(ranked) // 2], "seed")


def pick_representative_day(per_day: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]]) -> str:
    """Pick a representative day from retained-study data or benchmark CSV rows.

    Accepted input schemas:
    - retained ``per_day`` JSON mapping: ``{"YYYY/MM/DD": {...}}``
    - MIQP benchmark CSV rows: iterable of row dicts with date, volume-penalty,
      and ex-post-profit columns

    The heuristic prefers zero volume penalty and then higher ex-post profit.
    """
    rows = list(_as_day_rows(per_day))
    if not rows:
        raise ValueError("No day rows supplied")

    zero_penalty = [row for row in rows if abs(_read_float(row, "vol", "pen")) <= 1e-9]
    if zero_penalty:
        candidates = zero_penalty
        key = lambda row: (-_read_float(row, "expost", "profit"), _read_date(row))
    else:
        min_penalty = min(_read_float(row, "vol", "pen") for row in rows)
        candidates = [row for row in rows if abs(_read_float(row, "vol", "pen") - min_penalty) <= 1e-9]
        key = lambda row: (-_read_float(row, "expost", "profit"), _read_date(row))
    best = min(
        candidates,
        key=key,
    )
    return _read_date(best)


def _as_day_rows(
    per_day: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
) -> Iterable[dict[str, Any]]:
    if isinstance(per_day, Mapping):
        for date, row in per_day.items():
            if not isinstance(row, Mapping):
                continue
            normalized = dict(row)
            normalized.setdefault("date", date)
            yield normalized
        return

    for row in per_day:
        yield dict(row)


def _read_float(row: Mapping[str, Any], *needles: str) -> float:
    value = _read_value(row, *needles)
    if value is None:
        raise KeyError(f"Missing numeric field matching {needles!r}")
    return float(value)


def _read_int(row: Mapping[str, Any], *needles: str) -> int:
    return int(float(_read_float(row, *needles)))


def _read_date(row: Mapping[str, Any]) -> str:
    for key in row:
        if key.lower() == "date":
            value = row[key]
            if value is None:
                raise KeyError("Date field is empty")
            return str(value).strip()
    raise KeyError("No date field found")


def _read_value(row: Mapping[str, Any], *needles: str) -> Any:
    key = _find_key(row, *needles)
    if key is None:
        return None
    value = row[key]
    if isinstance(value, str):
        value = value.strip()
    return value


def _find_key(row: Mapping[str, Any], *needles: str) -> str | None:
    normalized_needles = [_normalize_text(needle) for needle in needles]
    for key in row:
        normalized = _normalize_text(key)
        if all(needle in normalized for needle in normalized_needles):
            return key
    return None


def _normalize_text(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())
