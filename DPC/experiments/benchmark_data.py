"""Daily price pool construction and sampling for benchmark tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
from neuromancer.dataset import DictDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from DPC.config import MIN_PRICE, PRICE_NOISE_STD


DEFAULT_EXTREME_DATE = "2024/12/12"


@dataclass(frozen=True)
class PricePools:
    """Price dictionaries for train, development, and fixed benchmark use."""

    benchmark_prices: Dict[str, list[float]]
    train_pool_prices: Dict[str, list[float]]
    dev_pool_prices: Dict[str, list[float]]
    metadata: dict


def _canonical_date(value) -> str:
    return pd.Timestamp(value).strftime("%Y/%m/%d")


def load_benchmark_price_days(csv_path: str) -> Dict[str, list[float]]:
    """Load the fixed 19-day benchmark CSV."""

    df = pd.read_csv(csv_path, dtype={"prices_hourly": str})
    price_data = {}
    for _, row in df.iterrows():
        prices = [float(x.strip()) for x in str(row["prices_hourly"]).split(",")]
        if len(prices) != 24:
            continue
        price_data[_canonical_date(row["date"])] = prices
    return dict(sorted(price_data.items()))


def load_raw_year_price_days(csv_path: str, year: int = 2024) -> Dict[str, list[float]]:
    """Load all complete 24-hour price days from the raw Belgium dataset."""

    df = pd.read_csv(csv_path)
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df[df["Datetime (UTC)"].dt.year == year].copy()
    df["date"] = df["Datetime (UTC)"].dt.strftime("%Y/%m/%d")

    price_data = {}
    for day, group in df.groupby("date"):
        group = group.sort_values("Datetime (UTC)")
        if len(group) < 24:
            continue
        prices = group["Price (EUR/MWhe)"].to_numpy(dtype=np.float32)[:24]
        if prices.shape[0] == 24:
            price_data[day] = prices.tolist()
    return dict(sorted(price_data.items()))


def _monthly_split(
    price_pool: Dict[str, list[float]],
    dev_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[Dict[str, list[float]], Dict[str, list[float]]]:
    """Hold out a month-stratified development subset from the training pool."""

    rng = np.random.default_rng(seed)
    train_dates = []
    dev_dates = []

    date_index = pd.Index([pd.Timestamp(d) for d in sorted(price_pool)])
    by_month = {}
    for ts in date_index:
        by_month.setdefault(ts.month, []).append(ts)

    for month_dates in by_month.values():
        month_dates = list(month_dates)
        order = rng.permutation(len(month_dates))
        month_dates = [month_dates[i] for i in order]
        if len(month_dates) <= 1:
            train_dates.extend(month_dates)
            continue
        n_dev = max(1, int(round(len(month_dates) * dev_fraction)))
        n_dev = min(n_dev, len(month_dates) - 1)
        dev_dates.extend(month_dates[:n_dev])
        train_dates.extend(month_dates[n_dev:])

    train_pool = {_canonical_date(d): price_pool[_canonical_date(d)] for d in sorted(train_dates)}
    dev_pool = {_canonical_date(d): price_pool[_canonical_date(d)] for d in sorted(dev_dates)}
    return train_pool, dev_pool


def build_price_pools(
    raw_csv_path: str,
    benchmark_csv_path: str,
    year: int = 2024,
    extreme_date: str = DEFAULT_EXTREME_DATE,
    dev_fraction: float = 0.2,
    seed: int = 0,
) -> PricePools:
    """Create separate train/dev pools while preserving the fixed benchmark set."""

    benchmark_prices = load_benchmark_price_days(benchmark_csv_path)
    raw_prices = load_raw_year_price_days(raw_csv_path, year=year)

    benchmark_dates = set(benchmark_prices)
    extreme_date = _canonical_date(extreme_date)

    filtered_pool = {
        date: prices
        for date, prices in raw_prices.items()
        if date not in benchmark_dates and date != extreme_date
    }
    train_pool, dev_pool = _monthly_split(filtered_pool, dev_fraction=dev_fraction, seed=seed)

    metadata = {
        "year": year,
        "benchmark_days": len(benchmark_prices),
        "raw_complete_days": len(raw_prices),
        "train_pool_days": len(train_pool),
        "dev_pool_days": len(dev_pool),
        "excluded_extreme_date": extreme_date,
        "benchmark_dates": sorted(benchmark_dates),
    }
    return PricePools(
        benchmark_prices=benchmark_prices,
        train_pool_prices=train_pool,
        dev_pool_prices=dev_pool,
        metadata=metadata,
    )


def _uniform_group_sample(groups: list[np.ndarray], num_samples: int, rng: np.random.Generator) -> np.ndarray:
    picks = np.empty(num_samples, dtype=np.int64)
    for i in range(num_samples):
        group = groups[i % len(groups)]
        picks[i] = group[rng.integers(0, len(group))]
    rng.shuffle(picks)
    return picks


def _cluster_assignments(matrix: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    n_clusters = max(1, min(n_clusters, len(matrix)))
    if n_clusters == 1:
        return np.zeros(len(matrix), dtype=np.int64)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(matrix)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    return kmeans.fit_predict(x_scaled)


def _volatility_assignments(matrix: np.ndarray, n_bins: int = 4) -> np.ndarray:
    volatility = matrix.std(axis=1)
    if len(volatility) <= 1:
        return np.zeros(len(volatility), dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, min(n_bins, len(volatility)) + 1)
    edges = np.quantile(volatility, quantiles)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros(len(volatility), dtype=np.int64)
    bins = np.digitize(volatility, edges[1:-1], right=True)
    return bins.astype(np.int64)


def _shape_features(matrix: np.ndarray) -> np.ndarray:
    diff = np.diff(matrix, axis=1, prepend=matrix[:, :1])
    return np.stack(
        [
            matrix.mean(axis=1),
            matrix.std(axis=1),
            matrix.min(axis=1),
            matrix.max(axis=1),
            (matrix[:, 17:24].mean(axis=1) - matrix[:, 0:6].mean(axis=1)),
            np.abs(diff).mean(axis=1),
            np.abs(diff).max(axis=1),
            (matrix < 0.0).mean(axis=1),
        ],
        axis=1,
    )


def sample_price_dataset(
    price_pool: Dict[str, list[float]],
    h_init: float,
    v_init: float,
    num_samples: int,
    sampler: str = "noisy_resampling",
    noise_std: float = PRICE_NOISE_STD,
    min_price: float = MIN_PRICE,
    seed: int = 0,
    n_clusters: int = 12,
    shape_clusters: int = 8,
    name: str = "train",
) -> tuple[DictDataset, dict]:
    """Sample a Neuromancer dataset from the non-benchmark 2024 pool."""

    if not price_pool:
        raise ValueError("price_pool must contain at least one day")

    dates = sorted(price_pool)
    matrix = np.asarray([price_pool[d] for d in dates], dtype=np.float32)
    rng = np.random.default_rng(seed)

    if sampler == "noisy_resampling":
        picks = rng.integers(0, len(dates), size=num_samples)
        group_labels = np.zeros(len(dates), dtype=np.int64)
    elif sampler == "cluster_balanced":
        group_labels = _cluster_assignments(matrix, n_clusters=n_clusters, seed=seed)
        groups = [np.flatnonzero(group_labels == label) for label in sorted(np.unique(group_labels))]
        picks = _uniform_group_sample(groups, num_samples, rng)
    elif sampler == "volatility_stratified":
        group_labels = _volatility_assignments(matrix)
        groups = [np.flatnonzero(group_labels == label) for label in sorted(np.unique(group_labels))]
        picks = _uniform_group_sample(groups, num_samples, rng)
    elif sampler == "shape_stratified":
        features = _shape_features(matrix)
        group_labels = _cluster_assignments(features, n_clusters=shape_clusters, seed=seed)
        groups = [np.flatnonzero(group_labels == label) for label in sorted(np.unique(group_labels))]
        picks = _uniform_group_sample(groups, num_samples, rng)
    else:
        raise ValueError(f"Unknown sampler '{sampler}'")

    sampled = matrix[picks].copy()
    if noise_std > 0.0:
        sampled += rng.normal(0.0, noise_std, sampled.shape).astype(np.float32)
    sampled = np.maximum(min_price, sampled)

    x = np.tile(np.array([[h_init, v_init]], dtype=np.float32), (num_samples, 1))
    data = {
        "x": torch.tensor(x, dtype=torch.float32).unsqueeze(1),
        "d": torch.tensor(sampled, dtype=torch.float32).unsqueeze(-1),
    }

    sample_dates, sample_counts = np.unique([dates[i] for i in picks], return_counts=True)
    metadata = {
        "sampler": sampler,
        "num_samples": num_samples,
        "noise_std": noise_std,
        "source_days": len(dates),
        "sampled_day_histogram": {date: int(count) for date, count in zip(sample_dates, sample_counts)},
        "group_histogram": {int(label): int((group_labels == label).sum()) for label in np.unique(group_labels)},
    }
    return DictDataset(data, name=name), metadata
