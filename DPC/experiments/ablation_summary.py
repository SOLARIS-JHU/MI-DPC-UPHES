"""Summaries for dynamics ablation run histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_EPS = 1e-12
DEFAULT_NEAR_ZERO_THRESHOLD = 1e-3
DEFAULT_TAIL_WINDOW = 5


def _as_history_dict(history) -> dict:
    if isinstance(history, dict):
        return history
    if isinstance(history, list):
        if not history:
            return {}
        keys = set().union(*(row.keys() for row in history))
        return {key: [row.get(key) for row in history] for key in keys}
    raise TypeError(f"Unsupported history container: {type(history)!r}")


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _finite_values(values) -> np.ndarray:
    values = _as_float_array(values).reshape(-1)
    return values[np.isfinite(values)]


def _history_length(history: dict) -> int:
    lengths = []
    for value in history.values():
        if isinstance(value, (str, bytes)):
            continue
        try:
            lengths.append(len(value))
        except TypeError:
            continue
    return max(lengths, default=0)


def _mean_and_std(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def _tail_slope(x: np.ndarray, y: np.ndarray, tail_window: int) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x_tail = x[mask]
    y_tail = y[mask]
    window = max(2, min(int(tail_window), x_tail.size, y_tail.size))
    x_tail = x_tail[-window:]
    y_tail = y_tail[-window:]
    if x_tail.size < 2 or np.allclose(x_tail, x_tail[0]):
        return float("nan")
    try:
        return float(np.polyfit(x_tail, y_tail, 1)[0])
    except np.linalg.LinAlgError:
        return float("nan")


def summarize_history(
    history,
    *,
    eps: float = DEFAULT_EPS,
    near_zero_threshold: float = DEFAULT_NEAR_ZERO_THRESHOLD,
    tail_window: int = DEFAULT_TAIL_WINDOW,
) -> dict:
    """Summarize a training history into gradient-quality proxies.

    Parameters
    ----------
    history:
        A dict of metric arrays or a list of row dicts.
    eps:
        Numerical guard used for near-zero thresholds and division safety.
    near_zero_threshold:
        Threshold used to count a gradient norm as near-zero.
    tail_window:
        Number of trailing epochs to use for the dev-expost slope estimate.
    """

    history = _as_history_dict(history)
    grad_norm = _as_float_array(history.get("grad_norm", []))
    dev_expost = _as_float_array(history.get("dev_expost", []))
    history_len = _history_length(history) or max(len(grad_norm), len(dev_expost))
    epoch_values = history.get("epoch")
    if epoch_values is None:
        epoch = np.arange(1, history_len + 1, dtype=float)
    else:
        epoch = _as_float_array(epoch_values)

    grad_mean, grad_std = _mean_and_std(grad_norm)
    denom = max(abs(grad_mean), eps)
    finite_grad = _finite_values(grad_norm)
    if finite_grad.size:
        near_zero_grad_frac = float(np.mean(np.abs(finite_grad) <= near_zero_threshold))
    else:
        near_zero_grad_frac = float("nan")

    if finite_grad.size >= 2:
        split = max(1, finite_grad.size // 2)
        early = finite_grad[:split]
        late = finite_grad[split:]
        early_mean = float(np.mean(early)) if early.size else grad_mean
        late_mean = float(np.mean(late)) if late.size else grad_mean
        late_to_early_grad_ratio = float(late_mean / max(abs(early_mean), eps))
    elif finite_grad.size == 1:
        late_to_early_grad_ratio = 1.0
    else:
        late_to_early_grad_ratio = float("nan")

    finite_dev = _finite_values(dev_expost)
    if finite_dev.size:
        finite_mask = np.isfinite(dev_expost)
        finite_indices = np.flatnonzero(finite_mask)
        best_local_idx = int(np.argmax(dev_expost[finite_mask]))
        best_idx = int(finite_indices[best_local_idx])
        if best_idx < epoch.size and np.isfinite(epoch[best_idx]):
            best_dev_epoch = int(epoch[best_idx])
        else:
            best_dev_epoch = best_idx + 1
    else:
        best_dev_epoch = -1

    summary = {
        "grad_norm_mean": grad_mean,
        "grad_norm_std": grad_std,
        "grad_norm_cv": float(grad_std / denom) if finite_grad.size else float("nan"),
        "near_zero_grad_frac": near_zero_grad_frac,
        "late_to_early_grad_ratio": late_to_early_grad_ratio,
        "best_dev_epoch": best_dev_epoch,
        "dev_expost_slope_tail": _tail_slope(epoch, dev_expost, tail_window),
    }
    return summary


def summarize_eval_results(
    path,
    *,
    eps: float = DEFAULT_EPS,
    near_zero_threshold: float = DEFAULT_NEAR_ZERO_THRESHOLD,
    tail_window: int = DEFAULT_TAIL_WINDOW,
) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "eval_results.json"
    with path.open() as f:
        data = json.load(f)
    return summarize_history(
        data.get("history", {}),
        eps=eps,
        near_zero_threshold=near_zero_threshold,
        tail_window=tail_window,
    )


def summarize_eval_results_many(
    paths: Iterable[str | Path],
    *,
    eps: float = DEFAULT_EPS,
    near_zero_threshold: float = DEFAULT_NEAR_ZERO_THRESHOLD,
    tail_window: int = DEFAULT_TAIL_WINDOW,
) -> list[dict]:
    return [
        summarize_eval_results(path, eps=eps, near_zero_threshold=near_zero_threshold, tail_window=tail_window)
        for path in paths
    ]


def aggregate_summaries(summaries: Iterable[dict]) -> dict:
    summaries = list(summaries)
    if not summaries:
        return {"count": 0}

    aggregated = {"count": len(summaries)}
    for key in summaries[0]:
        try:
            values = np.asarray([row[key] for row in summaries], dtype=float)
        except (TypeError, ValueError, KeyError):
            continue
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            aggregated[f"{key}_mean"] = float("nan")
            aggregated[f"{key}_std"] = float("nan")
            continue
        aggregated[f"{key}_mean"] = float(np.mean(finite))
        aggregated[f"{key}_std"] = float(np.std(finite))
    return aggregated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize ablation eval_results.json histories.")
    parser.add_argument("paths", nargs="+", help="Run directories or eval_results.json files")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Epsilon used for safe denominators")
    parser.add_argument(
        "--near-zero-threshold",
        type=float,
        default=DEFAULT_NEAR_ZERO_THRESHOLD,
        help="Threshold used to count near-zero gradient norms",
    )
    parser.add_argument("--tail-window", type=int, default=DEFAULT_TAIL_WINDOW, help="Number of trailing epochs for the dev-expost slope")
    parser.add_argument("--aggregate", action="store_true", help="Print an aggregate summary over all runs")
    args = parser.parse_args(argv)

    summaries = summarize_eval_results_many(
        args.paths,
        eps=args.eps,
        near_zero_threshold=args.near_zero_threshold,
        tail_window=args.tail_window,
    )
    if args.aggregate:
        payload = aggregate_summaries(summaries)
    else:
        payload = summaries
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "aggregate_summaries",
    "main",
    "summarize_eval_results",
    "summarize_eval_results_many",
    "summarize_history",
]
