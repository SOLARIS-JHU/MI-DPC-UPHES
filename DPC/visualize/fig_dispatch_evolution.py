"""Epoch-colored dispatch evolution figure for replayed checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from DPC.config import load_system_params
from DPC.visualize.style import (
    apply_style,
    cleanup_axes,
    C_MIQP_GL,
    C_MIQP_PW,
    COL_WIDTH,
    EPOCH_CMAP,
    EPOCH_DISPATCH_ALPHA,
    EPOCH_DISPATCH_LINEWIDTH,
    FIGS_OUT,
    GRID_KW,
    MIQP_ROOT,
)

MIQP_PW_RESULTS_CSV = MIQP_ROOT / "MIQP_piecewise" / "MIQP_piecewise_results.csv"
MIQP_GL_RESULTS_CSV = MIQP_ROOT / "MIQP_linear" / "MILP_global_linear_results.csv"

OUTPUT_PDF = "epoch_dispatch_colormap.pdf"
OUTPUT_PDF_WITH_MIQP = "epoch_dispatch_colormap_with_miqp.pdf"
MIQP_PW_OVERLAY_COLOR = "#1F77B4"
PRICE_COLOR = "#4F4F4F"
HEAD_COLOR = "#D62728"
PLAIN_PRICE_COLOR = "#1F77B4"
PLAIN_POWER_COLOR = "#D62728"
PLAIN_POS_REGION_COLOR = "#F2B6B6"
PLAIN_NEG_REGION_COLOR = "#F2B6B6"
LOSS_COLOR = "#7B1FA2"
TAU_COLOR = "#FF7F0E"
LR_COLOR = "#008080"
TARGET_HEAD_COLOR = "#3A3A3A"
LOSS_DOT_CMAP = plt.get_cmap("viridis_r")
FIG_LABEL_FONTSIZE = 8
FIG_TICK_FONTSIZE = 7
FIG_LEGEND_FONTSIZE = 6.5


def _format_sci_tick(value: float, _pos: float) -> str:
    if abs(value) < 1e-15:
        return "0"
    return f"{value:.0e}"


def load_epoch_cache(cache_path: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the replay cache and return date, epochs, power traces, prices, and head traces."""
    with np.load(Path(cache_path), allow_pickle=True) as cache:
        date = str(np.asarray(cache["date"]).item())
        epochs = np.asarray(cache["epoch"], dtype=int).reshape(-1)
        power = np.asarray(cache["p_exec"], dtype=float)
        price = np.asarray(cache["price"], dtype=float).reshape(-1)
        head = np.asarray(cache["h"], dtype=float)
    if power.ndim == 1:
        power = power.reshape(1, -1)
    if head.ndim == 1:
        head = head.reshape(1, -1)
    if epochs.shape[0] != power.shape[0]:
        raise ValueError(
            "Replay cache is malformed: epoch count does not match the number of power traces."
        )
    if epochs.shape[0] != head.shape[0]:
        raise ValueError(
            "Replay cache is malformed: epoch count does not match the number of head traces."
        )
    if power.shape[1] != price.shape[0] or head.shape[1] != price.shape[0]:
        raise ValueError(
            "Replay cache is malformed: price, power, and head trajectories must share the same horizon length."
        )
    return date, epochs, power, price, head


def load_training_curve(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load epoch/loss/tau/lr history for the replay run referenced by the cache metadata."""
    with np.load(Path(cache_path), allow_pickle=True) as cache:
        meta_raw = cache["meta_json"].item()
    meta = json.loads(str(meta_raw))
    run_dir = Path(meta["run_dir"])
    history_path = run_dir / "history.csv"
    if history_path.exists():
        epochs: list[int] = []
        losses: list[float] = []
        taus: list[float] = []
        lrs: list[float] = []
        with history_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    epochs.append(int(row["epoch"]))
                    losses.append(float(row["loss"]))
                    taus.append(float(row["tau"]))
                    lrs.append(float(row["lr"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Malformed history.csv in {history_path!s}") from exc
        if not epochs:
            raise ValueError(f"No epoch rows found in {history_path!s}")
        return (
            np.asarray(epochs, dtype=int),
            np.asarray(losses, dtype=float),
            np.asarray(taus, dtype=float),
            np.asarray(lrs, dtype=float),
        )

    eval_results_path = run_dir / "eval_results.json"
    if eval_results_path.exists():
        payload = json.loads(eval_results_path.read_text(encoding="utf-8"))
        history = payload.get("history") or {}
        epochs = history.get("epoch")
        losses = history.get("loss")
        taus = history.get("tau")
        lrs = history.get("lr")
        if epochs is not None and losses is not None and taus is not None and lrs is not None:
            return (
                np.asarray(epochs, dtype=int),
                np.asarray(losses, dtype=float),
                np.asarray(taus, dtype=float),
                np.asarray(lrs, dtype=float),
            )

    raise FileNotFoundError(
        f"Could not find history.csv or eval_results.json history for replay cache run directory {run_dir!s}."
    )


def _load_cache_meta(cache_path: Path) -> dict[str, object]:
    with np.load(Path(cache_path), allow_pickle=True) as cache:
        meta_raw = cache["meta_json"].item()
    return json.loads(str(meta_raw))


def load_feasible_power_bounds(cache_path: Path, head: np.ndarray) -> dict[str, np.ndarray]:
    meta = _load_cache_meta(cache_path)
    system_params = load_system_params(
        str(meta.get("pkl_path", "preprocess.pkl")),
        device=torch.device("cpu"),
        physics_mode="nonlinear",
        inverse_pkl_path=meta.get("inverse_pkl"),
    )
    final_head = np.asarray(head, dtype=float)
    if final_head.ndim == 2:
        final_head = final_head[-1]
    h_t = torch.as_tensor(final_head, dtype=torch.float32)
    return {
        "pos_min": system_params["pos_min"](h_t).detach().cpu().numpy(),
        "pos_max": system_params["pos_max"](h_t).detach().cpu().numpy(),
        "neg_min": system_params["neg_min"](h_t).detach().cpu().numpy(),
        "neg_max": system_params["neg_max"](h_t).detach().cpu().numpy(),
    }


def load_target_head(cache_path: Path) -> float:
    meta = _load_cache_meta(cache_path)
    system_params = load_system_params(
        str(meta.get("pkl_path", "preprocess.pkl")),
        device=torch.device("cpu"),
        physics_mode="nonlinear",
        inverse_pkl_path=meta.get("inverse_pkl"),
    )
    return float(system_params["target_head"])


def load_hourly_power_schedule(csv_path: Path, date: str) -> np.ndarray:
    """Load the hourly power schedule for one date from a result CSV."""
    rows: list[dict[str, str]] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["date"].strip() == date:
                rows.append(row)
    if not rows:
        raise ValueError(f"No hourly rows found for date {date!r} in {csv_path!s}")

    hours: list[int] = []
    for row in rows:
        try:
            hours.append(int(row["hour"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid hourly row for date {date!r} in {csv_path!s}") from exc

    if len(rows) != 24 or set(hours) != set(range(24)):
        raise ValueError(
            f"Hourly schedule for date {date!r} in {csv_path!s} must contain exactly 24 unique hours 0..23."
        )

    rows.sort(key=lambda row: int(row["hour"]))
    return np.asarray([float(row["power"]) for row in rows], dtype=float)


def _plot_epoch_dispatch(
    ax: plt.Axes,
    *,
    epochs: np.ndarray,
    series: np.ndarray,
    ylabel: str,
    title: str,
    show_xlabel: bool = True,
    cax: plt.Axes | None = None,
    miqp_pw: np.ndarray | None = None,
    miqp_gl: np.ndarray | None = None,
    epoch_cmap=None,
    final_label_loc: str = "upper center",
    target_level: float | None = None,
    nonfinal_alpha: float = EPOCH_DISPATCH_ALPHA,
) -> None:
    hours = np.arange(series.shape[1], dtype=float)
    epochs = np.asarray(epochs, dtype=float)
    if epochs.size == 0:
        raise ValueError("Replay cache did not contain any epochs.")

    cmap = epoch_cmap or EPOCH_CMAP
    norm = Normalize(vmin=float(epochs.min()), vmax=float(epochs.max()) if epochs.size > 1 else float(epochs.min()) + 1.0)
    final_epoch = float(epochs.max())
    for epoch, curve in zip(epochs, series, strict=True):
        is_final_epoch = float(epoch) == final_epoch
        ax.plot(
            hours,
            curve,
            color="black" if is_final_epoch else cmap(norm(epoch)),
            alpha=0.95 if is_final_epoch else nonfinal_alpha,
            linewidth=EPOCH_DISPATCH_LINEWIDTH + 0.2 if is_final_epoch else EPOCH_DISPATCH_LINEWIDTH,
            zorder=4 if is_final_epoch else 2,
        )

    if miqp_pw is not None:
        ax.plot(hours, miqp_pw, color=MIQP_PW_OVERLAY_COLOR, linewidth=1.6, label="MIQP-PW", zorder=6)
    if miqp_gl is not None:
        ax.plot(hours, miqp_gl, color=C_MIQP_GL, linewidth=1.6, label="MIQP-GL", zorder=6)
    if target_level is not None:
        ax.axhline(
            target_level,
            color=TARGET_HEAD_COLOR,
            linewidth=1.15,
            linestyle=(0, (4, 2)),
            alpha=0.95,
            zorder=5,
        )

    cleanup_axes(ax)
    ax.xaxis.grid(True, **GRID_KW)
    ax.set_xlabel("Hour" if show_xlabel else "", fontsize=FIG_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FIG_LABEL_FONTSIZE)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 4))
    ax.tick_params(axis="both", labelsize=FIG_TICK_FONTSIZE)
    if not show_xlabel:
        ax.tick_params(axis="x", which="both", labelbottom=False, bottom=False)
    ax.set_title(title, fontsize=FIG_LABEL_FONTSIZE)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, cax=cax, ax=ax)
    cbar.set_label("Epoch index", fontsize=FIG_LABEL_FONTSIZE)
    cbar.set_ticks(_epoch_xticks(epochs))
    cbar.ax.tick_params(labelsize=FIG_TICK_FONTSIZE)

    if miqp_pw is not None or miqp_gl is not None:
        ax.legend(frameon=False, loc="upper center", fontsize=FIG_LEGEND_FONTSIZE)
    else:
        final_handle = Line2D([0], [0], color="black", linewidth=EPOCH_DISPATCH_LINEWIDTH + 0.2)
        handles = [final_handle]
        labels = ["Final epoch"]
        if target_level is not None:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=TARGET_HEAD_COLOR,
                    linewidth=1.15,
                    linestyle=(0, (4, 2)),
                    alpha=0.95,
                )
            )
            labels.append("Target head")
        legend = ax.legend(
            handles,
            labels,
            frameon=False,
            loc=final_label_loc,
            fontsize=FIG_LEGEND_FONTSIZE,
            handlelength=1.8,
        )
        for text in legend.get_texts():
            text.set_color("black" if text.get_text() == "Final epoch" else TARGET_HEAD_COLOR)


def _plot_price_and_head(
    ax_price: plt.Axes,
    *,
    price: np.ndarray,
    aux: np.ndarray,
    aux_label: str,
    price_color: str,
    aux_color: str,
    aux_as_step: bool = False,
) -> plt.Axes:
    hours = np.arange(price.shape[0], dtype=float)
    ax_aux = ax_price.twinx()

    ax_price.step(hours, price, where="mid", color=price_color, linewidth=0.9)
    if aux_as_step:
        ax_aux.step(hours, aux, where="mid", color=aux_color, linewidth=1.1)
    else:
        ax_aux.plot(hours, aux, color=aux_color, linewidth=1.1)

    cleanup_axes(ax_price)
    ax_price.xaxis.grid(True, **GRID_KW)
    ax_price.spines["bottom"].set_visible(True)
    ax_aux.spines["top"].set_visible(False)

    ax_price.set_ylabel("Price (€/MWh)", color=price_color, fontsize=FIG_LABEL_FONTSIZE)
    ax_aux.set_ylabel(aux_label, color=aux_color, fontsize=FIG_LABEL_FONTSIZE)
    ax_price.tick_params(axis="both", colors=price_color, labelsize=FIG_TICK_FONTSIZE)
    ax_aux.tick_params(axis="y", colors=aux_color, labelsize=FIG_TICK_FONTSIZE)
    ax_price.set_xlabel("", fontsize=FIG_LABEL_FONTSIZE)
    ax_price.set_xlim(0, 23)
    ax_price.set_xticks(range(0, 24, 4))
    return ax_aux


def _plot_plain_price_and_power(
    ax_power: plt.Axes,
    *,
    price: np.ndarray,
    power: np.ndarray,
    feasible_power_bounds: dict[str, np.ndarray] | None,
) -> plt.Axes:
    hours = np.arange(price.shape[0], dtype=float)
    ax_price = ax_power.twinx()
    ax_price.set_zorder(ax_power.get_zorder() - 1)
    ax_power.patch.set_alpha(0.0)

    if feasible_power_bounds is not None:
        pos_min = np.asarray(feasible_power_bounds["pos_min"], dtype=float)
        pos_max = np.asarray(feasible_power_bounds["pos_max"], dtype=float)
        neg_min = np.asarray(feasible_power_bounds["neg_min"], dtype=float)
        neg_max = np.asarray(feasible_power_bounds["neg_max"], dtype=float)
        ax_power.fill_between(hours, pos_min, pos_max, step="mid", color=PLAIN_POS_REGION_COLOR, alpha=0.16, zorder=0)
        ax_power.fill_between(hours, neg_min, neg_max, step="mid", color=PLAIN_NEG_REGION_COLOR, alpha=0.16, zorder=0)
        for bound in [pos_min, pos_max, neg_min, neg_max]:
            ax_power.step(hours, bound, where="mid", color=PLAIN_POWER_COLOR, linewidth=0.6, alpha=0.4, zorder=1)

    ax_power.step(hours, power, where="mid", color=PLAIN_POWER_COLOR, linewidth=1.1, zorder=3)
    ax_power.axhline(0.0, color=PLAIN_POWER_COLOR, linewidth=0.6, alpha=0.4, zorder=1)
    ax_price.step(hours, price, where="mid", color=PLAIN_PRICE_COLOR, linewidth=0.9, zorder=0.5)

    cleanup_axes(ax_power)
    ax_power.xaxis.grid(True, **GRID_KW)
    ax_power.spines["bottom"].set_visible(True)
    ax_price.spines["top"].set_visible(False)

    ax_power.set_ylabel("Power (MW)", color=PLAIN_POWER_COLOR, fontsize=FIG_LABEL_FONTSIZE)
    ax_price.set_ylabel("Price (€/MWh)", color=PLAIN_PRICE_COLOR, fontsize=FIG_LABEL_FONTSIZE)
    ax_power.tick_params(axis="both", colors=PLAIN_POWER_COLOR, labelsize=FIG_TICK_FONTSIZE)
    ax_price.tick_params(axis="y", colors=PLAIN_PRICE_COLOR, labelsize=FIG_TICK_FONTSIZE)
    ax_power.set_xlabel("", fontsize=FIG_LABEL_FONTSIZE)
    ax_power.set_xlim(0, 23)
    ax_power.set_xticks(range(0, 24, 4))
    return ax_price


def _plot_training_curve(
    ax: plt.Axes,
    *,
    history_epochs: np.ndarray,
    losses: np.ndarray,
    tau: np.ndarray,
    lr: np.ndarray,
    include_lr: bool = True,
) -> tuple[plt.Axes, plt.Axes | None]:
    history_epochs = np.asarray(history_epochs, dtype=float)
    losses = np.asarray(losses, dtype=float)
    tau = np.asarray(tau, dtype=float)
    lr = np.asarray(lr, dtype=float)
    valid = np.isfinite(losses)
    history_epochs = history_epochs[valid]
    losses = losses[valid]
    tau = tau[valid]
    lr = lr[valid]
    if history_epochs.size == 0:
        raise ValueError("Training curve did not contain any finite loss values.")
    norm = Normalize(
        vmin=float(history_epochs.min()),
        vmax=float(history_epochs.max()) if history_epochs.size > 1 else float(history_epochs.min()) + 1.0,
    )
    ax.plot(history_epochs, losses, color=LOSS_COLOR, linewidth=0.9, alpha=0.85, zorder=1)
    ax.scatter(history_epochs, losses, c=history_epochs, cmap=LOSS_DOT_CMAP, norm=norm, s=7, zorder=2)
    final_epoch = float(history_epochs.max())
    final_value = float(losses[np.argmax(history_epochs)])
    ax.scatter([final_epoch], [final_value], color="black", s=18, zorder=3)

    cleanup_axes(ax)
    ax.set_xlabel("Epoch", fontsize=FIG_LABEL_FONTSIZE)
    ax.set_ylabel("Loss", color=LOSS_COLOR, fontsize=FIG_LABEL_FONTSIZE)
    ax.yaxis.set_label_coords(-0.1, 0.5)  
    ax.yaxis.set_major_formatter(FuncFormatter(_format_sci_tick))
    ax.tick_params(axis="both", labelsize=FIG_TICK_FONTSIZE)
    ax.tick_params(axis="y", colors=LOSS_COLOR)
    if history_epochs.size > 1:
        ax.set_xlim(float(history_epochs.min()), float(history_epochs.max()))
    xticks = _epoch_xticks(history_epochs)
    ax.set_xticks(xticks)

    tau_ax = ax.twinx()
    tau_ax.plot(history_epochs, tau, color=TAU_COLOR, linewidth=0.9, linestyle="--", zorder=2)
    tau_ax.set_ylabel("Gumbel-Softmax\ntempreture τ", color=TAU_COLOR, labelpad=1.5, fontsize=FIG_LABEL_FONTSIZE)
    tau_ax.yaxis.set_label_coords(1.08, 0.5) 
    tau_ax.tick_params(axis="y", colors=TAU_COLOR, labelsize=FIG_TICK_FONTSIZE)
    tau_ax.spines["top"].set_visible(False)

    if not include_lr:
        return tau_ax, None

    lr_ax = ax.twinx()
    lr_ax.spines["right"].set_position(("axes", 1.15))
    lr_ax.plot(history_epochs, lr, color=LR_COLOR, linewidth=0.9, linestyle=":", zorder=2)
    lr_ax.yaxis.set_major_formatter(FuncFormatter(_format_sci_tick))
    lr_ax.set_ylabel("Learning rate", color=LR_COLOR, labelpad=5.0, fontsize=FIG_LABEL_FONTSIZE)
    lr_ax.tick_params(axis="y", colors=LR_COLOR, pad=2.0, labelsize=FIG_TICK_FONTSIZE)
    lr_ax.spines["top"].set_visible(False)
    lr_ax.spines["right"].set_visible(True)
    return tau_ax, lr_ax


def _epoch_xticks(history_epochs: np.ndarray) -> np.ndarray:
    epochs_int = np.asarray(history_epochs, dtype=int)
    if epochs_int.size == 0:
        return epochs_int
    start = int(epochs_int.min())
    stop = int(epochs_int.max())
    ticks = [start]
    next_tick = ((start + 4) // 5) * 5
    while next_tick <= stop:
        if next_tick != ticks[-1]:
            ticks.append(next_tick)
        next_tick += 5
    if ticks[-1] != stop:
        ticks.append(stop)
    return np.asarray(ticks, dtype=int)


def _add_shared_hour_label(fig: plt.Figure, ax_hour: plt.Axes, ax_below: plt.Axes) -> None:
    hour_pos = ax_hour.get_position()
    below_pos = ax_below.get_position()
    fig.text(
        0.5 * (hour_pos.x0 + hour_pos.x1),
        0.47 * (hour_pos.y0 + below_pos.y1),
        "Hour",
        ha="center",
        va="center",
        fontsize=FIG_LABEL_FONTSIZE,
    )


def _build_figure(
    *,
    date: str,
    epochs: np.ndarray,
    power: np.ndarray,
    price: np.ndarray | None = None,
    head: np.ndarray | None = None,
    history_epochs: np.ndarray | None = None,
    losses: np.ndarray | None = None,
    tau: np.ndarray | None = None,
    lr: np.ndarray | None = None,
    miqp_pw: np.ndarray | None = None,
    miqp_gl: np.ndarray | None = None,
    include_lr: bool = True,
    plain_mode: bool = False,
    feasible_power_bounds: dict[str, np.ndarray] | None = None,
    target_head: float | None = None,
    show_title: bool = True,
) -> plt.Figure:
    apply_style()
    if (
        price is not None
        and head is not None
        and history_epochs is not None
        and losses is not None
        and tau is not None
        and lr is not None
    ):
        fig = plt.figure(figsize=(COL_WIDTH, 3.25))
        gs = fig.add_gridspec(
            4,
            2,
            width_ratios=[40.0, 1.4],
            height_ratios=[1.0, 1.0, 0.22, 1.0],
            hspace=0.10,
            wspace=0.04,
        )
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax_price = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_train = fig.add_subplot(gs[3, 0])
    else:
        fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))
        ax_price = None
        ax_train = None
        cax = None
    _plot_epoch_dispatch(
        ax,
        epochs=epochs,
        series=np.asarray(head if plain_mode else power, dtype=float),
        ylabel="Head (m)" if plain_mode else "Power (MW)",
        title="",
        show_xlabel=ax_price is None,
        cax=cax,
        miqp_pw=miqp_pw,
        miqp_gl=miqp_gl,
        epoch_cmap=plt.get_cmap("viridis_r") if plain_mode else EPOCH_CMAP,
        final_label_loc="upper left" if plain_mode else "upper center",
        target_level=target_head if plain_mode else None,
        nonfinal_alpha=0.37 if plain_mode else EPOCH_DISPATCH_ALPHA,
    )
    if ax_price is not None:
        final_aux = np.asarray(power if plain_mode else head, dtype=float)
        if final_aux.ndim == 2:
            final_aux = final_aux[-1]
        if plain_mode:
            _plot_plain_price_and_power(
                ax_price,
                price=np.asarray(price, dtype=float),
                power=final_aux,
                feasible_power_bounds=feasible_power_bounds,
            )
        else:
            _plot_price_and_head(
                ax_price,
                price=np.asarray(price, dtype=float),
                aux=final_aux,
                aux_label="Head (m)",
                price_color=PRICE_COLOR,
                aux_color=HEAD_COLOR,
                aux_as_step=False,
            )
    if ax_train is not None:
        _plot_training_curve(
            ax_train,
            history_epochs=np.asarray(history_epochs, dtype=float),
            losses=np.asarray(losses, dtype=float),
            tau=np.asarray(tau, dtype=float),
            lr=np.asarray(lr, dtype=float),
            include_lr=include_lr,
        )
    if show_title:
        fig.suptitle(f"Epoch dispatch colormap - {date}", fontsize=9)
        fig.subplots_adjust(left=0.14, right=0.87, bottom=0.16, top=0.90, hspace=0.14)
    else:
        fig.subplots_adjust(left=0.16, right=0.82, bottom=0.11, top=0.98)
    if ax_price is not None and ax_train is not None:
        _add_shared_hour_label(fig, ax_price, ax_train)
    return fig


def make_figures(cache_path: Path, output_dir: Path, *, include_miqp: bool = True) -> tuple[Path, Path]:
    """Write the epoch dispatch evolution PDFs and return their paths."""
    cache_path = Path(cache_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date, epochs, power, price, head = load_epoch_cache(cache_path)
    history_epochs, losses, tau, lr = load_training_curve(cache_path)

    plain_path = output_dir / OUTPUT_PDF
    overlay_path = output_dir / OUTPUT_PDF_WITH_MIQP
    feasible_power_bounds = load_feasible_power_bounds(cache_path, head)
    target_head = load_target_head(cache_path)

    fig = _build_figure(
        date=date,
        epochs=epochs,
        power=power,
        price=price,
        head=head,
        history_epochs=history_epochs,
        losses=losses,
        tau=tau,
        lr=lr,
        miqp_pw=None,
        miqp_gl=None,
        include_lr=False,
        plain_mode=True,
        feasible_power_bounds=feasible_power_bounds,
        target_head=target_head,
        show_title=False,
    )
    fig.savefig(plain_path, bbox_inches="tight")
    plt.close(fig)

    miqp_pw = None
    miqp_gl = None
    if include_miqp:
        miqp_pw = load_hourly_power_schedule(MIQP_PW_RESULTS_CSV, date)
        miqp_gl = load_hourly_power_schedule(MIQP_GL_RESULTS_CSV, date)

    fig = _build_figure(
        date=date,
        epochs=epochs,
        power=power,
        price=price,
        head=head,
        history_epochs=history_epochs,
        losses=losses,
        tau=tau,
        lr=lr,
        miqp_pw=miqp_pw,
        miqp_gl=miqp_gl,
        include_lr=True,
        plain_mode=False,
        show_title=False,
    )
    fig.savefig(overlay_path, bbox_inches="tight")
    plt.close(fig)

    return plain_path, overlay_path


def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    """CLI entrypoint for generating the epoch dispatch evolution figures."""
    parser = argparse.ArgumentParser(description="Generate epoch dispatch evolution figures.")
    parser.add_argument("--cache", required=True, type=Path, help="Path to the replay cache NPZ.")
    parser.add_argument(
        "--output-dir",
        default=FIGS_OUT,
        type=Path,
        help="Directory for output PDFs.",
    )
    parser.add_argument(
        "--no-miqp",
        action="store_true",
        help="Skip overlaying MIQP comparison curves.",
    )
    args = parser.parse_args(argv)
    return make_figures(args.cache, args.output_dir, include_miqp=not args.no_miqp)


__all__ = [
    "load_epoch_cache",
    "load_hourly_power_schedule",
    "make_figures",
    "main",
]


if __name__ == "__main__":
    main()
