"""Website animation: how the schedule for one day changes as training proceeds.

Each frame shows the policy's schedule for the representative day at one point
in training (power against price, then head and volume), with the schedules of
the two preceding epochs as faint ghosts. A strip underneath tracks the
training loss and the Gumbel-Softmax temperature: the part of training already
done is opaque, the part still to come is very translucent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

from DPC.config import load_system_params
from DPC.visualize.anim_utils import fig_to_rgb, save_gif
from DPC.visualize.fig_dispatch_evolution import _load_cache_meta, load_epoch_cache, load_training_curve
from DPC.visualize.style import (
    FULL_WIDTH,
    apply_style,
    by_theme,
    cleanup_axes,
    grid_kw,
    theme_background,
    themed,
)

FUTURE_ALPHA = 0.12
GHOST_ALPHAS = (0.28, 0.12)  # previous epoch, the one before it
FRAME_MS = 110
EPOCH_HOLD_MS = 320
FINAL_HOLD_MS = 2200
FIGURE_HEIGHT = 3.9


def load_training_run_payload(cache_path: str | Path) -> dict[str, object]:
    """Collect everything the animation needs from a replay cache and its run directory."""
    cache_path = Path(cache_path)
    date, epochs, power, price, head = load_epoch_cache(cache_path)
    hist_epochs, losses, tau, _ = load_training_curve(cache_path)
    by_epoch = {int(e): (float(l), float(t)) for e, l, t in zip(hist_epochs, losses, tau)}
    missing = [int(e) for e in epochs if int(e) not in by_epoch]
    if missing:
        raise ValueError(f"Training history has no entry for epochs {missing}")

    meta = _load_cache_meta(cache_path)
    system_params = load_system_params(
        str(meta.get("pkl_path", "preprocess.pkl")),
        device=torch.device("cpu"),
        physics_mode="nonlinear",
        inverse_pkl_path=meta.get("inverse_pkl"),
    )
    head_t = torch.as_tensor(head, dtype=torch.float32)
    bounds = {
        key: system_params[key](head_t).detach().cpu().numpy() for key in ("pos_min", "pos_max", "neg_min", "neg_max")
    }
    volume = system_params["h_to_v_low"](head_t).detach().cpu().numpy()
    return {
        "date": date,
        "hours": np.arange(price.shape[0], dtype=float),
        "price": price,
        "epochs": epochs,
        "power": power,
        "head": head,
        "volume": volume,
        "bounds": bounds,
        "min_volume": 0.0,
        "max_volume": float(system_params.get("max_vol_low", np.max(volume))),
        "target_volume": float(system_params["target_vol_low"]),
        "loss": np.asarray([by_epoch[int(e)][0] for e in epochs]),
        "tau": np.asarray([by_epoch[int(e)][1] for e in epochs]),
    }


def _blend(values: np.ndarray, position: float) -> np.ndarray:
    """Linear interpolation between the rows of values at a fractional row index."""
    lower = int(np.floor(position))
    upper = min(lower + 1, values.shape[0] - 1)
    weight = position - lower
    return (1.0 - weight) * values[lower] + weight * values[upper]


def render_frames(
    payload: dict[str, object], *, steps_per_epoch: int = 2, dpi: int = 150
) -> tuple[list[Image.Image], list[float]]:
    """Render the animation; returns the frames and the (fractional) epoch each one shows."""
    apply_style()
    hours = np.asarray(payload["hours"], dtype=float)
    price = np.asarray(payload["price"], dtype=float)
    epochs = np.asarray(payload["epochs"], dtype=float)
    power = np.asarray(payload["power"], dtype=float)
    head = np.asarray(payload["head"], dtype=float)
    volume = np.asarray(payload["volume"], dtype=float)
    bounds = {key: np.asarray(value, dtype=float) for key, value in payload["bounds"].items()}
    loss = np.asarray(payload["loss"], dtype=float)
    tau = np.asarray(payload["tau"], dtype=float)
    n_epochs = epochs.shape[0]
    if loss.shape[0] != n_epochs or tau.shape[0] != n_epochs:
        raise ValueError("The payload needs one loss and tau value per epoch")

    power_color = by_theme("#D62728", "#ff6b6b")
    price_color = by_theme("#1F77B4", "#68ace5")
    head_color = by_theme("#E8850C", "#ffa94d")
    volume_color = by_theme("#2CA02C", "#63d471")
    loss_color = by_theme("#7B1FA2", "#c792ea")
    tau_color = by_theme("#FF7F0E", "#ffa94d")
    neutral = themed("0.35", "muted")

    fig, (ax_power, ax_head, ax_loss) = plt.subplots(3, 1, figsize=(FULL_WIDTH, FIGURE_HEIGHT), dpi=dpi)
    fig.patch.set_facecolor(theme_background())
    fig.patch.set_alpha(1.0)
    ax_price = ax_power.twinx()
    ax_volume = ax_head.twinx()
    ax_tau = ax_loss.twinx()
    ax_price.set_zorder(1)
    ax_power.set_zorder(2)
    ax_power.patch.set_alpha(0.0)

    # ── Static content ──
    ax_price.step(hours, price, where="mid", color=price_color, linewidth=1.1, zorder=0)
    ax_power.axhline(0.0, color=power_color, linewidth=0.7, alpha=0.65, zorder=2)
    ax_volume.axhline(
        float(payload["target_volume"]), color=volume_color, linestyle=(0, (4, 2)), linewidth=0.7, zorder=2,
        label="Target volume",
    )
    ax_volume.legend(loc="upper left", frameon=False, fontsize=7)

    power_pad = 0.08 * (bounds["pos_max"].max() - bounds["neg_min"].min())
    ax_power.set_ylim(bounds["neg_min"].min() - power_pad, bounds["pos_max"].max() + power_pad)
    head_pad = max(0.06 * np.ptp(head), 0.5)
    ax_head.set_ylim(head.min() - head_pad, head.max() + head_pad)
    volume_pad = 0.05 * (float(payload["max_volume"]) - float(payload["min_volume"]))
    ax_volume.set_ylim(float(payload["min_volume"]) - volume_pad, float(payload["max_volume"]) + volume_pad)

    for ax in (ax_power, ax_head):
        cleanup_axes(ax)
        ax.xaxis.grid(True, **grid_kw())
        ax.set_xlim(hours[0], hours[-1])
        ax.set_xticks(range(0, int(hours[-1]) + 1, 4))
    ax_power.tick_params(axis="x", labelbottom=False)
    ax_power.set_ylabel("Power (MW)", color=power_color, fontsize=8)
    ax_power.tick_params(axis="y", colors=power_color, labelsize=7)
    ax_price.set_ylabel("Price (€/MWh)", color=price_color, fontsize=8)
    ax_price.tick_params(axis="y", colors=price_color, labelsize=7)
    ax_price.spines["top"].set_visible(False)
    ax_head.set_ylabel("Head (m)", color=head_color, fontsize=8)
    ax_head.tick_params(axis="y", colors=head_color, labelsize=7)
    ax_head.tick_params(axis="x", labelsize=7)
    ax_head.set_xlabel("Hour of the day", fontsize=8, labelpad=1)
    ax_volume.set_ylabel("Volume (m$^3$)", color=volume_color, fontsize=8)
    ax_volume.tick_params(axis="y", colors=volume_color, labelsize=7)
    ax_volume.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax_volume.set_yticks([tick for tick in ax_volume.get_yticks() if 0.0 <= tick <= 0.9 * float(payload["max_volume"])])
    ax_volume.spines["top"].set_visible(False)

    # ── Training strip: faint full curves now, opaque copies revealed by a clip ──
    cleanup_axes(ax_loss, grid=False)
    ax_tau.spines["top"].set_visible(False)
    ax_loss.plot(epochs, loss, color=loss_color, linewidth=1.0, alpha=FUTURE_ALPHA)
    ax_tau.plot(epochs, tau, color=tau_color, linewidth=1.0, linestyle="--", alpha=FUTURE_ALPHA)
    (loss_line,) = ax_loss.plot(epochs, loss, color=loss_color, linewidth=1.0, marker="o", markersize=2.2)
    (tau_line,) = ax_tau.plot(epochs, tau, color=tau_color, linewidth=1.0, linestyle="--")
    cursor = ax_loss.axvline(epochs[0], color=neutral, linewidth=0.7, linestyle=(0, (2, 2)), zorder=5)
    ax_loss.set_xlim(epochs[0], epochs[-1])
    ax_loss.set_xlabel("Training epoch", fontsize=8, labelpad=1)
    ax_loss.set_ylabel("Loss", color=loss_color, fontsize=8)
    ax_loss.tick_params(axis="y", colors=loss_color, labelsize=7)
    ax_loss.tick_params(axis="x", labelsize=7)
    ax_tau.set_ylabel(r"Temperature $\tau$", color=tau_color, fontsize=8)
    ax_tau.tick_params(axis="y", colors=tau_color, labelsize=7)
    ax_tau.set_ylim(0.0, tau.max() * 1.08)

    # Two schedule panels close together, the training strip set apart below them.
    for axes, (bottom, height) in (((ax_power, ax_price), (0.655, 0.275)), ((ax_head, ax_volume), (0.35, 0.275)), ((ax_loss, ax_tau), (0.105, 0.135))):
        for ax in axes:
            ax.set_position([0.10, bottom, 0.80, height])
    label = fig.text(0.5, 0.965, "", ha="center", va="center", fontsize=9, color=themed("black", "fg"))

    # ── Animated artists ──
    (power_line,) = ax_power.step(hours, power[0], where="mid", color=power_color, linewidth=1.3, zorder=10)
    (head_line,) = ax_head.plot(hours, head[0], color=head_color, linewidth=1.3, zorder=4)
    (volume_line,) = ax_volume.plot(hours, volume[0], color=volume_color, linewidth=1.3, zorder=4)
    ghosts = [
        (
            ax_power.step(hours, power[0], where="mid", color=power_color, linewidth=1.0, alpha=alpha, zorder=6)[0],
            ax_head.plot(hours, head[0], color=head_color, linewidth=1.0, alpha=alpha, zorder=3)[0],
        )
        for alpha in GHOST_ALPHAS
    ]

    positions = np.linspace(0.0, n_epochs - 1.0, (n_epochs - 1) * steps_per_epoch + 1)
    frames: list[Image.Image] = []
    try:
        for position in positions:
            state = {key: _blend(value, position) for key, value in bounds.items()}
            # fill_between cannot be updated in place; redraw the feasible bands each frame.
            bands = [
                ax_power.fill_between(hours, state[lo], state[hi], step="mid", color=power_color, alpha=0.14, zorder=0)
                for lo, hi in (("pos_min", "pos_max"), ("neg_min", "neg_max"))
            ]
            edges = [
                ax_power.step(hours, state[key], where="mid", color=power_color, linewidth=0.65, alpha=0.45, zorder=1)[0]
                for key in ("pos_min", "pos_max", "neg_min", "neg_max")
            ]
            power_line.set_ydata(_blend(power, position))
            head_line.set_ydata(_blend(head, position))
            volume_line.set_ydata(_blend(volume, position))
            for back, (ghost_power, ghost_head) in enumerate(ghosts, start=1):
                index = int(np.floor(position)) - back + 1
                visible = index >= 0 and index < position
                ghost_power.set_visible(visible)
                ghost_head.set_visible(visible)
                if visible:
                    ghost_power.set_ydata(power[index])
                    ghost_head.set_ydata(head[index])

            epoch_now = float(_blend(epochs, position))
            for line, ax in ((loss_line, ax_loss), (tau_line, ax_tau)):
                line.set_clip_path(
                    Rectangle((epochs[0] - 1.0, 0.0), epoch_now - epochs[0] + 1.0, 1.0, transform=ax.get_xaxis_transform())
                )
            cursor.set_xdata([epoch_now, epoch_now])
            shown = int(round(epoch_now)) if abs(epoch_now - round(epoch_now)) < 1e-9 else int(np.floor(epoch_now))
            label.set_text(
                rf"Epoch {shown} of {int(epochs[-1])}     $\tau$ = {float(_blend(tau, position)):.2f}"
            )
            frames.append(fig_to_rgb(fig))
            for artist in (*bands, *edges):
                artist.remove()
    finally:
        plt.close(fig)
    return frames, [float(_blend(epochs, p)) for p in positions]


def make_animation(out_path: str | Path, *, cache_path: str | Path, still_path: str | Path | None = None) -> Path:
    payload = load_training_run_payload(cache_path)
    frames, positions = render_frames(payload)
    durations = [EPOCH_HOLD_MS if float(p).is_integer() else FRAME_MS for p in positions]
    durations[-1] = FINAL_HOLD_MS
    if still_path is not None:
        Path(still_path).parent.mkdir(parents=True, exist_ok=True)
        frames[-1].save(still_path, optimize=True)
    return save_gif(frames, durations, out_path)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Generate the training-run GIF for the project page.")
    parser.add_argument("--cache", type=Path, required=True, help="Path to an epoch_dispatch_trace.npz replay cache.")
    parser.add_argument("--output", type=Path, required=True, help="Output GIF path.")
    parser.add_argument("--still", type=Path, default=None, help="Also write the final frame as a PNG (reduced-motion fallback).")
    args = parser.parse_args(argv)
    out = make_animation(args.output, cache_path=args.cache, still_path=args.still)
    print(f"Saved: {out}  ({out.stat().st_size:,} bytes)")
    return out


if __name__ == "__main__":
    main()
