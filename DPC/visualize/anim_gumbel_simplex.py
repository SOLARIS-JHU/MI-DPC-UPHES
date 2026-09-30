"""Website animation: Gumbel-Softmax samples on the simplex while tau anneals
from 5.0 down to 0.05 on a log scale. Companion to fig_gumbel_simplex.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from DPC.visualize.anim_utils import fig_to_rgb, save_gif
from DPC.visualize.style import apply_theme, current_theme, theme_background, themed

# ── CDC / IEEE style (matches fig_gumbel_simplex.py) ──
_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 0.9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
}

# ── Vertices of equilateral triangle ──
V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])

# ── Parameters ──
LOGITS = np.array([0.5, 0.2, 1.5])  # pump, idle, turbine
TAU_HI, TAU_LO = 5.0, 0.05
N_SAMPLES = 1500
DOT_SIZE = 6.0
FRAME_MS = 100
HOLD_MS = 1500

MODE_COLORS = np.array([
    [0.220, 0.459, 0.694],  # pump  — steel blue
    [0.506, 0.694, 0.337],  # idle  — muted green
    [0.906, 0.494, 0.200],  # turbine — warm orange
])


def bary_to_cart(p):
    """Barycentric coordinates (N×3) -> Cartesian (N×2)."""
    return p @ V


def gumbel_softmax_samples(logits, tau, n, rng):
    """Draw n Gumbel-Softmax samples (no STE, just soft)."""
    g = rng.gumbel(size=(n, len(logits)))
    y = (logits + g) / tau
    y -= y.max(axis=1, keepdims=True)
    e = np.exp(y)
    return e / e.sum(axis=1, keepdims=True)


def nearest_vertex(samples):
    """Return index of nearest vertex for each sample."""
    return np.argmax(samples, axis=1)


def render_frames(*, n_frames: int = 60, dpi: int = 160) -> tuple[list[Image.Image], list[float]]:
    """Render the annealing animation; returns the frames and their temperatures."""
    rng = np.random.default_rng(42)
    # Log-linear schedule: tau_t = TAU_HI * (TAU_LO/TAU_HI)^(t/(N-1))
    taus = TAU_HI * (TAU_LO / TAU_HI) ** (np.arange(n_frames) / (n_frames - 1))
    epochs = np.arange(1, n_frames + 1)

    with plt.rc_context(_RC):
        apply_theme()
        mode_colors = np.clip(MODE_COLORS * (1.25 if current_theme() == "web-dark" else 1.0), 0, 1)
        label_scale = 1.0 if current_theme() == "web-dark" else 0.7

        # ── Figure layout: simplex on left, tau-vs-epoch curve on right ──
        fig = plt.figure(figsize=(4.6, 2.2), dpi=dpi)
        fig.patch.set_facecolor(theme_background())
        fig.patch.set_alpha(1.0)
        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[1, 1],
            left=0.04, right=0.97, top=0.92, bottom=0.20,
            wspace=0.15,
        )
        ax = fig.add_subplot(gs[0])
        ax_curve = fig.add_subplot(gs[1])

        # Triangle outline + vertex markers + vertex labels (static)
        tri = np.vstack([V, V[0]])
        ax.plot(tri[:, 0], tri[:, 1], color=themed("0.6", "faint"), linewidth=0.5, zorder=1)
        for k in range(3):
            ax.plot(V[k, 0], V[k, 1], "o", color=mode_colors[k],
                    markersize=3.5, markeredgecolor=themed("0.3", "muted"), markeredgewidth=0.4,
                    zorder=5)
        ax.text(V[0, 0] - 0.02, V[0, 1] - 0.02, "P",
                ha="right", va="top", fontsize=6, fontweight="bold",
                color=mode_colors[0] * label_scale)
        ax.text(V[1, 0] + 0.02, V[1, 1] - 0.02, "I",
                ha="left", va="top", fontsize=6, fontweight="bold",
                color=mode_colors[1] * label_scale)
        ax.text(V[2, 0], V[2, 1] + 0.03, "T",
                ha="center", va="bottom", fontsize=6, fontweight="bold",
                color=mode_colors[2] * label_scale)

        pad = 0.02
        ax.set_xlim(-pad, 1 + pad)
        ax.set_ylim(-pad, np.sqrt(3) / 2 + pad)
        ax.set_aspect("equal")
        ax.set_anchor("C")
        ax.axis("off")

        scat = ax.scatter(
            np.zeros(N_SAMPLES), np.zeros(N_SAMPLES),
            s=DOT_SIZE, alpha=0.45, c=np.tile(mode_colors[0], (N_SAMPLES, 1)),
            edgecolors="none", zorder=3,
        )

        # ── Tau-vs-epoch curve (linear y) ──
        ax_curve.plot(epochs, taus, color=themed("0.4", "muted"), linewidth=1.0, zorder=2)
        ax_curve.set_xlim(1, n_frames)
        ax_curve.set_ylim(0, TAU_HI * 1.05)
        ax_curve.set_xlabel("Annealing step", labelpad=1, fontsize=8)
        ax_curve.set_ylabel(r"$\tau$", labelpad=1, fontsize=8)
        ax_curve.set_yticks([0, 2, 4])
        ax_curve.tick_params(axis="both", length=2.0, pad=1, labelsize=7)
        for spine in ("top", "right"):
            ax_curve.spines[spine].set_visible(False)

        title = ax_curve.set_title(rf"$\tau = {TAU_HI:.2f}$", fontsize=8, pad=2)

        # Moving marker showing current (epoch, tau)
        (tau_marker,) = ax_curve.plot(
            [epochs[0]], [taus[0]], marker="o", color=themed("0.15", "fg"),
            markersize=4.5, markeredgewidth=0, zorder=4,
        )

        frames = []
        try:
            for i, tau in enumerate(taus):
                samples = gumbel_softmax_samples(LOGITS, tau, N_SAMPLES, rng)
                scat.set_offsets(bary_to_cart(samples))
                scat.set_facecolors(mode_colors[nearest_vertex(samples)])
                tau_marker.set_data([epochs[i]], [tau])
                title.set_text(rf"$\tau = {tau:.3g}$")
                frames.append(fig_to_rgb(fig))
        finally:
            plt.close(fig)
    return frames, [float(t) for t in taus]


def make_animation(out_path: str | Path) -> Path:
    frames, _ = render_frames()
    durations = [FRAME_MS] * (len(frames) - 1) + [HOLD_MS]
    return save_gif(frames, durations, out_path)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Generate the Gumbel-Softmax annealing GIF for the project page.")
    parser.add_argument("--output", type=Path, required=True, help="Output GIF path.")
    args = parser.parse_args(argv)
    out = make_animation(args.output)
    print(f"Saved: {out}  ({out.stat().st_size:,} bytes)")
    return out


if __name__ == "__main__":
    main()
