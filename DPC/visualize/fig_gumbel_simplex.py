"""Generate a 4-panel simplex visualization of Gumbel-Softmax samples
at different temperatures, showing exploration-to-commitment transition."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ── CDC / IEEE style ──
plt.rcParams.update({
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
    "figure.dpi": 300,
})

# ── Vertices of equilateral triangle ──
V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])


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


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "figs" / "gumbel_simplex.pdf"


def main(out_path: Path | None = None) -> Path:
    out_path = Path(out_path) if out_path is not None else DEFAULT_OUTPUT

    # ── Parameters ──
    logits = np.array([0.5, 0.2, 1.5])  # pump, idle, turbine
    taus = [5.0, 1.0, 0.3, 0.05]
    n_samples = 600
    rng = np.random.default_rng(42)

    # Colors for each mode vertex
    mode_colors = np.array([
        [0.220, 0.459, 0.694],  # pump  — steel blue
        [0.506, 0.694, 0.337],  # idle  — muted green
        [0.906, 0.494, 0.200],  # turbine — warm orange
    ])
    mode_labels = ["Pump", "Idle", "Turbine"]

    # ── Figure ──
    fig, axes = plt.subplots(1, 4, figsize=(3.5, 1.35))

    # Triangle outline coordinates (closed path)
    tri = np.vstack([V, V[0]])

    for ax, tau in zip(axes, taus):
        samples = gumbel_softmax_samples(logits, tau, n_samples, rng)
        xy = bary_to_cart(samples)
        vidx = nearest_vertex(samples)

        # Color each point by its nearest vertex
        colors = mode_colors[vidx]

        # Draw triangle
        ax.plot(tri[:, 0], tri[:, 1], color="0.6", linewidth=0.5, zorder=1)

        # Vertex markers
        for k in range(3):
            ax.plot(V[k, 0], V[k, 1], "o", color=mode_colors[k],
                    markersize=3.5, markeredgecolor="0.3", markeredgewidth=0.4,
                    zorder=5)

        # Scatter samples — larger points at low tau where they cluster on edges
        alpha = 0.5 if tau >= 1.0 else 0.4
        size = 1.2 if tau >= 0.3 else 2.0
        ax.scatter(xy[:, 0], xy[:, 1], s=size, alpha=alpha, c=colors,
                   edgecolors="none", zorder=3)

        # Vertex labels — outside corners, offset away from triangle
        ax.text(V[0, 0] - 0.06, V[0, 1] - 0.06, "P",
                ha="right", va="top", fontsize=6, fontweight="bold",
                color=mode_colors[0] * 0.7)
        ax.text(V[1, 0] + 0.06, V[1, 1] - 0.06, "I",
                ha="left", va="top", fontsize=6, fontweight="bold",
                color=mode_colors[1] * 0.7)
        ax.text(V[2, 0] + 0.08, V[2, 1] + 0.02, "T",
                ha="left", va="center", fontsize=6, fontweight="bold",
                color=mode_colors[2] * 0.7)

        # Temperature label
        ax.set_title(rf"$\tau\!=\!{tau}$", fontsize=7.5, pad=4)

        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.15, np.sqrt(3) / 2 + 0.22)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout(w_pad=0.2)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(main())
