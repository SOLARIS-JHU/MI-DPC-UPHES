"""Generate a 2-panel figure comparing standard clamp vs STE clamp."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ── CDC / IEEE style (no usetex for portability) ──
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

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "figs" / "ste_clamp.pdf"


def main(out_path: Path | None = None) -> Path:
    out_path = Path(out_path) if out_path is not None else DEFAULT_OUTPUT

    # ── Bounds ──
    a, b = 0.0, 1.0
    x = np.linspace(a - 1.5, b + 1.5, 500)

    # Forward output (identical for both)
    y_clamp = np.clip(x, a, b)

    # Gradients
    grad_standard = np.where((x > a) & (x < b), 1.0, 0.0)
    grad_ste = np.ones_like(x)

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.3))

    # Left panel: forward output
    ax1.plot(x, y_clamp, color="C0", zorder=3, alpha=0.95)
    ax1.plot(x, y_clamp, color="C1", linestyle="--", zorder=4, alpha=0.95)
    ax1.axhline(a, color="0.7", linewidth=0.5, linestyle="--", zorder=1)
    ax1.axhline(b, color="0.7", linewidth=0.5, linestyle="--", zorder=1)
    ax1.axvline(a, color="0.85", linewidth=0.4, linestyle=":", zorder=1)
    ax1.axvline(b, color="0.85", linewidth=0.4, linestyle=":", zorder=1)
    ax1.set_xlabel(r"$x$", labelpad=-4)
    ax1.set_ylabel(r"$y$")
    ax1.set_xlim(-1.0, 2.0)
    ax1.set_xticks([a, b])
    ax1.set_xticklabels([r"$a$", r"$b$"])
    ax1.set_ylim(-0.3, 1.5)
    ax1.set_yticks([a, b])
    ax1.set_yticklabels([r"$a$", r"$b$"])
    ax1.text(
        0.05, 0.93, "(a) Forward pass",
        transform=ax1.transAxes, va="top", fontsize=8,
    )

    # Right panel: gradients
    ax2.plot(x, grad_standard, color="C0", label=r"Standard $y=\Pi_{[a,b]}(x)$", zorder=3, alpha=0.95)
    ax2.plot(x, grad_ste, color="C1", linestyle="--", label=r"STE $y=\widetilde{\Pi}_{[a,b]}(x)$", zorder=4, alpha=0.95)
    ax2.axvline(a, color="0.85", linewidth=0.4, linestyle=":", zorder=1)
    ax2.axvline(b, color="0.85", linewidth=0.4, linestyle=":", zorder=1)
    ax2.set_xlabel(r"$x$", labelpad=-4)
    ax2.set_ylabel(r"$\partial y\,/\,\partial x$")
    ax2.set_xlim(-1.0, 2.0)
    ax2.set_xticks([a, b])
    ax2.set_xticklabels([r"$a$", r"$b$"])
    ax2.set_ylim(-0.3, 1.5)
    ax2.text(
        0.05, 0.93, "(b) Backward pass",
        transform=ax2.transAxes, va="top", fontsize=8,
    )

    fig.tight_layout(rect=(0, 0, 1, 1))

    # Shared legend below both panels, anchored to figure
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=2,
               bbox_to_anchor=(0.5, -0.08), columnspacing=1.5, handlelength=1.5,
               fontsize=8, borderaxespad=0.0)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    print(main())
