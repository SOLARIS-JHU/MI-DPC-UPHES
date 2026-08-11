"""CDC / IEEE conference figure style and color palette."""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Default output directory ──
# All figure scripts write PDFs here unless overridden by --output-dir.
FIGS_OUT = Path("/mnt/d/Repositories/L2O_UPHES_Project/figs")

# ── Benchmark data root ──
_DPC_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = _DPC_ROOT / "outputs" / "benchmark_suite"
MIQP_ROOT = _DPC_ROOT.parent / "MIQP"

# ── CDC / IEEE style ──
STYLE = {
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
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
}

# ── Color palette ──
C_MIDPC = "#4682B4"       # MI-DPC / Transformer / Batch / Annealed — steel blue
C_MIQP_PW = "#E8850C"     # MIQP piecewise — orange
C_MIQP_GL = "#2CA02C"     # MIQP global-linear — green
C_BILSTM = "#E8685D"      # Bi-LSTM — coral
C_MLP = "#9467BD"          # MLP — purple
C_CNN = "#7F7F7F"          # CNN — gray
C_STEP = "#DC143C"         # Step dynamics — crimson
C_FIXED = "#DAA520"        # Fixed-low temperature — goldenrod

# Mode colors (for dispatch plots)
C_TURBINE = "#E8850C"      # orange
C_IDLE = "#D3D3D3"         # light gray
C_PUMP = "#4682B4"         # steel blue

# Epoch-dispatch overlay style
EPOCH_CMAP = plt.get_cmap("plasma_r")
EPOCH_DISPATCH_ALPHA = 0.5
EPOCH_DISPATCH_LINEWIDTH = 1.1

# ── Sizing ──
COL_WIDTH = 3.5            # single-column figure width (inches)
FULL_WIDTH = 7.0           # full-width figure width (inches)

# ── Grid style ──
GRID_KW = dict(color="#E0E0E0", linewidth=0.5, zorder=0)

# ── Reference values ──
MIQP_PW_MEAN_EXPOST = 2529.85  # EUR/day, 19-day benchmark mean
MIQP_GL_MEAN_EXPOST = 1997.35  # EUR/day, 19-day benchmark mean


def apply_style():
    """Apply CDC style globally."""
    plt.rcParams.update(STYLE)


def cleanup_axes(ax, grid=True):
    """Remove top/right spines, optionally add horizontal grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.yaxis.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
