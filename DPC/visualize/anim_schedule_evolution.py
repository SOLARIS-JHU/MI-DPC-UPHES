"""Website animation: the converged MI-DPC schedule revealed hour by hour.

Everything up to the moving cursor is opaque; the rest of the day is drawn
very translucent so the future is visible only as a faint preview.
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

from DPC.visualize.anim_utils import fig_to_rgb, save_gif
from DPC.visualize.fig_schedule_evolution import (
    LINEWIDTH_MAIN,
    draw_schedule,
    load_schedule_payload,
)
from DPC.visualize.style import theme_background, themed

FUTURE_ALPHA = 0.12
FRAME_MS = 80
HOLD_MS = 1500
_X_START = -1.0  # left of hour 0, so the clip starts empty


def render_frames(payload: dict[str, object], *, steps_per_hour: int = 3, dpi: int = 200) -> list[Image.Image]:
    fig, axes, faint = draw_schedule(payload, trajectory_alpha=FUTURE_ALPHA)
    fig.set_dpi(dpi)
    fig.patch.set_facecolor(theme_background())
    fig.patch.set_alpha(1.0)

    hours = np.asarray(payload["hours"], dtype=float)
    series = {
        "price": ("step", np.asarray(payload["price"], dtype=float), 0),
        "power": ("step", np.asarray(payload["power"], dtype=float), 10),
        "head": ("plot", np.asarray(payload["head"], dtype=float), 3),
        "volume": ("plot", np.asarray(payload["volume"], dtype=float), 3),
    }

    revealed = []
    for key, (kind, values, zorder) in series.items():
        ax = axes[key]
        color = faint[key].get_color()  # same themed colour as the faint preview
        if kind == "step":
            (line,) = ax.step(hours, values, where="mid", color=color, linewidth=LINEWIDTH_MAIN, zorder=zorder + 0.5)
        else:
            (line,) = ax.plot(hours, values, color=color, linewidth=LINEWIDTH_MAIN, zorder=zorder + 0.5)
        revealed.append((line, ax))

    cursors = [
        axes[key].axvline(0.0, color=themed("0.35", "muted"), linewidth=0.7, linestyle=(0, (2, 2)), zorder=20)
        for key in ("power", "head")
    ]

    last = float(hours[-1])
    n_frames = int(round(last * steps_per_hour)) + 1 if steps_per_hour > 1 else len(hours)
    positions = np.linspace(0.0, last, n_frames)

    frames = []
    try:
        for x in positions:
            for line, ax in revealed:
                # x in data coordinates, y spanning the axes; a fresh patch per
                # frame because a clip path does not track later patch edits.
                line.set_clip_path(Rectangle((_X_START, 0.0), x - _X_START, 1.0, transform=ax.get_xaxis_transform()))
            for cursor in cursors:
                cursor.set_xdata([x, x])
            frames.append(fig_to_rgb(fig))
    finally:
        plt.close(fig)
    return frames


def make_animation(out_path: str | Path, *, cache_path: str | Path, device: str = "cpu") -> Path:
    payload = load_schedule_payload(cache_path, device=device)
    frames = render_frames(payload)
    durations = [FRAME_MS] * (len(frames) - 1) + [HOLD_MS]
    return save_gif(frames, durations, out_path)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Generate the schedule time-sweep GIF for the project page.")
    parser.add_argument("--cache", type=Path, required=True, help="Path to an epoch_dispatch_trace.npz replay cache.")
    parser.add_argument("--output", type=Path, required=True, help="Output GIF path.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    out = make_animation(args.output, cache_path=args.cache, device=args.device)
    print(f"Saved: {out}  ({out.stat().st_size:,} bytes)")
    return out


if __name__ == "__main__":
    main()
