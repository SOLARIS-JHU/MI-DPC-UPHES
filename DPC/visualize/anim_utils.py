"""Shared helpers for the website animations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def fig_to_rgb(fig) -> Image.Image:
    """Rasterize a Matplotlib figure (Agg) to an RGB image at the figure's dpi."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba).convert("RGB")


def save_gif(frames: list[Image.Image], durations_ms: list[int], out_path: str | Path) -> Path:
    """Write a looping GIF. All frames share the palette of the last frame, which
    holds every colour at full strength, so colours do not flicker between frames."""
    if len(frames) != len(durations_ms):
        raise ValueError("frames and durations_ms must have the same length")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reference = frames[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    paletted = [f.quantize(palette=reference, dither=Image.Dither.NONE) for f in frames]
    paletted[0].save(
        out_path,
        save_all=True,
        append_images=paletted[1:],
        duration=durations_ms,
        loop=0,
        optimize=False,
        disposal=1,
    )
    return out_path
