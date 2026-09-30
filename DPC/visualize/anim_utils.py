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


def _to_palette(frame: Image.Image, palette: np.ndarray) -> Image.Image:
    """Map an RGB frame onto palette (K×3) by exact nearest colour. Pillow's own
    palette mapping is approximate and can miss an exact match."""
    pixels = np.asarray(frame).reshape(-1, 3)
    colours, inverse = np.unique(pixels, axis=0, return_inverse=True)
    nearest = np.empty(len(colours), dtype=np.uint8)
    for start in range(0, len(colours), 4096):
        chunk = colours[start : start + 4096].astype(np.int32)
        distance = ((chunk[:, None, :] - palette[None, :, :].astype(np.int32)) ** 2).sum(axis=2)
        nearest[start : start + 4096] = distance.argmin(axis=1)
    out = Image.fromarray(nearest[inverse.reshape(-1)].reshape(frame.height, frame.width), mode="P")
    out.putpalette(palette.astype(np.uint8).tobytes())
    return out


def save_gif(frames: list[Image.Image], durations_ms: list[int], out_path: str | Path) -> Path:
    """Write a looping GIF. All frames share the palette of the last frame, which
    holds every colour at full strength, so colours do not flicker between frames."""
    if len(frames) != len(durations_ms):
        raise ValueError("frames and durations_ms must have the same length")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reference = frames[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    palette = np.asarray(reference.getpalette(), dtype=np.uint8).reshape(-1, 3)
    # Median cut averages the background with its anti-aliased neighbours; pin
    # that entry back to the exact corner colour so the GIF blends into the
    # page element it sits on.
    palette[reference.getpixel((0, 0))] = frames[-1].getpixel((0, 0))
    paletted = [_to_palette(f, palette) for f in frames]
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
