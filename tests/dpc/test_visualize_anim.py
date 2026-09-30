from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from DPC.visualize import anim_schedule_evolution, anim_utils


def _sample_schedule_payload() -> dict[str, object]:
    hours = np.arange(24, dtype=int)
    return {
        "date": "2024/08/09",
        "hours": hours,
        "power": np.where(hours < 8, -35.0, np.where(hours < 18, 48.0, 0.0)),
        "price": np.linspace(40.0, 140.0, 24, dtype=float),
        "head": np.linspace(74.5, 77.5, 24, dtype=float),
        "volume": np.linspace(360000.0, 372000.0, 24, dtype=float),
        "target_volume": 370000.0,
        "feasible_power_bounds": {
            "pos_min": np.full(24, 15.0, dtype=float),
            "pos_max": np.full(24, 55.0, dtype=float),
            "neg_min": np.full(24, -52.0, dtype=float),
            "neg_max": np.full(24, -18.0, dtype=float),
        },
    }


def test_save_gif_writes_looping_gif_with_durations(tmp_path):
    # The palette comes from the last frame, so it must contain every colour used.
    frames = []
    for n_red in (0, 10, 20):
        frame = Image.new("RGB", (40, 20), "#ffffff")
        frame.paste("#0000ff", (0, 0, 5, 20))
        frame.paste("#ff0000", (20, 0, 20 + n_red, 20))
        frames.append(frame)
    out = anim_utils.save_gif(frames, [80, 80, 1500], tmp_path / "a.gif")
    with Image.open(out) as gif:
        assert gif.n_frames == 3
        assert gif.info["loop"] == 0
        gif.seek(2)
        assert gif.info["duration"] == 1500


def test_schedule_frames_reveal_forward_in_time():
    frames = anim_schedule_evolution.render_frames(_sample_schedule_payload(), steps_per_hour=1, dpi=60)
    assert len(frames) == 24
    arrays = [np.asarray(f, dtype=np.int16) for f in frames]
    assert all(a.shape == arrays[0].shape for a in arrays)
    # Each step changes the picture, and only to the right of the previous cursor.
    first_changed = []
    for prev, cur in zip(arrays, arrays[1:]):
        cols = np.flatnonzero(np.abs(cur - prev).sum(axis=(0, 2)))
        assert cols.size > 0
        first_changed.append(int(cols.min()))
    assert first_changed == sorted(first_changed)
    assert len(set(first_changed)) > 12


def _colourfulness(frame) -> int:
    a = np.asarray(frame, dtype=np.int16)
    return int((a.max(axis=2) - a.min(axis=2)).sum())


def test_revealed_trajectories_gain_colour_every_hour():
    frames = anim_schedule_evolution.render_frames(_sample_schedule_payload(), steps_per_hour=1, dpi=60)
    totals = [_colourfulness(f) for f in frames]
    assert all(b > a for a, b in zip(totals, totals[1:]))


def test_last_frame_matches_the_static_figure_trajectories():
    from DPC.visualize import fig_schedule_evolution
    from DPC.visualize.anim_utils import fig_to_rgb
    import matplotlib.pyplot as plt

    payload = _sample_schedule_payload()
    last = anim_schedule_evolution.render_frames(payload, steps_per_hour=1, dpi=60)[-1]
    fig = fig_schedule_evolution.build_figure(payload)
    try:
        fig.set_dpi(60)
        fig.patch.set_facecolor("#ffffff")
        static = fig_to_rgb(fig)
    finally:
        plt.close(fig)
    # Fully revealed, the animation is as colourful as the static figure (plus the cursor).
    assert _colourfulness(last) >= 0.97 * _colourfulness(static)


def test_gumbel_frames_anneal_temperature():
    from DPC.visualize import anim_gumbel_simplex

    frames, taus = anim_gumbel_simplex.render_frames(n_frames=6, dpi=50)
    assert len(frames) == len(taus) == 6
    assert taus[0] == 5.0 and abs(taus[-1] - 0.05) < 1e-9
    assert all(a > b for a, b in zip(taus, taus[1:]))
    assert frames[0].size == frames[-1].size


def test_save_gif_keeps_the_background_colour_exact(tmp_path):
    # The page frames animations on a card of exactly this colour; a palette
    # that rounds white to #fcfcfc shows as a visible grey rectangle.
    from DPC.visualize.fig_schedule_evolution import draw_schedule
    from DPC.visualize.anim_utils import fig_to_rgb
    import matplotlib.pyplot as plt

    fig, _, _ = draw_schedule(_sample_schedule_payload(), trajectory_alpha=0.12)
    try:
        fig.set_dpi(80)
        fig.patch.set_facecolor("#ffffff")
        frame = fig_to_rgb(fig)
    finally:
        plt.close(fig)
    assert frame.getpixel((0, 0)) == (255, 255, 255)
    out = anim_utils.save_gif([frame, frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT)], [80, 80], tmp_path / "bg.gif")
    with Image.open(out) as gif:
        for index in range(gif.n_frames):
            gif.seek(index)
            assert gif.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def _sample_training_run_payload(n_epochs: int = 4) -> dict[str, object]:
    hours = np.arange(24, dtype=float)
    final_power = np.where(hours < 8, -6.0, np.where(hours > 15, 6.0, 0.0))
    blend = np.linspace(0.0, 1.0, n_epochs)[:, None]
    power = blend * final_power[None, :] + (1.0 - blend) * 3.0
    head = 77.0 + blend * np.linspace(-4.0, 10.0, 24)[None, :]
    ones = np.ones((n_epochs, 24))
    return {
        "date": "2024/08/09",
        "hours": hours,
        "price": np.linspace(10.0, 110.0, 24),
        "epochs": np.arange(1, n_epochs + 1),
        "power": power,
        "head": head,
        "volume": 3.0e5 - 1.5e4 * (head - 77.0),
        "bounds": {"pos_min": 3.0 * ones, "pos_max": 7.5 * ones, "neg_min": -8.0 * ones, "neg_max": -5.0 * ones},
        "min_volume": 0.0,
        "max_volume": 588000.0,
        "target_volume": 3.0e5,
        "loss": np.linspace(2000.0, -2500.0, n_epochs),
        "tau": np.array([10.0, 10.0, 1.0, 0.08])[:n_epochs],
    }


def test_training_run_frames_cover_every_epoch_with_interpolation():
    from DPC.visualize import anim_training_run

    payload = _sample_training_run_payload()
    frames, positions = anim_training_run.render_frames(payload, steps_per_epoch=3, dpi=50)
    assert len(frames) == len(positions) == 3 * 3 + 1
    assert positions[0] == 1.0 and positions[-1] == 4.0
    assert all(b > a for a, b in zip(positions, positions[1:]))
    assert {1.0, 2.0, 3.0, 4.0} <= set(positions)
    assert len({f.size for f in frames}) == 1


def test_training_run_schedule_moves_toward_the_final_epoch():
    from DPC.visualize import anim_training_run

    payload = _sample_training_run_payload()
    frames, _ = anim_training_run.render_frames(payload, steps_per_epoch=1, dpi=50)
    arrays = [np.asarray(f, dtype=np.int16) for f in frames]
    distance_to_last = [int(np.abs(a - arrays[-1]).sum()) for a in arrays]
    assert distance_to_last[-1] == 0
    assert all(b < a for a, b in zip(distance_to_last, distance_to_last[1:]))


def test_training_run_rejects_mismatched_history():
    import pytest
    from DPC.visualize import anim_training_run

    payload = _sample_training_run_payload()
    payload["tau"] = payload["tau"][:-1]
    with pytest.raises(ValueError, match="one loss and tau value per epoch"):
        anim_training_run.render_frames(payload, steps_per_epoch=1, dpi=50)
