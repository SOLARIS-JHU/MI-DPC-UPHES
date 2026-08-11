from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import DPC.visualize.fig_schedule_evolution as fig_schedule_evolution


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


def test_build_figure_creates_two_panel_cdc_layout():
    fig = fig_schedule_evolution.build_figure(_sample_schedule_payload())

    try:
        assert np.allclose(fig.get_size_inches(), [7.0, 2.8])
        assert len(fig.axes) == 4

        power_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Power (MW)")
        price_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Price (€/MWh)")
        head_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Head (m)")
        volume_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Volume (m$^3$)")

        assert fig._suptitle is None
        assert power_ax.get_shared_x_axes().joined(power_ax, head_ax)
        assert power_ax.get_xlim() == (0.0, 23.0)
        assert head_ax.get_xlabel() == "Hour"
        assert 0 in power_ax.get_xticks()
        assert 20 in head_ax.get_xticks()

        assert len(power_ax.collections) == 2
        assert len(power_ax.get_lines()) >= 3
        assert len(price_ax.get_lines()) == 1
        assert len(head_ax.get_lines()) == 1
        assert len(volume_ax.get_lines()) == 2

        legend = volume_ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["Target volume"]
    finally:
        plt.close(fig)


def test_make_figure_writes_schedule_evolution_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fig_schedule_evolution,
        "load_schedule_payload",
        lambda cache_path, device="cpu": _sample_schedule_payload(),
    )

    out_path = fig_schedule_evolution.make_figure(tmp_path, cache_path=tmp_path / "cache.npz")

    assert out_path == tmp_path / "schedule_evolution.pdf"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_load_schedule_payload_uses_cached_final_trace_for_power_and_head(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.npz"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "eval_results.json").write_text('{"config": {"inverse_pkl_name": null}}', encoding="utf-8")

    np.savez_compressed(
        cache_path,
        date=np.array("2024/08/09"),
        epoch=np.array([1, 2], dtype=np.int64),
        epoch_files=np.array(["policy_epoch001.pt", "policy_epoch002.pt"], dtype=object),
        meta_json=np.array(
            '{"run_dir": "%s", "pkl_path": "preprocess.pkl", "benchmark_csv": "Data/price_data_2024.csv"}'
            % str(run_dir).replace("\\", "\\\\")
        ),
        price=np.linspace(50.0, 73.0, 24, dtype=np.float32),
        h=np.vstack([
            np.linspace(75.0, 78.0, 24, dtype=np.float32),
            np.linspace(74.0, 77.0, 24, dtype=np.float32),
        ]),
        p_exec=np.vstack([
            np.full(24, 1.0, dtype=np.float32),
            np.full(24, 3.0, dtype=np.float32),
        ]),
    )

    def fake_load_system_params(*args, **kwargs):
        return {
            "h_to_v_low": lambda h: h * 1000.0,
            "pos_min": lambda h: torch.full_like(h, 10.0),
            "pos_max": lambda h: torch.full_like(h, 20.0),
            "neg_min": lambda h: torch.full_like(h, -20.0),
            "neg_max": lambda h: torch.full_like(h, -10.0),
            "target_vol_low": 74000.0,
        }

    monkeypatch.setattr(fig_schedule_evolution, "load_system_params", fake_load_system_params)
    monkeypatch.setattr(
        fig_schedule_evolution,
        "load_benchmark_price_days",
        lambda path: {"2024/08/09": [float(i) for i in range(24)]},
    )
    monkeypatch.setattr(
        fig_schedule_evolution,
        "evaluate_day_oneshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not re-evaluate policy")),
    )

    payload = fig_schedule_evolution.load_schedule_payload(cache_path)

    assert payload["date"] == "2024/08/09"
    assert np.allclose(payload["power"], 3.0)
    assert np.allclose(payload["head"], np.linspace(74.0, 77.0, 24, dtype=float))
    assert np.allclose(payload["volume"], np.linspace(74000.0, 77000.0, 24, dtype=float))
    assert payload["target_volume"] == 74000.0
