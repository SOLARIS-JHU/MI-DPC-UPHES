from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import DPC.visualize.fig_schedule_evolution as fig_schedule_evolution
import DPC.visualize.fig_training_diagnostics as fig_training_diagnostics


def _sample_training_payload() -> dict[str, object]:
    epochs = np.arange(1, 25, dtype=int)
    return {
        "epochs": epochs,
        "grad_norm_batch": np.geomspace(8e4, 4e1, num=epochs.size),
        "grad_norm_step": np.geomspace(6e4, 3e1, num=epochs.size),
        "loss": np.linspace(4200.0, -900.0, num=epochs.size, dtype=float),
        "tau": np.linspace(10.0, 0.08, num=epochs.size, dtype=float),
    }


def test_build_figure_creates_compact_line_only_training_layout():
    fig = fig_training_diagnostics.build_figure(_sample_training_payload())

    try:
        assert np.allclose(
            fig.get_size_inches(),
            [
                fig_schedule_evolution.FULL_WIDTH * fig_schedule_evolution.FIGURE_WIDTH_SCALE,
                fig_schedule_evolution.BASE_FIGURE_HEIGHT * fig_schedule_evolution.FIGURE_HEIGHT_SCALE,
            ],
        )
        assert len(fig.axes) == 3

        grad_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Gradient norm")
        loss_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Loss")
        tau_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Gumbel-Softmax\ntempreture τ")

        assert fig._suptitle is None
        assert grad_ax.get_shared_x_axes().joined(grad_ax, loss_ax)
        assert loss_ax.get_xlabel() == "Epoch"
        assert loss_ax.yaxis.get_label().get_position()[0] > 0
        assert len(grad_ax.get_lines()) == 2
        assert len(loss_ax.get_lines()) == 1
        assert len(tau_ax.get_lines()) == 1
        assert len(grad_ax.collections) == 0
        assert len(loss_ax.collections) == 0
        assert len(tau_ax.collections) == 0
        legend = grad_ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["Batch", "Step"]
        assert legend._loc == 8
    finally:
        plt.close(fig)


def test_make_figure_writes_training_diagnostics_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fig_training_diagnostics,
        "load_training_payload",
        lambda cache_path: _sample_training_payload(),
    )

    out_path = fig_training_diagnostics.make_figure(tmp_path, cache_path=tmp_path / "cache.npz")

    assert out_path == tmp_path / "training_diagnostics.pdf"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_load_training_payload_reads_history_from_cache_referenced_run(tmp_path):
    cache_path = tmp_path / "cache.npz"
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "tau", "lr", "grad_norm"])
        writer.writerow([1, 3000.0, 10.0, 3e-4, 8000.0])
        writer.writerow([2, 1800.0, 5.0, 2e-4, 4000.0])
        writer.writerow([3, 900.0, 0.08, 1e-4, 1200.0])

    np.savez_compressed(
        cache_path,
        date=np.array("2024/08/09"),
        meta_json=np.array('{"run_dir": "%s"}' % str(run_dir).replace("\\", "\\\\")),
    )

    payload = fig_training_diagnostics.load_training_payload(cache_path)

    assert payload["epochs"].tolist() == [1, 2, 3]
    assert payload["loss"].tolist() == [3000.0, 1800.0, 900.0]
    assert payload["tau"].tolist() == [10.0, 5.0, 0.08]
    assert payload["grad_norm_batch"].tolist() == [8000.0, 4000.0, 1200.0]
    assert payload["grad_norm_step"].tolist() == [8000.0, 4000.0, 1200.0]


def test_load_training_payload_reads_retained_seed_mean_for_gradient_norm(tmp_path, monkeypatch):
    bench_dir = tmp_path / "bench"
    run_dir = bench_dir / "viz_epoch_mean_transformer_seed19_epch"
    run_dir.mkdir(parents=True)

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "tau", "lr", "grad_norm"])
        writer.writerow([1, 3000.0, 10.0, 3e-4, 8000.0])
        writer.writerow([2, 1800.0, 5.0, 2e-4, 4000.0])

    batch_a = bench_dir / "abl_dyn_batch_transformer_seed0"
    batch_b = bench_dir / "abl_dyn_batch_transformer_seed1"
    step_a = bench_dir / "abl_dyn_step_transformer_seed0_dynstep"
    step_b = bench_dir / "abl_dyn_step_transformer_seed1_dynstep"
    for path in (batch_a, batch_b, step_a, step_b):
        path.mkdir()

    with (batch_a / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "tau", "lr", "grad_norm"])
        writer.writerow([1, 3500.0, 10.0, 3e-4, 6000.0])
        writer.writerow([2, 2100.0, 5.0, 2e-4, 3000.0])

    with (batch_b / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "tau", "lr", "grad_norm"])
        writer.writerow([1, 3700.0, 10.0, 3e-4, 10000.0])
        writer.writerow([2, 2200.0, 5.0, 2e-4, 5000.0])

    with (step_a / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "tau", "lr", "grad_norm"])
        writer.writerow([1, 3200.0, 10.0, 3e-4, 4000.0])
        writer.writerow([2, 2000.0, 5.0, 2e-4, 2000.0])

    with (step_b / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "tau", "lr", "grad_norm"])
        writer.writerow([1, 3400.0, 10.0, 3e-4, 8000.0])
        writer.writerow([2, 1900.0, 5.0, 2e-4, 3000.0])

    (run_dir / "eval_results.json").write_text('{"config": {"seed": 19, "architecture": "transformer"}}', encoding="utf-8")
    (bench_dir / "ABLATION_47SEED_RUNS.csv").write_text(
        "\n".join(
            [
                "study,variant,seed,run_dir",
                "dynamics,batch,0,abl_dyn_batch_transformer_seed0",
                "dynamics,batch,1,abl_dyn_batch_transformer_seed1",
                "dynamics,step,0,abl_dyn_step_transformer_seed0_dynstep",
                "dynamics,step,1,abl_dyn_step_transformer_seed1_dynstep",
            ]
        ),
        encoding="utf-8",
    )

    cache_path = tmp_path / "cache.npz"
    np.savez_compressed(
        cache_path,
        date=np.array("2024/08/09"),
        meta_json=np.array('{"run_dir": "%s"}' % str(run_dir).replace("\\", "\\\\")),
    )

    payload = fig_training_diagnostics.load_training_payload(cache_path)

    assert payload["grad_norm_batch"].tolist() == [8000.0, 4000.0]
    assert payload["grad_norm_step"].tolist() == [6000.0, 2500.0]
