from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import DPC.visualize.style as style
import DPC.visualize.epoch_replay as epoch_replay
import DPC.visualize.fig_dispatch_evolution as fig_dispatch_evolution


def _write_eval_results(run_dir: Path) -> None:
    payload = {
        "config": {
            "architecture": "transformer",
            "sampler": "cluster_balanced",
            "batch_size": 32,
            "optimizer": "adamw",
            "scheduler": "warmup_cosine",
            "epochs": 2,
            "seed": 7,
            "lr": 3e-4,
            "weight_decay": 5e-3,
            "grad_clip": 0.5,
            "tau_start": 5.0,
            "tau_end": 0.08,
            "tau_schedule": "two_stage",
            "tau_decay_ratio": 0.75,
            "ste_method": "gumbel",
            "num_train_samples": 8,
            "noise_std": 0.0,
            "min_price": 0.0,
            "sampler_clusters": 12,
            "shape_clusters": 8,
            "dev_fraction": 0.2,
            "eval_interval": 1,
            "vol_balance_mode": "squared_asymmetric",
            "vol_balance_scale": 6.2e-8,
            "vol_balance_weight": 2.0,
            "vol_deficit_factor": 1.0,
            "vol_surplus_factor": 1.5,
            "vol_weight_schedule": "late_ramp",
            "head_penalty": 50.0,
            "vol_traj_penalty": 50.0,
            "h_lb_scale": 1.0,
            "h_ub_scale": 1.0,
            "vol_lb_scale": 1.0,
            "vol_ub_scale": 1.0,
            "c_op": 0.4,
            "si_shortage_multiplier": -2.0,
            "si_surplus_multiplier": -0.5,
            "si_penalty_weight": 1.0,
            "target_vol_penalty_weight": 1.0,
            "hidden_size": 128,
            "num_layers": 3,
            "dropout": 0.2,
            "nhead": 4,
            "dim_ff": 256,
            "cnn_kernel_size": 3,
            "mlp_hidden_sizes": [512, 512, 512],
            "physics_mode": "nonlinear",
            "run_prefix": "test",
            "inverse_pkl_name": "preprocess_inverse_upc.pkl",
            "extra_field": "ignored",
        }
    }
    (run_dir / "eval_results.json").write_text(json.dumps(payload), encoding="utf-8")


def test_discover_epoch_checkpoints_sorts_numerically(tmp_path):
    for name in [
        "policy_epoch010.pt",
        "policy_epoch001.pt",
        "policy_epoch002.pt",
        "policy_epoch100.pt",
    ]:
        (tmp_path / name).write_bytes(b"")

    assert [path.name for path in epoch_replay.discover_epoch_checkpoints(tmp_path)] == [
        "policy_epoch001.pt",
        "policy_epoch002.pt",
        "policy_epoch010.pt",
        "policy_epoch100.pt",
    ]


def test_replay_epoch_dispatch_writes_npz_with_epoch_curves(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_eval_results(run_dir)

    torch.save({"weight": torch.tensor(1.0)}, run_dir / "policy_epoch001.pt")
    torch.save({"weight": torch.tensor(2.0)}, run_dir / "policy_epoch010.pt")

    seen = {}

    class FakeProblem:
        def __init__(self):
            self.device = "cpu"

        def to(self, device):
            seen["problem_to"] = device
            return self

        def parameters(self):
            return iter([torch.nn.Parameter(torch.tensor([0.0]))])

        def load_state_dict(self, state_dict):
            seen.setdefault("loaded_weights", []).append(float(state_dict["weight"]))

    class FakeDynamics:
        def __init__(self, params):
            seen["dynamics_params"] = dict(params)

    def fake_load_system_params(*args, **kwargs):
        seen["load_system_params"] = {"args": args, "kwargs": kwargs}
        return {"h_init": 1.0, "v_init": 2.0}

    def fake_load_benchmark_price_days(csv_path):
        seen["benchmark_csv"] = csv_path
        return {"2024/08/09": [float(i) for i in range(24)]}

    def fake_build_oneshot_architecture(**kwargs):
        seen["arch_kwargs"] = kwargs
        return torch.nn.Identity(), torch.nn.Identity()

    def fake_create_oneshot_ste(*args, **kwargs):
        seen["ste_args"] = {"args": args, "kwargs": kwargs}
        return torch.nn.Identity()

    def fake_build_oneshot_system(*args):
        seen["system_args"] = args
        return object()

    def fake_build_suite_loss(cfg, params):
        seen["loss_cfg"] = cfg
        seen["loss_params"] = params
        return object(), {"vol_balance_obj": type("W", (), {"weight": 0.0})()}

    def fake_build_problem(nodes, loss):
        seen["problem_nodes"] = nodes
        seen["problem_loss"] = loss
        return FakeProblem()

    def fake_evaluate_day_oneshot(problem, prices_24h, h_init, v_init, c_op, *, system_params=None, return_trace=False, **kwargs):
        epoch = int(seen["loaded_weights"][-1])
        trace = {
            "p_exec": np.full(24, epoch, dtype=float),
            "h": np.full(24, 70.0 + epoch, dtype=float),
        }
        return {
            "p_sim": np.full(24, epoch + 100.0, dtype=float),
            "trace": trace if return_trace else None,
        }

    monkeypatch.setattr(epoch_replay, "UPHESDynamicsBatch", FakeDynamics)
    monkeypatch.setattr(epoch_replay.benchmark_tuner, "build_oneshot_architecture", fake_build_oneshot_architecture)
    monkeypatch.setattr(epoch_replay.benchmark_tuner, "create_oneshot_ste", fake_create_oneshot_ste)
    monkeypatch.setattr(epoch_replay, "build_oneshot_system", fake_build_oneshot_system)
    monkeypatch.setattr(epoch_replay.benchmark_tuner, "build_suite_loss", fake_build_suite_loss)
    monkeypatch.setattr(epoch_replay.benchmark_tuner, "build_problem", fake_build_problem)
    monkeypatch.setattr(epoch_replay, "load_system_params", fake_load_system_params)
    monkeypatch.setattr(epoch_replay, "load_benchmark_price_days", fake_load_benchmark_price_days)
    monkeypatch.setattr(epoch_replay, "evaluate_day_oneshot", fake_evaluate_day_oneshot)

    out_path = epoch_replay.replay_epoch_dispatch(
        run_dir,
        "2024/08/09",
        tmp_path / "cache.npz",
    )

    assert out_path == tmp_path / "cache.npz"
    assert seen["benchmark_csv"] == "Data/price_data_2024.csv"
    assert seen["load_system_params"]["kwargs"]["inverse_pkl_path"] == "Data/UPCs/preprocess_inverse_upc.pkl"
    assert seen["loaded_weights"] == [1.0, 2.0]
    assert seen["dynamics_params"] == {"h_init": 1.0, "v_init": 2.0}
    assert str(seen["problem_to"]) == "cpu"

    payload = np.load(out_path, allow_pickle=True)
    assert set(payload.files) == {"date", "epoch", "epoch_files", "h", "meta_json", "p_exec", "price"}
    assert payload["date"].item() == "2024/08/09"
    assert payload["epoch"].tolist() == [1, 10]
    assert payload["p_exec"].shape == (2, 24)
    assert payload["h"].shape == (2, 24)
    assert np.all(payload["p_exec"][0] == 1.0)
    assert np.all(payload["p_exec"][1] == 2.0)
    assert np.all(payload["h"][0] == 71.0)
    assert np.all(payload["h"][1] == 72.0)
    assert json.loads(payload["meta_json"].item())["run_dir"] == str(run_dir)


def test_replay_epoch_dispatch_honors_explicit_inverse_override(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_eval_results(run_dir)

    torch.save({"weight": torch.tensor(1.0)}, run_dir / "policy_epoch001.pt")

    seen = {}

    class FakeProblem:
        def to(self, device):
            return self

        def parameters(self):
            return iter([torch.nn.Parameter(torch.tensor([0.0]))])

        def load_state_dict(self, state_dict):
            seen["loaded_weight"] = float(state_dict["weight"])

    def fake_load_system_params(*args, **kwargs):
        seen["inverse_pkl_path"] = kwargs.get("inverse_pkl_path")
        return {"h_init": 1.0, "v_init": 2.0}

    monkeypatch.setattr(epoch_replay, "UPHESDynamicsBatch", lambda params: object())
    monkeypatch.setattr(epoch_replay, "build_oneshot_system", lambda *args: object())
    monkeypatch.setattr(epoch_replay.benchmark_tuner, "build_oneshot_architecture", lambda **kwargs: (torch.nn.Identity(), torch.nn.Identity()))
    monkeypatch.setattr(epoch_replay.benchmark_tuner, "create_oneshot_ste", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(epoch_replay.benchmark_tuner, "build_suite_loss", lambda cfg, params: (object(), {"vol_balance_obj": type("W", (), {"weight": 0.0})()}))
    monkeypatch.setattr(epoch_replay.benchmark_tuner, "build_problem", lambda nodes, loss: FakeProblem())
    monkeypatch.setattr(epoch_replay, "load_system_params", fake_load_system_params)
    monkeypatch.setattr(epoch_replay, "load_benchmark_price_days", lambda path: {"2024/08/09": [0.0] * 24})
    monkeypatch.setattr(
        epoch_replay,
        "evaluate_day_oneshot",
        lambda *args, **kwargs: {"p_sim": np.zeros(24), "trace": {"p_exec": np.zeros(24), "h": np.full(24, 77.0)}},
    )

    epoch_replay.replay_epoch_dispatch(
        run_dir,
        "2024/08/09",
        tmp_path / "cache.npz",
        inverse_pkl="custom/override.pkl",
    )

    assert seen["inverse_pkl_path"] == "custom/override.pkl"


def test_epoch_dispatch_style_uses_plasma_r():
    assert style.EPOCH_CMAP.name == "plasma_r"


def _write_hourly_results_csv(path: Path, *, date: str, power: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "hour", "power", "price"])
        for hour, value in enumerate(power):
            writer.writerow([date, hour, value, 100.0 + hour])


def _write_history_csv(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "tau", "lr", "dev_expost"])
        for epoch, loss, tau, lr, value in [
            (1, 2100.0, 10.0, 3.0e-5, 300.0),
            (3, 2250.0, 9.5, 9.0e-5, 200.0),
            (10, 2490.0, 0.08, 1.5e-5, 100.0),
        ]:
            writer.writerow([epoch, loss, tau, lr, value])


def _write_epoch_dispatch_cache(path: Path, *, run_dir: Path | None = None) -> None:
    run_dir = run_dir or Path("/tmp/run")
    payload = {
        "date": np.array("2024/08/09"),
        "epoch": np.array([1, 3, 10], dtype=np.int64),
        "epoch_files": np.array(["policy_epoch001.pt", "policy_epoch003.pt", "policy_epoch010.pt"], dtype=object),
        "meta_json": np.array(json.dumps({"run_dir": str(run_dir)})),
        "price": np.linspace(50.0, 73.0, 24, dtype=np.float32),
        "h": np.vstack([
            np.linspace(75.0, 78.0, 24, dtype=np.float32),
            np.linspace(74.5, 77.5, 24, dtype=np.float32),
            np.linspace(74.0, 77.0, 24, dtype=np.float32),
        ]),
        "p_exec": np.vstack([
            np.full(24, 1.0, dtype=np.float32),
            np.full(24, 2.0, dtype=np.float32),
            np.full(24, 3.0, dtype=np.float32),
        ]),
    }
    np.savez_compressed(path, **payload)


def test_make_figures_writes_epoch_dispatch_pdfs_with_miqp_overlay(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.npz"
    run_dir = tmp_path / "run"
    _write_history_csv(run_dir)
    _write_epoch_dispatch_cache(cache_path, run_dir=run_dir)

    pw_csv = tmp_path / "MIQP" / "MIQP_piecewise" / "MIQP_piecewise_results.csv"
    gl_csv = tmp_path / "MIQP" / "MIQP_linear" / "MILP_global_linear_results.csv"
    _write_hourly_results_csv(pw_csv, date="2024/08/09", power=[10.0] * 24)
    _write_hourly_results_csv(gl_csv, date="2024/08/09", power=[20.0] * 24)

    save_calls = []
    original_savefig = plt.Figure.savefig

    def fake_savefig(self, fname, *args, **kwargs):
        save_calls.append(
            {
                "name": Path(fname).name,
                "axes": len(self.axes),
                "legend": self.axes[0].get_legend() is not None,
            }
        )
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(fig_dispatch_evolution, "MIQP_PW_RESULTS_CSV", pw_csv)
    monkeypatch.setattr(fig_dispatch_evolution, "MIQP_GL_RESULTS_CSV", gl_csv)
    monkeypatch.setattr(
        fig_dispatch_evolution,
        "load_feasible_power_bounds",
        lambda cache_arg, head_arg: {
            "pos_min": np.full(24, 3.0, dtype=float),
            "pos_max": np.full(24, 5.0, dtype=float),
            "neg_min": np.full(24, -5.0, dtype=float),
            "neg_max": np.full(24, -3.0, dtype=float),
        },
    )
    monkeypatch.setattr(fig_dispatch_evolution, "load_target_head", lambda cache_arg: 77.0)
    monkeypatch.setattr(plt.Figure, "savefig", fake_savefig)

    plain_path, overlay_path = fig_dispatch_evolution.make_figures(cache_path, tmp_path)

    assert plain_path == tmp_path / "epoch_dispatch_colormap.pdf"
    assert overlay_path == tmp_path / "epoch_dispatch_colormap_with_miqp.pdf"
    assert plain_path.exists()
    assert overlay_path.exists()
    assert plain_path.stat().st_size > 0
    assert overlay_path.stat().st_size > 0
    assert save_calls == [
        {"name": "epoch_dispatch_colormap.pdf", "axes": 6, "legend": True},
        {"name": "epoch_dispatch_colormap_with_miqp.pdf", "axes": 7, "legend": True},
    ]


def test_load_epoch_cache_rejects_mismatched_epoch_and_trace_counts(tmp_path):
    cache_path = tmp_path / "bad_cache.npz"
    np.savez_compressed(
        cache_path,
        date=np.array("2024/08/09"),
        epoch=np.array([1, 2], dtype=np.int64),
        epoch_files=np.array(["policy_epoch001.pt", "policy_epoch002.pt"], dtype=object),
        meta_json=np.array(json.dumps({"run_dir": "/tmp/run"})),
        price=np.linspace(50.0, 73.0, 24, dtype=np.float32),
        h=np.full((1, 24), 77.0, dtype=np.float32),
        p_exec=np.full((1, 24), 1.0, dtype=np.float32),
    )

    try:
        fig_dispatch_evolution.load_epoch_cache(cache_path)
    except ValueError as exc:
        assert "epoch" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for mismatched replay cache")


def test_load_hourly_power_schedule_rejects_duplicate_or_missing_hours(tmp_path):
    csv_path = tmp_path / "bad_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "hour", "power", "price"])
        for hour in range(23):
            writer.writerow(["2024/08/09", hour, float(hour), 100.0 + hour])
        writer.writerow(["2024/08/09", 0, 999.0, 200.0])

    try:
        fig_dispatch_evolution.load_hourly_power_schedule(csv_path, "2024/08/09")
    except ValueError as exc:
        assert "24" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid hourly schedule")


def test_build_figure_includes_colorbar_and_miqp_overlay_lines():
    fig = fig_dispatch_evolution._build_figure(
        date="2024/08/09",
        epochs=np.array([1, 10], dtype=np.int64),
        power=np.vstack([
            np.full(24, 1.0, dtype=float),
            np.full(24, 2.0, dtype=float),
        ]),
        price=np.linspace(50.0, 73.0, 24, dtype=float),
        head=np.vstack([
            np.linspace(75.0, 78.0, 24, dtype=float),
            np.linspace(74.0, 77.0, 24, dtype=float),
        ]),
        history_epochs=np.array([1, 10], dtype=np.int64),
        losses=np.array([2200.0, 2490.0], dtype=float),
        tau=np.array([10.0, 0.08], dtype=float),
        lr=np.array([3.0e-5, 1.5e-5], dtype=float),
        miqp_pw=np.full(24, 10.0, dtype=float),
        miqp_gl=np.full(24, 20.0, dtype=float),
        show_title=False,
    )

    try:
        assert len(fig.axes) == 7
        main_ax = fig.axes[0]
        price_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Price (€/MWh)")
        assert len(main_ax.get_lines()) == 4
        epoch_line_early, epoch_line_final, miqp_pw_line, miqp_gl_line = main_ax.get_lines()
        assert epoch_line_final.get_color() == "black"
        assert epoch_line_early.get_color() != "black"
        assert miqp_pw_line.get_color() == fig_dispatch_evolution.MIQP_PW_OVERLAY_COLOR
        assert miqp_pw_line.get_linewidth() > epoch_line_final.get_linewidth()
        assert miqp_gl_line.get_linewidth() > epoch_line_final.get_linewidth()
        assert miqp_pw_line.get_zorder() > epoch_line_final.get_zorder()
        assert miqp_gl_line.get_zorder() > epoch_line_final.get_zorder()
        legend = main_ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["MIQP-PW", "MIQP-GL"]
        assert legend._loc == 9
        assert main_ax.get_title() == ""
        assert fig._suptitle is None
        ylabels = {ax.get_ylabel() for ax in fig.axes}
        assert {"Power (MW)", "Price (€/MWh)", "Head (m)", "Epoch index", "Loss", "Gumbel-Softmax\ntempreture τ", "Learning rate"} <= ylabels
        assert main_ax.get_xlabel() == ""
        assert price_ax.get_xlabel() == ""
        assert [text.get_text() for text in fig.texts].count("Hour") == 1
        assert abs(main_ax.get_position().x0 - price_ax.get_position().x0) < 1e-6
        assert abs(main_ax.get_position().x1 - price_ax.get_position().x1) < 1e-6
        assert len(fig.lines) == 0
        assert any(line.get_visible() for line in main_ax.get_xgridlines())
        assert any(line.get_visible() for line in price_ax.get_xgridlines())
        head_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Head (m)")
        assert head_ax.get_lines()[0].get_color() == fig_dispatch_evolution.HEAD_COLOR
        train_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Loss")
        top_mid_gap = main_ax.get_position().y0 - price_ax.get_position().y1
        mid_bot_gap = price_ax.get_position().y0 - train_ax.get_position().y1
        assert top_mid_gap < 0.05
        assert mid_bot_gap > 0.07
        assert train_ax.get_xlabel() == "Epoch"
        tau_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Gumbel-Softmax\ntempreture τ")
        lr_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Learning rate")
        assert len(train_ax.get_lines()) == 1
        assert len(tau_ax.get_lines()) == 1
        assert len(lr_ax.get_lines()) == 1
        assert tau_ax.yaxis.labelpad <= 2.5
        assert lr_ax.spines["right"].get_position()[1] <= 1.15
        assert lr_ax.yaxis.labelpad <= 6.5
        fig.canvas.draw()
        assert any("e" in tick.get_text() for tick in lr_ax.get_yticklabels() if tick.get_text())
        colorbar_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Epoch index")
        assert colorbar_ax.get_ylabel() == "Epoch index"
        assert colorbar_ax.get_yticks().tolist() == [1, 5, 10]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_build_plain_figure_adds_initialization_and_final_epoch_labels():
    fig = fig_dispatch_evolution._build_figure(
        date="2024/08/09",
        epochs=np.array([1, 10], dtype=np.int64),
        power=np.vstack([
            np.full(24, 1.0, dtype=float),
            np.full(24, 2.0, dtype=float),
        ]),
        price=np.linspace(50.0, 73.0, 24, dtype=float),
        head=np.vstack([
            np.linspace(75.0, 78.0, 24, dtype=float),
            np.linspace(74.0, 77.0, 24, dtype=float),
        ]),
        history_epochs=np.array([1, 10], dtype=np.int64),
        losses=np.array([2200.0, 2490.0], dtype=float),
        tau=np.array([10.0, 0.08], dtype=float),
        lr=np.array([3.0e-5, 1.5e-5], dtype=float),
        include_lr=False,
        plain_mode=True,
        feasible_power_bounds={
            "pos_min": np.full(24, 3.0, dtype=float),
            "pos_max": np.full(24, 5.0, dtype=float),
            "neg_min": np.full(24, -5.0, dtype=float),
            "neg_max": np.full(24, -3.0, dtype=float),
        },
        target_head=77.0,
        show_title=False,
    )

    try:
        main_ax = fig.axes[0]
        assert main_ax.get_ylabel() == "Head (m)"
        ylabels = {ax.get_ylabel() for ax in fig.axes}
        assert "Learning rate" not in ylabels
        assert {"Head (m)", "Price (€/MWh)", "Power (MW)", "Epoch index", "Loss", "Gumbel-Softmax\ntempreture τ"} <= ylabels
        legend = main_ax.get_legend()
        assert legend is not None
        assert legend._loc == 2
        assert [text.get_text() for text in legend.get_texts()] == ["Final epoch", "Target head"]
        assert legend.get_texts()[0].get_color() == "black"
        assert legend.get_texts()[1].get_color() == fig_dispatch_evolution.TARGET_HEAD_COLOR
        assert len(main_ax.get_lines()) == 3
        head_line_early, head_line_final, target_line = main_ax.get_lines()
        assert head_line_final.get_color() == "black"
        assert head_line_early.get_color() != "black"
        assert head_line_early.get_alpha() == 0.37
        assert np.allclose(target_line.get_ydata(), 77.0)
        assert target_line.get_color() == fig_dispatch_evolution.TARGET_HEAD_COLOR
        assert target_line.get_linestyle() != "-"
        power_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Power (MW)")
        price_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Price (€/MWh)")
        assert price_ax.get_zorder() < power_ax.get_zorder()
        assert power_ax.get_lines()[0].get_color() == fig_dispatch_evolution.PLAIN_POWER_COLOR
        assert power_ax.get_lines()[0].get_drawstyle() == "steps-mid"
        zero_lines = [line for line in power_ax.get_lines()[1:] if np.allclose(line.get_ydata(), 0.0)]
        assert len(zero_lines) == 1
        assert zero_lines[0].get_color() == fig_dispatch_evolution.PLAIN_POWER_COLOR
        assert zero_lines[0].get_alpha() == 0.4
        assert len(power_ax.collections) == 2
        for coll in power_ax.collections:
            rgba = tuple(np.round(coll.get_facecolor()[0], 3))
            assert rgba[:3] == tuple(np.round(plt.matplotlib.colors.to_rgba(fig_dispatch_evolution.PLAIN_POS_REGION_COLOR)[:3], 3))
        assert price_ax.get_lines()[0].get_color() == fig_dispatch_evolution.PLAIN_PRICE_COLOR
        assert price_ax.get_lines()[0].get_drawstyle() == "steps-mid"
        train_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Loss")
        assert len(train_ax.collections) == 2
        assert train_ax.collections[0].get_cmap().name == "viridis_r"
        assert np.allclose(train_ax.collections[0].get_sizes(), 7.0)
        colorbar_ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Epoch index")
        assert colorbar_ax.get_yticks().tolist() == [1, 5, 10]
        assert colorbar_ax._colorbar.mappable.cmap.name == "viridis_r"
    finally:
        plt.close(fig)


def test_load_training_curve_uses_cache_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    _write_history_csv(run_dir)
    cache_path = tmp_path / "cache.npz"
    _write_epoch_dispatch_cache(cache_path, run_dir=run_dir)

    epochs, loss, tau, lr = fig_dispatch_evolution.load_training_curve(cache_path)

    assert epochs.tolist() == [1, 3, 10]
    assert loss.tolist() == [2100.0, 2250.0, 2490.0]
    assert tau.tolist() == [10.0, 9.5, 0.08]
    assert lr.tolist() == [3.0e-5, 9.0e-5, 1.5e-5]


def test_load_training_curve_reads_loss_even_if_dev_expost_is_blank(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "tau", "lr", "dev_expost"])
        writer.writerow([1, 10.0, 5.0, 3.0e-5, ""])
        writer.writerow([2, 8.0, 4.0, 4.0e-5, 2200.0])
        writer.writerow([3, 6.0, 3.0, 5.0e-5, ""])
        writer.writerow([4, 4.0, 2.0, 6.0e-5, 2400.0])
    cache_path = tmp_path / "cache.npz"
    _write_epoch_dispatch_cache(cache_path, run_dir=run_dir)

    epochs, losses, tau, lr = fig_dispatch_evolution.load_training_curve(cache_path)

    assert epochs.tolist() == [1, 2, 3, 4]
    assert losses.tolist() == [10.0, 8.0, 6.0, 4.0]
    assert tau.tolist() == [5.0, 4.0, 3.0, 2.0]
    assert lr.tolist() == [3.0e-5, 4.0e-5, 5.0e-5, 6.0e-5]


def test_main_parses_cli_and_disables_miqp(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.npz"
    output_dir = tmp_path / "figs"
    seen = {}

    def fake_make_figures(cache_arg, output_arg, *, include_miqp):
        seen["cache"] = cache_arg
        seen["output"] = output_arg
        seen["include_miqp"] = include_miqp
        return output_arg / "epoch_dispatch_colormap.pdf", output_arg / "epoch_dispatch_colormap_with_miqp.pdf"

    monkeypatch.setattr(fig_dispatch_evolution, "make_figures", fake_make_figures)

    result = fig_dispatch_evolution.main(
        [
            "--cache",
            str(cache_path),
            "--output-dir",
            str(output_dir),
            "--no-miqp",
        ]
    )

    assert result == (output_dir / "epoch_dispatch_colormap.pdf", output_dir / "epoch_dispatch_colormap_with_miqp.pdf")
    assert seen["cache"] == cache_path
    assert seen["output"] == output_dir
    assert seen["include_miqp"] is False


def test_epoch_replay_main_parses_cli_and_delegates(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    out_path = tmp_path / "cache.npz"
    seen = {}

    def fake_replay_epoch_dispatch(run_dir_arg, date_arg, out_path_arg, *, device, pkl_path, benchmark_csv, inverse_pkl):
        seen["run_dir"] = run_dir_arg
        seen["date"] = date_arg
        seen["out_path"] = out_path_arg
        seen["device"] = device
        seen["pkl_path"] = pkl_path
        seen["benchmark_csv"] = benchmark_csv
        seen["inverse_pkl"] = inverse_pkl
        return out_path_arg

    monkeypatch.setattr(epoch_replay, "replay_epoch_dispatch", fake_replay_epoch_dispatch)

    result = epoch_replay.main(
        [
            "--run-dir",
            str(run_dir),
            "--date",
            "2024/08/09",
            "--output",
            str(out_path),
            "--device",
            "cpu",
            "--pkl",
            "preprocess.pkl",
            "--benchmark-csv",
            "Data/price_data_2024.csv",
            "--inverse-pkl",
            "custom_inverse.pkl",
        ]
    )

    assert result == out_path
    assert seen["run_dir"] == run_dir
    assert seen["date"] == "2024/08/09"
    assert seen["out_path"] == out_path
    assert seen["device"] == "cpu"
    assert seen["pkl_path"] == "preprocess.pkl"
    assert seen["benchmark_csv"] == "Data/price_data_2024.csv"
    assert seen["inverse_pkl"] == "custom_inverse.pkl"
