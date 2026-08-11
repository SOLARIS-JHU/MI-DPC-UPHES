from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from DPC.experiments.generate_ablation_commands import build_ablation_command_sets
import DPC.experiments.benchmark_tuner as benchmark_tuner
from DPC.experiments.benchmark_tuner import (
    ExperimentConfig,
    build_configs,
    build_dynamics,
    build_suite_loss,
    evaluate_price_dict,
)
from DPC.dynamics import UPHESDynamicsBatch, UPHESDynamicsStep


def _make_args(**overrides):
    defaults = {
        "architectures": "transformer",
        "samplers": "cluster_balanced",
        "batch_sizes": "32",
        "optimizers": "adamw",
        "schedulers": "warmup_cosine",
        "tau_schedules": "two_stage",
        "tau_ends": "0.08",
        "vol_balance_modes": "squared_asymmetric",
        "vol_balance_weights": "2.0",
        "vol_surplus_factors": "1.5",
        "head_penalties": "50",
        "vol_traj_penalties": "50",
        "h_lb_scales": "1.0",
        "h_ub_scales": "1.0",
        "vol_lb_scales": "1.0",
        "vol_ub_scales": "1.0",
        "seeds": "0",
        "epochs": 25,
        "lr": 3e-4,
        "weight_decay": 5e-3,
        "grad_clip": 0.5,
        "tau_start": 5.0,
        "tau_decay_ratio": 0.75,
        "c_op": 1.0,
        "si_shortage_multiplier": -2.0,
        "si_surplus_multiplier": -0.5,
        "si_penalty_weight": 1.0,
        "target_vol_penalty_weight": 1.0,
        "ste": "gumbel",
        "num_train_samples": 128,
        "noise_std": 0.0,
        "min_price": 0.0,
        "sampler_clusters": 12,
        "shape_clusters": 8,
        "dev_fraction": 0.2,
        "eval_interval": 10,
        "vol_balance_scale": 6.2e-8,
        "vol_deficit_factor": 1.0,
        "vol_weight_schedule": "late_ramp",
        "hidden_size": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "nhead": 4,
        "dim_ff": 256,
        "cnn_kernel_size": 3,
        "mlp_hidden_sizes": "512,512,512",
        "physics": "nonlinear",
        "run_prefix": "",
        "dynamics": "batch",
        "inverse_pkl": None,
        "pkl": "preprocess.pkl",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _identity(h):
    return h


def _zeros(a, b):
    return torch.zeros_like(a)


def _constant(h):
    return torch.ones_like(h)


def _fake_system_params(include_inverse=False):
    params = {
        "pos_min": _constant,
        "pos_max": _constant,
        "neg_min": _constant,
        "neg_max": _constant,
        "UPC_poly_tur": _zeros,
        "UPC_poly_pump": _zeros,
        "v_low_to_h": _identity,
        "head_min": 0.0,
        "head_max": 10.0,
        "max_vol_low": 10.0,
        "h_init": 1.0,
        "v_init": 1.0,
        "target_vol_low": 10.0,
        "target_head": 1.0,
    }
    if include_inverse:
        params["UPC_inv_tur"] = _zeros
        params["UPC_inv_pump"] = _zeros
    return params


def _exact_system_params():
    def pos_min(h):
        return torch.zeros_like(h)

    def pos_max(h):
        return torch.full_like(h, 100.0)

    def neg_min(h):
        return torch.full_like(h, -100.0)

    def neg_max(h):
        return torch.zeros_like(h)

    def upc(p, h):
        del h
        return p / 3600.0

    return {
        "pos_min": pos_min,
        "pos_max": pos_max,
        "neg_min": neg_min,
        "neg_max": neg_max,
        "UPC_poly_tur": upc,
        "UPC_poly_pump": upc,
        "v_low_to_h": _identity,
        "head_min": 0.0,
        "head_max": 1e6,
        "max_vol_low": 1e9,
        "h_init": 1.0,
        "v_init": 1.0,
        "target_vol_low": 1e9,
        "target_head": 1.0,
    }


class _FakeProblem:
    def __init__(self, x, aux, u):
        self._x = x
        self._aux = aux
        self._u = u
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def __call__(self, data):
        name = data["name"]
        return {
            f"{name}_x": self._x,
            f"{name}_aux": self._aux,
            f"{name}_u": self._u,
        }


def test_build_configs_defaults_to_batch_metadata_and_name():
    cfg = build_configs(_make_args())[0]

    assert cfg.dynamics_name == "batch"
    assert cfg.inverse_pkl_path is None
    assert "dyn" not in cfg.run_name()
    assert cfg.output_metadata()["dynamics_name"] == "batch"
    assert cfg.output_metadata()["inverse_pkl_name"] is None


def test_build_configs_rejects_step_dynamics_with_linear_physics():
    with pytest.raises(ValueError, match="Step dynamics requires physics='nonlinear'"):
        build_configs(_make_args(physics="linear", dynamics="step"))


def test_build_configs_includes_step_metadata_in_stable_outputs():
    cfg = build_configs(
        _make_args(dynamics="step", inverse_pkl="artifacts/inverse_surface.pkl", run_prefix="rollout")
    )[0]

    assert cfg.dynamics_name == "step"
    assert cfg.inverse_pkl_path == "artifacts/inverse_surface.pkl"
    assert "dynstep" in cfg.run_name()
    assert "invinverse-surface" in cfg.run_name()
    assert cfg.output_metadata()["inverse_pkl_name"] == "inverse_surface.pkl"


def test_output_metadata_omits_inverse_pkl_path():
    cfg = ExperimentConfig(
        architecture="transformer",
        sampler="cluster_balanced",
        batch_size=32,
        dynamics_name="step",
        inverse_pkl_path="artifacts/inverse_surface.pkl",
    )

    metadata = cfg.output_metadata()

    assert metadata["inverse_pkl_name"] == "inverse_surface.pkl"
    assert "inverse_pkl_path" not in metadata


def test_build_configs_accepts_step_specific_objective_grid():
    cfgs = build_configs(
        _make_args(
            dynamics="step",
            inverse_pkl="artifacts/inverse_surface.pkl",
            target_vol_penalty_weights="1.0,1.5,2.0,3.0",
            lrs="2e-4,3e-4",
            grad_clips="0.25,0.5",
            weight_decays="0.0,5e-3",
            output_root="DPC/outputs/benchmark_suite/step_sweep",
        )
    )

    assert {cfg.target_vol_penalty_weight for cfg in cfgs} == {1.0, 1.5, 2.0, 3.0}
    assert {cfg.lr for cfg in cfgs} == {2e-4, 3e-4}
    assert {cfg.grad_clip for cfg in cfgs} == {0.25, 0.5}
    assert {cfg.weight_decay for cfg in cfgs} == {0.0, 5e-3}


def test_build_dynamics_uses_batch_by_default():
    cfg = ExperimentConfig(architecture="transformer", sampler="cluster_balanced", batch_size=32)

    dynamics = build_dynamics(cfg, _fake_system_params())

    assert isinstance(dynamics, UPHESDynamicsBatch)


def test_build_dynamics_uses_step_when_requested():
    cfg = ExperimentConfig(
        architecture="transformer",
        sampler="cluster_balanced",
        batch_size=32,
        dynamics_name="step",
        inverse_pkl_path="artifacts/inverse_surface.pkl",
    )

    dynamics = build_dynamics(cfg, _fake_system_params(include_inverse=True))

    assert isinstance(dynamics, UPHESDynamicsStep)


def test_build_dynamics_step_requires_inverse_upc_params():
    cfg = ExperimentConfig(
        architecture="transformer",
        sampler="cluster_balanced",
        batch_size=32,
        dynamics_name="step",
        inverse_pkl_path=None,
    )

    with pytest.raises(ValueError, match="Step dynamics requires inverse UPC params"):
        build_dynamics(cfg, _fake_system_params(include_inverse=False))


def test_build_suite_loss_keeps_batch_fallback_for_8_channel_aux():
    cfg = ExperimentConfig(
        architecture="transformer",
        sampler="cluster_balanced",
        batch_size=1,
        c_op=0.0,
        si_penalty_weight=1.0,
        target_vol_penalty_weight=0.0,
    )
    loss, _ = build_suite_loss(cfg, {"target_head": 1.0, "target_vol_low": 0.0})
    profit_obj = next(obj for obj in loss.objectives if obj.name == "profit_loss")
    x = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    aux = torch.tensor([[[10.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    d = torch.tensor([[[5.0]]], dtype=torch.float32)

    loss_value = profit_obj.loss(x, aux, d)

    assert torch.isclose(loss_value, torch.tensor(100.0))


def test_build_suite_loss_excludes_vol_balance_from_active_objectives():
    cfg = ExperimentConfig(
        architecture="transformer",
        sampler="cluster_balanced",
        batch_size=1,
        vol_balance_weight=2.0,
        c_op=0.0,
        si_penalty_weight=1.0,
        target_vol_penalty_weight=0.0,
    )

    loss, metadata = build_suite_loss(cfg, {"target_head": 1.0, "target_vol_low": 0.0})

    assert [obj.name for obj in loss.objectives] == [
        "profit_loss",
        "vol_lb",
        "vol_ub",
        "h_lb",
        "h_ub",
    ]
    assert "vol_balance_obj" in metadata
    assert metadata["vol_balance_obj"].name == "vol_balance"
    metadata["vol_balance_obj"].weight = 0.0
    assert metadata["vol_balance_obj"].weight == 0.0
    assert [obj.name for obj in loss.objectives] == [
        "profit_loss",
        "vol_lb",
        "vol_ub",
        "h_lb",
        "h_ub",
    ]


def test_build_suite_loss_uses_executed_power_when_step_aux_is_available():
    cfg = ExperimentConfig(
        architecture="transformer",
        sampler="cluster_balanced",
        batch_size=1,
        c_op=0.0,
        si_penalty_weight=1.0,
        target_vol_penalty_weight=0.0,
        dynamics_name="step",
        inverse_pkl_path="artifacts/inverse_surface.pkl",
    )
    loss, _ = build_suite_loss(cfg, {"target_head": 1.0, "target_vol_low": 0.0})
    profit_obj = next(obj for obj in loss.objectives if obj.name == "profit_loss")
    x = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    aux = torch.tensor(
        [[[10.0, 0.0, 6.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    d = torch.tensor([[[5.0]]], dtype=torch.float32)

    loss_value = profit_obj.loss(x, aux, d)

    assert torch.isclose(loss_value, torch.tensor(10.0))


def test_evaluate_price_dict_uses_executed_power_for_step_aux():
    x = torch.tensor(
        [[[1.0, 1.0]]] + [[[2.0, 2.0]] for _ in range(24)],
        dtype=torch.float32,
    ).reshape(1, 25, 2)
    aux = torch.zeros((1, 24, 12), dtype=torch.float32)
    aux[..., 0] = 10.0
    aux[..., 2] = 6.0
    u = torch.zeros((1, 24, 3), dtype=torch.float32)
    problem = _FakeProblem(x, aux, u)
    price_data = {"2024-01-01": [2.0] * 24}
    system_params = _exact_system_params()

    results, summary = evaluate_price_dict(problem, price_data, system_params, c_op=1.0)

    day = results["2024-01-01"]
    assert np.allclose(day["p_net"], np.full(24, 10.0))
    assert np.allclose(day["p_sim"], np.full(24, 10.0))
    assert day["revenue"] == 480.0
    assert day["op_cost"] == 2400.0
    assert day["profit"] == -1920.0
    assert summary["mean_profit"] == -1920.0


def test_evaluate_price_dict_keeps_batch_fallback_for_8_channel_aux():
    x = torch.tensor(
        [[[1.0, 1.0]]] + [[[2.0, 2.0]] for _ in range(24)],
        dtype=torch.float32,
    ).reshape(1, 25, 2)
    aux = torch.zeros((1, 24, 8), dtype=torch.float32)
    aux[..., 0] = 6.0
    aux[..., 1] = 0.0
    aux[..., 4:8] = 1.0
    u = torch.zeros((1, 24, 3), dtype=torch.float32)
    problem = _FakeProblem(x, aux, u)
    price_data = {"2024-01-01": [2.0] * 24}
    system_params = _exact_system_params()

    results, summary = evaluate_price_dict(problem, price_data, system_params, c_op=1.0)

    day = results["2024-01-01"]
    assert np.allclose(day["p_net"], np.full(24, 6.0))
    assert np.allclose(day["p_sim"], np.full(24, 6.0))
    assert day["revenue"] == 288.0
    assert day["op_cost"] == 864.0
    assert day["profit"] == -576.0
    assert summary["mean_profit"] == -576.0


def test_main_forwards_inverse_pkl_path_for_step_dynamics(monkeypatch):
    args = _make_args(dynamics="step", inverse_pkl="artifacts/inverse_surface.pkl")

    class _StopAfterForwarding(RuntimeError):
        pass

    def fake_parse_args(self):
        return args

    def fake_load_system_params(pkl_path, device=None, physics_mode="nonlinear", inverse_pkl_path=None):
        assert pkl_path == "preprocess.pkl"
        assert inverse_pkl_path == "artifacts/inverse_surface.pkl"
        raise _StopAfterForwarding()

    monkeypatch.setattr(benchmark_tuner.argparse.ArgumentParser, "parse_args", fake_parse_args)
    monkeypatch.setattr(benchmark_tuner, "load_system_params", fake_load_system_params)
    monkeypatch.setattr(benchmark_tuner, "load_reference_metrics", lambda: {})
    monkeypatch.setattr(benchmark_tuner.os, "makedirs", lambda *args, **kwargs: None)

    with pytest.raises(_StopAfterForwarding):
        benchmark_tuner.main()


def test_main_uses_output_root_for_step_sweep_artifacts(monkeypatch, tmp_path):
    args = _make_args(
        dynamics="step",
        inverse_pkl="artifacts/inverse_surface.pkl",
    )
    args.output_root = str(tmp_path / "step_sweep_outputs")

    recorded = {}

    def fake_parse_args(self):
        return args

    def fake_load_system_params(pkl_path, device=None, physics_mode="nonlinear", inverse_pkl_path=None):
        assert pkl_path == "preprocess.pkl"
        assert inverse_pkl_path == "artifacts/inverse_surface.pkl"
        return _fake_system_params(include_inverse=True)

    def fake_run_single_experiment(cfg, parsed_args, refs):
        return {"run_name": cfg.run_name(), "dev_expost": 0.0}

    def fake_makedirs(path, exist_ok=False):
        recorded["makedirs"] = path

    def fake_write_sweep_summary(rows, out_dir):
        recorded["summary"] = out_dir

    monkeypatch.setattr(benchmark_tuner.argparse.ArgumentParser, "parse_args", fake_parse_args)
    monkeypatch.setattr(benchmark_tuner, "load_system_params", fake_load_system_params)
    monkeypatch.setattr(benchmark_tuner, "load_reference_metrics", lambda: {})
    monkeypatch.setattr(benchmark_tuner, "run_single_experiment", fake_run_single_experiment)
    monkeypatch.setattr(benchmark_tuner.os, "makedirs", fake_makedirs)
    monkeypatch.setattr(benchmark_tuner, "write_sweep_summary", fake_write_sweep_summary)

    benchmark_tuner.main()

    assert recorded["makedirs"] == args.output_root
    assert recorded["summary"] == args.output_root


def test_rerun_saved_benchmark_config_reuses_saved_eval_config(tmp_path):
    source_run_dir = tmp_path / "source_run"
    source_run_dir.mkdir()

    from DPC.experiments.rerun_saved_benchmark_config import rerun_from_eval_config

    rerun_from_eval_config(source_run_dir, new_prefix="stepfix_final", output_root=tmp_path)

    assert any(tmp_path.iterdir())


def test_build_ablation_command_sets_cover_all_manual_groups():
    command_sets = build_ablation_command_sets()

    architecture = command_sets["architecture"]
    temperature = command_sets["temperature"]
    dynamics = command_sets["dynamics"]

    assert "--seeds 0,1,2,3,4" in architecture
    assert "--seeds 0,1,2,3,4" in temperature
    assert "--seeds 0,1,2,3,4" in dynamics
    assert "transformer,mlp,cnn,bilstm" in architecture
    assert "tau_start=10.0" in temperature
    assert "tau_end=0.08" in temperature
    assert "tau_schedule=two_stage" in temperature
    assert "tau_start=0.08" in temperature
    assert "batch" in dynamics
    assert "step" in dynamics
    assert "--inverse-pkl Data/UPCs/preprocess_inverse_upc.pkl" in dynamics
    assert "abl_arch" in architecture
    assert "abl_tau_annealed" in temperature
    assert "abl_tau_fixed" in temperature
    assert "abl_dyn_batch" in dynamics
    assert "abl_dyn_step" in dynamics


def test_build_ablation_command_sets_preserves_custom_seed_csv():
    command_sets = build_ablation_command_sets(seed_csv="7,11,13")

    assert "--seeds 7,11,13" in command_sets["architecture"]
    assert "--seeds 7,11,13" in command_sets["temperature"]
    assert "--seeds 7,11,13" in command_sets["dynamics"]


