from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import DPC.experiments.benchmark_tuner as benchmark_tuner


class _FakeProblem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, batch):
        name = batch["name"]
        loss = self.weight.pow(2).sum()
        return {
            f"{name}_loss": loss,
            f"{name}_x": torch.zeros((1, 2, 2), dtype=torch.float32, device=self.weight.device),
            f"{name}_aux": torch.zeros((1, 1, 8), dtype=torch.float32, device=self.weight.device),
            f"{name}_u": torch.zeros((1, 1, 3), dtype=torch.float32, device=self.weight.device),
        }


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
        "epochs": 2,
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
        "num_train_samples": 8,
        "noise_std": 0.0,
        "min_price": 0.0,
        "sampler_clusters": 12,
        "shape_clusters": 8,
        "dev_fraction": 0.2,
        "eval_interval": 1,
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
        "pkl": "preprocess.pkl",
        "raw_csv": "Data/Belgium.csv",
        "benchmark_csv": "Data/price_data_2024.csv",
        "year": 2024,
        "extreme_date": "2024/08/09",
        "output_root": "DPC/outputs/benchmark_suite",
        "save_all_epochs": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_eval_summary():
    return {
        "mean_expost_profit": 1.0,
        "mean_profit": 2.0,
        "mean_volume_penalty": 3.0,
        "mean_turbine_hours": 0.0,
        "mean_pump_hours": 0.0,
        "mean_idle_hours": 0.0,
    }


def test_save_all_epochs_changes_run_name_without_affecting_default_name():
    base_cfg = benchmark_tuner.ExperimentConfig(
        architecture="transformer",
        sampler="cluster_balanced",
        batch_size=32,
    )
    opt_in_cfg = benchmark_tuner.ExperimentConfig(
        architecture="transformer",
        sampler="cluster_balanced",
        batch_size=32,
        save_all_epochs=True,
    )

    assert not base_cfg.run_name().endswith("_epch")
    assert opt_in_cfg.run_name().endswith("_epch")
    assert opt_in_cfg.run_name() != base_cfg.run_name()


def test_main_registers_save_all_epochs_flag(monkeypatch):
    args = _make_args(save_all_epochs=False)
    seen = []

    def fake_parse_args(self):
        return args

    original_add_argument = benchmark_tuner.argparse.ArgumentParser.add_argument

    def wrapped_add_argument(self, *pargs, **kwargs):
        if pargs and pargs[0] == "--save-all-epochs":
            seen.append(pargs[0])
        return original_add_argument(self, *pargs, **kwargs)

    def fake_run_single_experiment(cfg, parsed_args, refs):
        return {"run_name": cfg.run_name(), "dev_expost": 0.0}

    monkeypatch.setattr(benchmark_tuner.argparse.ArgumentParser, "add_argument", wrapped_add_argument)
    monkeypatch.setattr(benchmark_tuner.argparse.ArgumentParser, "parse_args", fake_parse_args)
    monkeypatch.setattr(benchmark_tuner, "load_reference_metrics", lambda: {})
    monkeypatch.setattr(benchmark_tuner, "run_single_experiment", fake_run_single_experiment)
    monkeypatch.setattr(benchmark_tuner.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark_tuner, "write_sweep_summary", lambda *args, **kwargs: None)

    benchmark_tuner.main()

    assert seen == ["--save-all-epochs"]


def test_run_single_experiment_saves_epoch_checkpoints_when_enabled(monkeypatch, tmp_path):
    args = _make_args(save_all_epochs=True, output_root=str(tmp_path / "runs"))
    cfg = benchmark_tuner.ExperimentConfig(
        architecture="transformer",
        sampler="cluster_balanced",
        batch_size=1,
        epochs=2,
        eval_interval=1,
        save_all_epochs=True,
    )
    checkpoint_epochs = []
    original_save = benchmark_tuner.save_policy_checkpoint

    def fake_load_system_params(*args, **kwargs):
        return {"h_init": 0.0, "v_init": 0.0}

    def fake_build_price_pools(*args, **kwargs):
        return SimpleNamespace(
            train_pool_prices=[[0.0]],
            dev_pool_prices={"dev": [0.0] * 24},
            benchmark_prices={"bench": [0.0] * 24},
            metadata={},
        )

    class _Dataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "x": torch.zeros((1, 2), dtype=torch.float32),
                "d": torch.zeros((1, 1), dtype=torch.float32),
                "name": "train",
            }

        @staticmethod
        def collate_fn(batch):
            return batch[0]

    def fake_sample_price_dataset(*args, **kwargs):
        return _Dataset(), {}

    def fake_create_oneshot_ste(*args, **kwargs):
        return SimpleNamespace(to=lambda device: SimpleNamespace(tau=1.0))

    def fake_build_oneshot_architecture(*args, **kwargs):
        return torch.nn.Identity(), torch.nn.Identity()

    def fake_build_oneshot_system(*args, **kwargs):
        return SimpleNamespace()

    def fake_build_suite_loss(cfg, params):
        return SimpleNamespace(), {"vol_balance_obj": SimpleNamespace(weight=0.0)}

    def fake_build_problem(*args, **kwargs):
        return _FakeProblem()

    def fake_evaluate_price_dict(*args, **kwargs):
        return {}, _fake_eval_summary()

    def spy_save_policy_checkpoint(problem, out_dir, epoch, save_all_epochs):
        checkpoint_epochs.append((epoch, Path(out_dir).name))
        return original_save(problem, out_dir, epoch, save_all_epochs)

    monkeypatch.setattr(benchmark_tuner, "load_system_params", fake_load_system_params)
    monkeypatch.setattr(benchmark_tuner, "build_price_pools", fake_build_price_pools)
    monkeypatch.setattr(benchmark_tuner, "sample_price_dataset", fake_sample_price_dataset)
    monkeypatch.setattr(benchmark_tuner, "UPHESDynamicsBatch", lambda params: SimpleNamespace())
    monkeypatch.setattr(benchmark_tuner, "build_oneshot_architecture", fake_build_oneshot_architecture)
    monkeypatch.setattr(benchmark_tuner, "create_oneshot_ste", fake_create_oneshot_ste)
    monkeypatch.setattr(benchmark_tuner, "build_oneshot_system", fake_build_oneshot_system)
    monkeypatch.setattr(benchmark_tuner, "build_suite_loss", fake_build_suite_loss)
    monkeypatch.setattr(benchmark_tuner, "build_problem", fake_build_problem)
    monkeypatch.setattr(benchmark_tuner, "evaluate_price_dict", fake_evaluate_price_dict)
    monkeypatch.setattr(benchmark_tuner, "plot_run_dashboard", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark_tuner, "save_policy_checkpoint", spy_save_policy_checkpoint)
    monkeypatch.setattr(benchmark_tuner, "OUTPUT_ROOT", str(tmp_path / "runs"))

    benchmark_tuner.run_single_experiment(cfg, args, {})

    assert checkpoint_epochs == [(1, cfg.run_name()), (2, cfg.run_name())]
    assert (tmp_path / "runs" / cfg.run_name() / "policy_epoch001.pt").is_file()
    assert (tmp_path / "runs" / cfg.run_name() / "policy_epoch002.pt").is_file()


def test_run_single_experiment_does_not_call_save_policy_checkpoint_when_disabled(monkeypatch, tmp_path):
    args = _make_args(save_all_epochs=False, output_root=str(tmp_path / "runs"))
    cfg = benchmark_tuner.ExperimentConfig(
        architecture="transformer",
        sampler="cluster_balanced",
        batch_size=1,
        epochs=2,
        eval_interval=1,
        save_all_epochs=False,
    )
    checkpoint_epochs = []

    def fake_load_system_params(*args, **kwargs):
        return {"h_init": 0.0, "v_init": 0.0}

    def fake_build_price_pools(*args, **kwargs):
        return SimpleNamespace(
            train_pool_prices=[[0.0]],
            dev_pool_prices={"dev": [0.0] * 24},
            benchmark_prices={"bench": [0.0] * 24},
            metadata={},
        )

    class _Dataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "x": torch.zeros((1, 2), dtype=torch.float32),
                "d": torch.zeros((1, 1), dtype=torch.float32),
                "name": "train",
            }

        @staticmethod
        def collate_fn(batch):
            return batch[0]

    def fake_sample_price_dataset(*args, **kwargs):
        return _Dataset(), {}

    def fake_create_oneshot_ste(*args, **kwargs):
        return SimpleNamespace(to=lambda device: SimpleNamespace(tau=1.0))

    def fake_build_oneshot_architecture(*args, **kwargs):
        return torch.nn.Identity(), torch.nn.Identity()

    def fake_build_oneshot_system(*args, **kwargs):
        return SimpleNamespace()

    def fake_build_suite_loss(cfg, params):
        return SimpleNamespace(), {"vol_balance_obj": SimpleNamespace(weight=0.0)}

    def fake_build_problem(*args, **kwargs):
        return _FakeProblem()

    def fake_evaluate_price_dict(*args, **kwargs):
        return {}, _fake_eval_summary()

    def spy_save_policy_checkpoint(problem, out_dir, epoch, save_all_epochs):
        checkpoint_epochs.append(epoch)
        raise AssertionError("save_policy_checkpoint should not be called when disabled")

    monkeypatch.setattr(benchmark_tuner, "load_system_params", fake_load_system_params)
    monkeypatch.setattr(benchmark_tuner, "build_price_pools", fake_build_price_pools)
    monkeypatch.setattr(benchmark_tuner, "sample_price_dataset", fake_sample_price_dataset)
    monkeypatch.setattr(benchmark_tuner, "UPHESDynamicsBatch", lambda params: SimpleNamespace())
    monkeypatch.setattr(benchmark_tuner, "build_oneshot_architecture", fake_build_oneshot_architecture)
    monkeypatch.setattr(benchmark_tuner, "create_oneshot_ste", fake_create_oneshot_ste)
    monkeypatch.setattr(benchmark_tuner, "build_oneshot_system", fake_build_oneshot_system)
    monkeypatch.setattr(benchmark_tuner, "build_suite_loss", fake_build_suite_loss)
    monkeypatch.setattr(benchmark_tuner, "build_problem", fake_build_problem)
    monkeypatch.setattr(benchmark_tuner, "evaluate_price_dict", fake_evaluate_price_dict)
    monkeypatch.setattr(benchmark_tuner, "plot_run_dashboard", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark_tuner, "save_policy_checkpoint", spy_save_policy_checkpoint)
    monkeypatch.setattr(benchmark_tuner, "OUTPUT_ROOT", str(tmp_path / "runs"))

    benchmark_tuner.run_single_experiment(cfg, args, {})

    assert checkpoint_epochs == []
    assert list((tmp_path / "runs").glob("**/policy_epoch*.pt")) == []
