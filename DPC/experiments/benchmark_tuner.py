"""Benchmark-oriented one-shot tuning harness for DPC.

This keeps the 19 representative 2024 benchmark days fixed for final inference
while building train/dev samples from the rest of 2024.

Run examples:
    python -m DPC.experiments.benchmark_tuner --architectures transformer,lstm --samplers noisy_resampling,cluster_balanced
    python -m DPC.experiments.benchmark_tuner --batch-sizes 32,128,256 --epochs 150 --run-prefix smoke
    python -m DPC.experiments.benchmark_tuner  # promoted default benchmark winner
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from neuromancer.constraint import Loss
from neuromancer.loss import PenaltyLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, OneCycleLR, ReduceLROnPlateau, SequentialLR
from torch.utils.data import DataLoader

from DPC.config import (
    C_OP,
    GRAD_CLIP,
    HEAD_PENALTY,
    LR,
    MIN_PRICE,
    NUM_TRAIN_SAMPLES,
    PRICE_NOISE_STD,
    RHO,
    G,
    ETA,
    TAU_START,
    WEIGHT_DECAY,
    WARMUP_EPOCHS,
    load_system_params,
)
from DPC.ste import create_oneshot_ste, get_temperature
from DPC.system import build_oneshot_system
from DPC.dynamics import UPHESDynamicsBatch, UPHESDynamicsStep
from DPC.experiments.benchmark_data import DEFAULT_EXTREME_DATE, build_price_pools, sample_price_dataset
from DPC.experiments.oneshot_architectures import build_oneshot_architecture
from DPC.objectives import VOL_TRAJ_PENALTY, build_problem


OUTPUT_ROOT = os.path.join("DPC", "outputs", "benchmark_suite")
MILP_CSV = os.path.join("MIQP", "MIQP_linear", "MILP_global_linear_benchmark.csv")
MIQP_CSV = os.path.join("MIQP", "MIQP_piecewise", "MIQP_piecewise_benchmark.csv")


@dataclass(frozen=True)
class ExperimentConfig:
    architecture: str
    sampler: str
    batch_size: int
    dynamics_name: str = "batch"
    inverse_pkl_path: str | None = None
    optimizer: str = "adamw"
    scheduler: str = "warmup_cosine"
    epochs: int = 25
    seed: int = 0
    lr: float = 3e-4
    weight_decay: float = 5e-3
    grad_clip: float = 0.5
    tau_start: float = TAU_START
    tau_end: float = 0.08
    tau_schedule: str = "two_stage"
    tau_decay_ratio: float = 0.75
    ste_method: str = "gumbel"
    num_train_samples: int = NUM_TRAIN_SAMPLES
    noise_std: float = PRICE_NOISE_STD
    min_price: float = MIN_PRICE
    sampler_clusters: int = 12
    shape_clusters: int = 8
    dev_fraction: float = 0.2
    eval_interval: int = 10
    vol_balance_mode: str = "squared_asymmetric"
    vol_balance_scale: float = 6.2e-8
    vol_balance_weight: float = 2.0
    vol_deficit_factor: float = 1.0
    vol_surplus_factor: float = 1.5
    vol_weight_schedule: str = "late_ramp"
    head_penalty: float = HEAD_PENALTY
    vol_traj_penalty: float = VOL_TRAJ_PENALTY
    h_lb_scale: float = 1.0
    h_ub_scale: float = 1.0
    vol_lb_scale: float = 1.0
    vol_ub_scale: float = 1.0
    c_op: float = C_OP
    si_shortage_multiplier: float = -2.0
    si_surplus_multiplier: float = -0.5
    si_penalty_weight: float = 1.0
    target_vol_penalty_weight: float = 1.0
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    nhead: int = 4
    dim_ff: int = 256
    cnn_kernel_size: int = 3
    mlp_hidden_sizes: tuple[int, ...] = (512, 512, 512)
    save_all_epochs: bool = False
    physics_mode: str = "nonlinear"
    run_prefix: str = ""

    def run_name(self) -> str:
        parts = [
            self.run_prefix.strip("_"),
            self.architecture,
            f"hs{self.hidden_size}",
            f"nl{self.num_layers}",
            f"ff{self.dim_ff}",
            self.sampler,
            f"bs{self.batch_size}",
            f"lr{self.lr:g}",
            f"gc{self.grad_clip:g}",
            f"wd{self.weight_decay:g}",
            f"do{self.dropout:g}",
            self.optimizer,
            self.scheduler,
            self.tau_schedule,
            f"tau{self.tau_end:g}",
            f"cop{self.c_op:g}",
            f"siw{self.si_penalty_weight:g}",
            f"tvw{self.target_vol_penalty_weight:g}",
            f"vw{self.vol_balance_weight:g}",
            f"sf{self.vol_surplus_factor:g}",
            f"vws{self.vol_weight_schedule}",
            f"hp{self.head_penalty:g}",
            f"vtp{self.vol_traj_penalty:g}",
            f"hls{self.h_lb_scale:g}",
            f"hus{self.h_ub_scale:g}",
            f"vls{self.vol_lb_scale:g}",
            f"vus{self.vol_ub_scale:g}",
            f"seed{self.seed}",
        ]
        if self.dynamics_name != "batch":
            parts.append(f"dyn{self.dynamics_name}")
        if self.inverse_pkl_path:
            parts.append(f"inv{Path(self.inverse_pkl_path).stem.replace('_', '-')}")
        if self.save_all_epochs:
            parts.append("epch")
        return "_".join([p for p in parts if p])

    def output_metadata(self) -> dict:
        metadata = asdict(self)
        inverse_name = Path(self.inverse_pkl_path).name if self.inverse_pkl_path else None
        metadata["inverse_pkl_name"] = inverse_name
        metadata.pop("inverse_pkl_path", None)
        return metadata


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_grid_values(args, plural_attr: str, singular_attr: str, cast):
    raw = getattr(args, plural_attr, None)
    if raw is None:
        raw = getattr(args, singular_attr)
    if isinstance(raw, str):
        items = parse_csv_list(raw)
    else:
        items = parse_csv_list(str(raw))
    return [cast(item) for item in items]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_policy_checkpoint(problem, out_dir: str, epoch: int, save_all_epochs: bool):
    if not save_all_epochs:
        return
    torch.save(problem.state_dict(), os.path.join(out_dir, f"policy_epoch{epoch:03d}.pt"))


def build_dynamics(cfg: ExperimentConfig, params: dict):
    if cfg.dynamics_name == "step":
        if params.get("UPC_inv_tur") is None or params.get("UPC_inv_pump") is None:
            raise ValueError("Step dynamics requires inverse UPC params")
        return UPHESDynamicsStep(params)
    return UPHESDynamicsBatch(params)


def load_reference_metrics() -> dict:
    refs = {}
    for label, path in (("MIQP-GL", MILP_CSV), ("MIQP-PW", MIQP_CSV)):
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        refs[label] = {
            "mean_expost": float(df["Ex-post Profit (€)"].mean()),
            "mean_profit": float(df["Expected Profit (€)"].mean()),
            "mean_volpen": float(df["Vol Penalty (€)"].mean()),
        }
    return refs


def temperature_at_epoch(epoch: int, total_epochs: int, cfg: ExperimentConfig) -> float:
    decay_epochs = max(1, int(total_epochs * cfg.tau_decay_ratio))
    progress = min(epoch / max(total_epochs - 1, 1), 1.0)
    if cfg.tau_schedule == "linear":
        if epoch < decay_epochs:
            return cfg.tau_start + (cfg.tau_end - cfg.tau_start) * (epoch / decay_epochs)
        return cfg.tau_end
    if cfg.tau_schedule == "cosine":
        if epoch < decay_epochs:
            frac = epoch / decay_epochs
            return cfg.tau_end + 0.5 * (cfg.tau_start - cfg.tau_end) * (1.0 + np.cos(np.pi * frac))
        return cfg.tau_end
    if cfg.tau_schedule == "two_stage":
        if progress < 0.35:
            return cfg.tau_start
        late_frac = (progress - 0.35) / 0.65
        return cfg.tau_end + (cfg.tau_start - cfg.tau_end) * np.exp(-5.0 * late_frac)
    return get_temperature(epoch, cfg.tau_start, cfg.tau_end, decay_epochs)


def volume_weight_at_epoch(epoch: int, total_epochs: int, cfg: ExperimentConfig) -> float:
    progress = min(epoch / max(total_epochs - 1, 1), 1.0)
    if cfg.vol_weight_schedule == "linear_ramp":
        return cfg.vol_balance_weight * (0.5 + 0.5 * progress)
    if cfg.vol_weight_schedule == "late_ramp":
        if progress < 0.5:
            return cfg.vol_balance_weight * 0.6
        late_frac = (progress - 0.5) / 0.5
        return cfg.vol_balance_weight * (0.6 + 0.4 * late_frac)
    return cfg.vol_balance_weight


def build_suite_loss(cfg: ExperimentConfig, params: dict):
    target_head = params["target_head"]
    target_vol_low = params["target_vol_low"]
    energy_conv = (RHO * G * ETA * target_head) / 3.6e9

    def profit_loss_fn(x, aux, d):
        p_opt = aux[:, :, 0:1] + aux[:, :, 1:2]
        prices = d[:, : p_opt.shape[1], :]

        if aux.shape[-1] >= 12:
            e_sim = aux[:, :, 2:3] + aux[:, :, 3:4]
        else:
            # Batch dynamics only expose commanded power and violation channels, so
            # approximate realized power by zeroing infeasible hours.
            infeas = (aux[:, :, 4:5] + aux[:, :, 5:6] + aux[:, :, 6:7] + aux[:, :, 7:8] > 1e-9).to(p_opt.dtype)
            e_sim = p_opt * (1.0 - infeas)

        revenue = torch.sum(e_sim * prices, dim=1)
        op_cost = cfg.c_op * torch.sum(e_sim ** 2, dim=1)

        si_price = torch.where(
            e_sim < p_opt,
            cfg.si_shortage_multiplier * prices,
            cfg.si_surplus_multiplier * prices,
        )
        imbalance = e_sim - p_opt
        si_penalty = torch.sum(imbalance * si_price, dim=1)

        v_final = x[:, -1, 1]
        vol_surplus = torch.relu(v_final - target_vol_low)
        med_price = torch.median(d[:, : x.shape[1] - 1, 0], dim=1)[0]
        target_vol_penalty = energy_conv * vol_surplus * med_price

        expost_profit = (
            revenue
            - op_cost
            - cfg.si_penalty_weight * si_penalty
            - cfg.target_vol_penalty_weight * target_vol_penalty.unsqueeze(-1)
        )
        return torch.mean(-expost_profit)

    profit_obj = Loss(["x", "aux", "d"], profit_loss_fn, weight=1.0, name="profit_loss")

    def vol_balance_fn(x, d):
        v_final = x[:, -1, 1]
        v_init = x[:, 0, 1]
        delta_v = v_final - v_init
        surplus = torch.relu(delta_v)
        deficit = torch.relu(-delta_v)

        if cfg.vol_balance_mode == "price_scaled_surplus":
            med_price = torch.median(d[:, : x.shape[1] - 1, 0], dim=1)[0]
            return torch.mean(energy_conv * surplus * med_price)
        if cfg.vol_balance_mode == "price_scaled_bilateral":
            med_price = torch.median(d[:, : x.shape[1] - 1, 0], dim=1)[0]
            deviation = surplus + deficit
            return torch.mean(energy_conv * deviation * med_price)
        if cfg.vol_balance_mode == "surplus_only":
            penalty = surplus ** 2
        else:
            penalty = cfg.vol_surplus_factor * surplus ** 2 + cfg.vol_deficit_factor * deficit ** 2
        return torch.mean(penalty) * cfg.vol_balance_scale

    vol_balance_obj = Loss(["x", "d"], vol_balance_fn, weight=cfg.vol_balance_weight, name="vol_balance")

    def vol_lb_fn(aux):
        return aux[:, :, 4].mean()

    def vol_ub_fn(aux):
        return aux[:, :, 5].mean()

    def h_lb_fn(aux):
        return aux[:, :, 6].mean()   # mean relu(head_min - h_raw) over (B, T)

    def h_ub_fn(aux):
        return aux[:, :, 7].mean()   # mean relu(h_raw - head_max) over (B, T)

    vol_lb_obj = Loss(["aux"], vol_lb_fn, weight=cfg.vol_traj_penalty * cfg.vol_lb_scale, name="vol_lb")
    vol_ub_obj = Loss(["aux"], vol_ub_fn, weight=cfg.vol_traj_penalty * cfg.vol_ub_scale, name="vol_ub")
    h_lb_obj = Loss(["aux"], h_lb_fn, weight=cfg.head_penalty * cfg.h_lb_scale, name="h_lb")
    h_ub_obj = Loss(["aux"], h_ub_fn, weight=cfg.head_penalty * cfg.h_ub_scale, name="h_ub")

    loss = PenaltyLoss(
        objectives=[profit_obj, vol_lb_obj, vol_ub_obj, h_lb_obj, h_ub_obj],
        constraints=[],
    )
    return loss, {"vol_balance_obj": vol_balance_obj}


def build_optimizer(cfg: ExperimentConfig, parameters):
    if cfg.optimizer == "adam":
        return Adam(parameters, lr=cfg.lr)
    if cfg.optimizer == "adamw":
        return AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


def build_scheduler(cfg: ExperimentConfig, optimizer, steps_per_epoch: int):
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr / 20), "epoch"
    if cfg.scheduler == "warmup_cosine":
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=min(WARMUP_EPOCHS, cfg.epochs))
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs - min(WARMUP_EPOCHS, cfg.epochs)), eta_min=cfg.lr / 20)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[min(WARMUP_EPOCHS, cfg.epochs)]), "epoch"
    if cfg.scheduler == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=cfg.lr * 10,
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1e4,
        ), "step"
    if cfg.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, min_lr=cfg.lr / 100), "plateau"
    raise ValueError(f"Unknown scheduler '{cfg.scheduler}'")


def evaluate_price_dict(
    problem,
    price_data: dict,
    system_params: dict,
    c_op: float = C_OP,
    si_shortage_multiplier: float = -2.0,
    si_surplus_multiplier: float = -0.5,
) -> tuple[dict, dict]:
    h_init = system_params["h_init"]
    v_init = system_params["v_init"]
    target_vol_low = system_params["target_vol_low"]
    target_head = system_params["target_head"]
    energy_conv = (RHO * G * ETA * target_head) / 3.6e9
    device = next(problem.parameters()).device

    all_results = {}
    for date, prices in price_data.items():
        prices_np = np.asarray(prices, dtype=float)
        prices_t = torch.tensor(prices_np, dtype=torch.float32).reshape(1, len(prices_np), 1)
        x0 = torch.tensor([[h_init, v_init]], dtype=torch.float32).unsqueeze(1)
        data = {
            "x": x0.to(device),
            "d": prices_t.to(device),
            "name": "test",
        }

        with torch.no_grad():
            output = problem(data)

        name = data["name"]
        x = output[f"{name}_x"][0].detach().cpu().numpy()
        aux = output[f"{name}_aux"][0].detach().cpu().numpy()
        u = output[f"{name}_u"][0].detach().cpu().numpy()

        p_net = np.asarray(aux[:, 0] + aux[:, 1], dtype=float)
        p_sim = p_net.copy()
        revenue = float(np.sum(prices_np * p_sim))
        op_cost = float(c_op * np.sum(p_sim ** 2))
        profit = revenue - op_cost
        v_final = float(x[-1, 1])
        vol_surplus = max(0.0, v_final - target_vol_low)
        med_price = float(np.median(prices_np))
        si_price = np.where(
            p_sim < p_net,
            si_shortage_multiplier * prices_np,
            si_surplus_multiplier * prices_np,
        )
        si_penalty = float(np.sum((p_sim - p_net) * si_price))

        res = {
            "date": date,
            "h_traj": np.asarray(x[1:, 0], dtype=float),
            "v_traj": np.asarray(x[1:, 1], dtype=float),
            "p_T": np.asarray(aux[:, 0], dtype=float),
            "p_P": np.asarray(aux[:, 1], dtype=float),
            "p_net": p_net,
            "p_sim": p_sim,
            "mode": np.asarray(u[:, 2], dtype=float),
            "profit": float(profit),
            "revenue": float(revenue),
            "op_cost": float(op_cost),
            "si_penalty": si_penalty,
            "volume_penalty": float(energy_conv * vol_surplus * med_price),
            "v_final": v_final,
            "n_turbine": int(np.sum(u[:, 2] > 0.5)),
            "n_idle": int(np.sum(np.abs(u[:, 2]) <= 0.5)),
            "n_pump": int(np.sum(u[:, 2] < -0.5)),
        }
        res["expost_profit"] = float(res["profit"] - res["si_penalty"] - res["volume_penalty"])
        all_results[date] = res

    profits = np.array([r["profit"] for r in all_results.values()], dtype=float)
    sipens = np.array([r["si_penalty"] for r in all_results.values()], dtype=float)
    volpens = np.array([r["volume_penalty"] for r in all_results.values()], dtype=float)
    expost = profits - sipens - volpens
    summary = {
        "num_dates": int(len(all_results)),
        "mean_profit": float(profits.mean()) if len(profits) else float("nan"),
        "mean_si_penalty": float(sipens.mean()) if len(sipens) else float("nan"),
        "mean_volume_penalty": float(volpens.mean()) if len(volpens) else float("nan"),
        "mean_expost_profit": float(expost.mean()) if len(expost) else float("nan"),
        "std_expost_profit": float(expost.std()) if len(expost) else float("nan"),
        "mean_turbine_hours": float(np.mean([r["n_turbine"] for r in all_results.values()])) if all_results else float("nan"),
        "mean_pump_hours": float(np.mean([r["n_pump"] for r in all_results.values()])) if all_results else float("nan"),
        "mean_idle_hours": float(np.mean([r["n_idle"] for r in all_results.values()])) if all_results else float("nan"),
    }
    return all_results, summary


def tensor_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().item())
    return float(value)


def plot_run_dashboard(history: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    epochs = history["epoch"]

    axes[0, 0].plot(epochs, history["loss"], label="train loss")
    axes[0, 0].plot(epochs, history["dev_expost"], label="dev ex-post")
    axes[0, 0].set_title("Loss And Dev Ex-Post")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["lr"], label="lr", color="tab:blue")
    ax_tau = axes[0, 1].twinx()
    ax_tau.plot(epochs, history["tau"], label="tau", color="tab:orange")
    axes[0, 1].set_title("Learning Rate And Temperature")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylabel("LR")
    ax_tau.set_ylabel("Tau")

    axes[1, 0].plot(epochs, history["grad_norm"], label="grad norm", color="tab:green")
    axes[1, 0].plot(epochs, history["constraint_violation"], label="constraint violation", color="tab:red")
    axes[1, 0].set_title("Gradient And Constraint Signals")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, history["profit_loss"], label="profit loss")
    axes[1, 1].plot(epochs, history["vol_balance"], label="vol balance")
    axes[1, 1].plot(epochs, history["vol_lb"], label="vol lb")
    axes[1, 1].plot(epochs, history["vol_ub"], label="vol ub")
    axes[1, 1].set_title("Loss Components")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    axes[2, 0].plot(epochs, history["mode_turbine"], label="turbine")
    axes[2, 0].plot(epochs, history["mode_pump"], label="pump")
    axes[2, 0].plot(epochs, history["mode_idle"], label="idle")
    axes[2, 0].set_title("Mode Fractions")
    axes[2, 0].grid(alpha=0.3)
    axes[2, 0].legend()

    axes[2, 1].plot(epochs, history["dev_profit"], label="dev gross")
    axes[2, 1].plot(epochs, history["dev_volpen"], label="dev vol penalty")
    axes[2, 1].plot(epochs, history["vol_balance_weight"], label="vol weight")
    axes[2, 1].set_title("Dev Metrics And Weight Schedule")
    axes[2, 1].grid(alpha=0.3)
    axes[2, 1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "training_dashboard.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_single_experiment(cfg: ExperimentConfig, args, refs: dict):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = load_system_params(
        args.pkl,
        device=device,
        physics_mode=cfg.physics_mode,
        inverse_pkl_path=cfg.inverse_pkl_path,
    )
    pools = build_price_pools(
        raw_csv_path=args.raw_csv,
        benchmark_csv_path=args.benchmark_csv,
        year=args.year,
        extreme_date=args.extreme_date,
        dev_fraction=cfg.dev_fraction,
        seed=cfg.seed,
    )

    dataset, dataset_meta = sample_price_dataset(
        pools.train_pool_prices,
        params["h_init"],
        params["v_init"],
        num_samples=cfg.num_train_samples,
        sampler=cfg.sampler,
        noise_std=cfg.noise_std,
        min_price=cfg.min_price,
        seed=cfg.seed,
        n_clusters=cfg.sampler_clusters,
        shape_clusters=cfg.shape_clusters,
        name="train",
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    dynamics = build_dynamics(cfg, params)
    net_cont, net_int = build_oneshot_architecture(
        architecture=cfg.architecture,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        nhead=cfg.nhead,
        dim_ff=cfg.dim_ff,
        cnn_kernel_size=cfg.cnn_kernel_size,
        mlp_hidden_sizes=cfg.mlp_hidden_sizes,
    )
    net_cont, net_int = net_cont.to(device), net_int.to(device)
    ste_fn = create_oneshot_ste(cfg.ste_method, tau=cfg.tau_start).to(device)

    nodes = build_oneshot_system(dynamics, net_cont, net_int, ste_fn)
    loss, handles = build_suite_loss(cfg, params)
    problem = build_problem(nodes, loss).to(device)

    optimizer = build_optimizer(cfg, problem.parameters())
    scheduler, scheduler_mode = build_scheduler(cfg, optimizer, steps_per_epoch=len(loader))

    output_root = getattr(args, "output_root", OUTPUT_ROOT)
    out_dir = os.path.join(output_root, cfg.run_name())
    os.makedirs(out_dir, exist_ok=True)

    history_rows = []
    best_state = copy.deepcopy(problem.state_dict())
    best_dev_expost = -np.inf
    best_epoch = 0

    for epoch in range(cfg.epochs):
        problem.train()
        tau = temperature_at_epoch(epoch, cfg.epochs, cfg)
        ste_fn.tau = tau
        vol_weight = volume_weight_at_epoch(epoch, cfg.epochs, cfg)
        handles["vol_balance_obj"].weight = vol_weight

        loss_sum = 0.0
        component_sums = {
            "profit_loss": 0.0,
            "vol_balance": 0.0,
            "vol_lb": 0.0,
            "vol_ub": 0.0,
            "h_lb": 0.0,
            "h_ub": 0.0,
            "objective_loss": 0.0,
            "penalty_loss": 0.0,
            "constraint_violation": 0.0,
        }
        mode_sums = {"mode_turbine": 0.0, "mode_pump": 0.0, "mode_idle": 0.0}
        grad_norm_sum = 0.0
        n_batches = 0

        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = problem(batch)
            name = batch["name"]
            loss_value = out[f"{name}_loss"]

            optimizer.zero_grad()
            loss_value.backward()
            grad_norm = clip_grad_norm_(problem.parameters(), cfg.grad_clip)
            optimizer.step()
            if scheduler_mode == "step":
                scheduler.step()

            loss_sum += tensor_scalar(loss_value)
            grad_norm_sum += float(grad_norm)
            for key in ("profit_loss", "vol_balance", "vol_lb", "vol_ub", "h_lb", "h_ub", "objective_loss", "penalty_loss"):
                out_key = f"{name}_{key}"
                if out_key in out:
                    component_sums[key] += tensor_scalar(out[out_key])
            if f"{name}_C_violations" in out:
                component_sums["constraint_violation"] += tensor_scalar(out[f"{name}_C_violations"])

            if f"{name}_u" in out:
                modes = out[f"{name}_u"][:, :, 2]
                mode_sums["mode_turbine"] += float((modes > 0.5).float().mean().item())
                mode_sums["mode_pump"] += float((modes < -0.5).float().mean().item())
                mode_sums["mode_idle"] += float((modes.abs() <= 0.5).float().mean().item())

            n_batches += 1

        if scheduler_mode == "epoch":
            scheduler.step()
        elif scheduler_mode == "plateau":
            scheduler.step(loss_sum / max(n_batches, 1))

        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / max(n_batches, 1),
            "tau": tau,
            "lr": current_lr,
            "grad_norm": grad_norm_sum / max(n_batches, 1),
            "vol_balance_weight": vol_weight,
        }
        for key, value in component_sums.items():
            row[key] = value / max(n_batches, 1)
        for key, value in mode_sums.items():
            row[key] = value / max(n_batches, 1)

        if (epoch + 1) % cfg.eval_interval == 0 or epoch == cfg.epochs - 1:
            problem.eval()
            dev_results, dev_summary = evaluate_price_dict(
                problem,
                pools.dev_pool_prices,
                params,
                c_op=cfg.c_op,
                si_shortage_multiplier=cfg.si_shortage_multiplier,
                si_surplus_multiplier=cfg.si_surplus_multiplier,
            )
            row["dev_expost"] = dev_summary["mean_expost_profit"]
            row["dev_profit"] = dev_summary["mean_profit"]
            row["dev_volpen"] = dev_summary["mean_volume_penalty"]
            if dev_summary["mean_expost_profit"] > best_dev_expost:
                best_dev_expost = dev_summary["mean_expost_profit"]
                best_state = copy.deepcopy(problem.state_dict())
                best_epoch = epoch + 1
        else:
            row["dev_expost"] = history_rows[-1]["dev_expost"] if history_rows else float("nan")
            row["dev_profit"] = history_rows[-1]["dev_profit"] if history_rows else float("nan")
            row["dev_volpen"] = history_rows[-1]["dev_volpen"] if history_rows else float("nan")

        history_rows.append(row)
        if cfg.save_all_epochs:
            save_policy_checkpoint(problem, out_dir, epoch + 1, cfg.save_all_epochs)

        if (epoch + 1) % max(1, cfg.eval_interval) == 0 or epoch == 0:
            print(
                f"[{cfg.run_name()}] epoch={epoch + 1:4d}/{cfg.epochs} "
                f"loss={row['loss']:.3f} dev_ex={row['dev_expost']:.1f} "
                f"lr={current_lr:.2e} tau={tau:.3f} bs={cfg.batch_size}"
            )

    problem.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, "policy_best.pt"))

    dev_results, dev_summary = evaluate_price_dict(
        problem,
        pools.dev_pool_prices,
        params,
        c_op=cfg.c_op,
        si_shortage_multiplier=cfg.si_shortage_multiplier,
        si_surplus_multiplier=cfg.si_surplus_multiplier,
    )
    benchmark_results, benchmark_summary = evaluate_price_dict(
        problem,
        pools.benchmark_prices,
        params,
        c_op=cfg.c_op,
        si_shortage_multiplier=cfg.si_shortage_multiplier,
        si_surplus_multiplier=cfg.si_surplus_multiplier,
    )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    plot_run_dashboard(history_df, out_dir)

    eval_data = {
        "config": cfg.output_metadata(),
        "device": str(device),
        "pool_metadata": pools.metadata,
        "sampler_metadata": dataset_meta,
        "best_epoch": best_epoch,
        "dev_summary": dev_summary,
        "benchmark_summary": benchmark_summary,
        "dev_per_day": {
            date: {
                "profit": result["profit"],
                "si_penalty": result["si_penalty"],
                "volume_penalty": result["volume_penalty"],
                "expost_profit": result["expost_profit"],
                "v_final": result["v_final"],
                "n_turbine": result["n_turbine"],
                "n_pump": result["n_pump"],
                "n_idle": result["n_idle"],
            }
            for date, result in dev_results.items()
        },
        "per_day": {
            date: {
                "profit": result["profit"],
                "si_penalty": result["si_penalty"],
                "volume_penalty": result["volume_penalty"],
                "expost_profit": result["expost_profit"],
                "v_final": result["v_final"],
                "n_turbine": result["n_turbine"],
                "n_pump": result["n_pump"],
                "n_idle": result["n_idle"],
            }
            for date, result in benchmark_results.items()
        },
        "history": {col: [float(v) for v in history_df[col].to_numpy()] for col in history_df.columns},
    }
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(eval_data, f, indent=2, cls=NumpyEncoder)

    summary_row = {
        "run_name": cfg.run_name(),
        "architecture": cfg.architecture,
        "sampler": cfg.sampler,
        "batch_size": cfg.batch_size,
        "dynamics_name": cfg.dynamics_name,
        "inverse_pkl_name": Path(cfg.inverse_pkl_path).name if cfg.inverse_pkl_path else None,
        "optimizer": cfg.optimizer,
        "scheduler": cfg.scheduler,
        "tau_schedule": cfg.tau_schedule,
        "tau_end": cfg.tau_end,
        "c_op": cfg.c_op,
        "si_penalty_weight": cfg.si_penalty_weight,
        "target_vol_penalty_weight": cfg.target_vol_penalty_weight,
        "si_shortage_multiplier": cfg.si_shortage_multiplier,
        "si_surplus_multiplier": cfg.si_surplus_multiplier,
        "vol_balance_mode": cfg.vol_balance_mode,
        "vol_balance_weight": cfg.vol_balance_weight,
        "vol_surplus_factor": cfg.vol_surplus_factor,
        "head_penalty": cfg.head_penalty,
        "vol_traj_penalty": cfg.vol_traj_penalty,
        "h_lb_scale": cfg.h_lb_scale,
        "h_ub_scale": cfg.h_ub_scale,
        "vol_lb_scale": cfg.vol_lb_scale,
        "vol_ub_scale": cfg.vol_ub_scale,
        "h_lb_penalty": cfg.head_penalty * cfg.h_lb_scale,
        "h_ub_penalty": cfg.head_penalty * cfg.h_ub_scale,
        "vol_lb_penalty": cfg.vol_traj_penalty * cfg.vol_lb_scale,
        "vol_ub_penalty": cfg.vol_traj_penalty * cfg.vol_ub_scale,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "dev_expost": dev_summary["mean_expost_profit"],
        "dev_profit": dev_summary["mean_profit"],
        "dev_volpen": dev_summary["mean_volume_penalty"],
        "benchmark_expost": benchmark_summary["mean_expost_profit"],
        "benchmark_profit": benchmark_summary["mean_profit"],
        "benchmark_volpen": benchmark_summary["mean_volume_penalty"],
        "benchmark_turbine_h": benchmark_summary["mean_turbine_hours"],
        "benchmark_pump_h": benchmark_summary["mean_pump_hours"],
        "benchmark_idle_h": benchmark_summary["mean_idle_hours"],
    }
    if "MIQP-GL" in refs:
        summary_row["vs_miqp_gl"] = benchmark_summary["mean_expost_profit"] - refs["MIQP-GL"]["mean_expost"]
    if "MIQP-PW" in refs:
        summary_row["vs_miqp_pw"] = benchmark_summary["mean_expost_profit"] - refs["MIQP-PW"]["mean_expost"]
    return summary_row


def write_sweep_summary(rows: list[dict], out_dir: str):
    discovered_rows = []
    refs = load_reference_metrics()
    pattern = os.path.join(out_dir, "*", "eval_results.json")
    for path in sorted(glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        benchmark_summary = data.get("benchmark_summary", {})
        dev_summary = data.get("dev_summary", {})
        discovered_rows.append(
            {
                "run_name": os.path.basename(os.path.dirname(path)),
                "architecture": cfg.get("architecture"),
                "sampler": cfg.get("sampler"),
                "batch_size": cfg.get("batch_size"),
                "optimizer": cfg.get("optimizer"),
                "scheduler": cfg.get("scheduler"),
                "lr": cfg.get("lr"),
                "grad_clip": cfg.get("grad_clip"),
                "hidden_size": cfg.get("hidden_size"),
                "num_layers": cfg.get("num_layers"),
                "dim_ff": cfg.get("dim_ff"),
                "nhead": cfg.get("nhead"),
                "dropout": cfg.get("dropout"),
                "weight_decay": cfg.get("weight_decay"),
                "tau_schedule": cfg.get("tau_schedule"),
                "tau_end": cfg.get("tau_end"),
                "c_op": cfg.get("c_op"),
                "si_penalty_weight": cfg.get("si_penalty_weight"),
                "target_vol_penalty_weight": cfg.get("target_vol_penalty_weight"),
                "si_shortage_multiplier": cfg.get("si_shortage_multiplier"),
                "si_surplus_multiplier": cfg.get("si_surplus_multiplier"),
                "vol_balance_mode": cfg.get("vol_balance_mode"),
                "vol_balance_weight": cfg.get("vol_balance_weight"),
                "vol_surplus_factor": cfg.get("vol_surplus_factor"),
                "head_penalty": cfg.get("head_penalty"),
                "vol_traj_penalty": cfg.get("vol_traj_penalty"),
                "h_lb_scale": cfg.get("h_lb_scale"),
                "h_ub_scale": cfg.get("h_ub_scale"),
                "vol_lb_scale": cfg.get("vol_lb_scale"),
                "vol_ub_scale": cfg.get("vol_ub_scale"),
                "h_lb_penalty": (
                    cfg.get("head_penalty") * cfg.get("h_lb_scale")
                    if cfg.get("head_penalty") is not None and cfg.get("h_lb_scale") is not None
                    else None
                ),
                "h_ub_penalty": (
                    cfg.get("head_penalty") * cfg.get("h_ub_scale")
                    if cfg.get("head_penalty") is not None and cfg.get("h_ub_scale") is not None
                    else None
                ),
                "vol_lb_penalty": (
                    cfg.get("vol_traj_penalty") * cfg.get("vol_lb_scale")
                    if cfg.get("vol_traj_penalty") is not None and cfg.get("vol_lb_scale") is not None
                    else None
                ),
                "vol_ub_penalty": (
                    cfg.get("vol_traj_penalty") * cfg.get("vol_ub_scale")
                    if cfg.get("vol_traj_penalty") is not None and cfg.get("vol_ub_scale") is not None
                    else None
                ),
                "seed": cfg.get("seed"),
                "best_epoch": data.get("best_epoch"),
                "dev_expost": dev_summary.get("mean_expost_profit"),
                "dev_profit": dev_summary.get("mean_profit"),
                "dev_volpen": dev_summary.get("mean_volume_penalty"),
                "benchmark_expost": benchmark_summary.get("mean_expost_profit"),
                "benchmark_profit": benchmark_summary.get("mean_profit"),
                "benchmark_volpen": benchmark_summary.get("mean_volume_penalty"),
                "benchmark_turbine_h": benchmark_summary.get("mean_turbine_hours"),
                "benchmark_pump_h": benchmark_summary.get("mean_pump_hours"),
                "benchmark_idle_h": benchmark_summary.get("mean_idle_hours"),
                "vs_miqp_gl": (
                    benchmark_summary.get("mean_expost_profit") - refs["MIQP-GL"]["mean_expost"]
                    if benchmark_summary.get("mean_expost_profit") is not None
                    else None
                ),
                "vs_miqp_pw": (
                    benchmark_summary.get("mean_expost_profit") - refs["MIQP-PW"]["mean_expost"]
                    if benchmark_summary.get("mean_expost_profit") is not None
                    else None
                ),
            }
        )

    if rows:
        # Keep any in-memory rows that do not yet have persisted eval JSONs.
        persisted = {row["run_name"] for row in discovered_rows}
        discovered_rows.extend([row for row in rows if row["run_name"] not in persisted])

    df = pd.DataFrame(discovered_rows).sort_values("benchmark_expost", ascending=False)
    csv_path = os.path.join(out_dir, "sweep_summary.csv")
    df.to_csv(csv_path, index=False)

    md_path = os.path.join(out_dir, "sweep_summary.md")
    with open(md_path, "w") as f:
        f.write("# Benchmark Sweep Summary\n\n")
        columns = list(df.columns)
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in df.itertuples(index=False):
            values = [str(getattr(row, col)) for col in columns]
            f.write("| " + " | ".join(values) + " |\n")
        f.write("\n")


def build_configs(args) -> list[ExperimentConfig]:
    architectures = parse_csv_list(args.architectures)
    samplers = parse_csv_list(args.samplers)
    batch_sizes = [int(x) for x in parse_csv_list(args.batch_sizes)]
    optimizers = parse_csv_list(args.optimizers)
    schedulers = parse_csv_list(args.schedulers)
    tau_schedules = parse_csv_list(args.tau_schedules)
    tau_ends = [float(x) for x in parse_csv_list(args.tau_ends)]
    vol_modes = parse_csv_list(args.vol_balance_modes)
    vol_weights = [float(x) for x in parse_csv_list(args.vol_balance_weights)]
    surplus_factors = [float(x) for x in parse_csv_list(args.vol_surplus_factors)]
    head_penalties = [float(x) for x in parse_csv_list(args.head_penalties)]
    vol_traj_penalties = [float(x) for x in parse_csv_list(args.vol_traj_penalties)]
    h_lb_scales = [float(x) for x in parse_csv_list(args.h_lb_scales)]
    h_ub_scales = [float(x) for x in parse_csv_list(args.h_ub_scales)]
    vol_lb_scales = [float(x) for x in parse_csv_list(args.vol_lb_scales)]
    vol_ub_scales = [float(x) for x in parse_csv_list(args.vol_ub_scales)]
    seeds = [int(x) for x in parse_csv_list(args.seeds)]
    lrs = parse_grid_values(args, "lrs", "lr", float)
    grad_clips = parse_grid_values(args, "grad_clips", "grad_clip", float)
    weight_decays = parse_grid_values(args, "weight_decays", "weight_decay", float)
    target_vol_penalty_weights = parse_grid_values(
        args,
        "target_vol_penalty_weights",
        "target_vol_penalty_weight",
        float,
    )
    dynamics_name = str(getattr(args, "dynamics", "batch")).lower()
    inverse_pkl_path = getattr(args, "inverse_pkl", None)
    if dynamics_name == "step" and getattr(args, "physics", "nonlinear") != "nonlinear":
        raise ValueError("Step dynamics requires physics='nonlinear'")

    configs = []
    for values in itertools.product(
        architectures,
        samplers,
        batch_sizes,
        optimizers,
        schedulers,
        tau_schedules,
        tau_ends,
        vol_modes,
        vol_weights,
        surplus_factors,
        head_penalties,
        vol_traj_penalties,
        h_lb_scales,
        h_ub_scales,
        vol_lb_scales,
        vol_ub_scales,
        seeds,
        lrs,
        grad_clips,
        weight_decays,
        target_vol_penalty_weights,
    ):
        (
            architecture,
            sampler,
            batch_size,
            optimizer,
            scheduler,
            tau_schedule,
            tau_end,
            vol_mode,
            vol_weight,
            surplus_factor,
            head_penalty,
            vol_traj_penalty,
            h_lb_scale,
            h_ub_scale,
            vol_lb_scale,
            vol_ub_scale,
            seed,
            lr,
            grad_clip,
            weight_decay,
            target_vol_penalty_weight,
        ) = values
        configs.append(
            ExperimentConfig(
                architecture=architecture,
                sampler=sampler,
                batch_size=batch_size,
                dynamics_name=dynamics_name,
                inverse_pkl_path=inverse_pkl_path,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=args.epochs,
                seed=seed,
                lr=lr,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                tau_start=args.tau_start,
                tau_end=tau_end,
                tau_schedule=tau_schedule,
                tau_decay_ratio=args.tau_decay_ratio,
                c_op=args.c_op,
                si_shortage_multiplier=args.si_shortage_multiplier,
                si_surplus_multiplier=args.si_surplus_multiplier,
                si_penalty_weight=args.si_penalty_weight,
                target_vol_penalty_weight=target_vol_penalty_weight,
                ste_method=args.ste,
                num_train_samples=args.num_train_samples,
                noise_std=args.noise_std,
                min_price=args.min_price,
                sampler_clusters=args.sampler_clusters,
                shape_clusters=args.shape_clusters,
                dev_fraction=args.dev_fraction,
                eval_interval=args.eval_interval,
                vol_balance_mode=vol_mode,
                vol_balance_scale=args.vol_balance_scale,
                vol_balance_weight=vol_weight,
                vol_deficit_factor=args.vol_deficit_factor,
                vol_surplus_factor=surplus_factor,
                vol_weight_schedule=args.vol_weight_schedule,
                head_penalty=head_penalty,
                vol_traj_penalty=vol_traj_penalty,
                h_lb_scale=h_lb_scale,
                h_ub_scale=h_ub_scale,
                vol_lb_scale=vol_lb_scale,
                vol_ub_scale=vol_ub_scale,
                save_all_epochs=getattr(args, "save_all_epochs", False),
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                nhead=args.nhead,
                dim_ff=args.dim_ff,
                cnn_kernel_size=args.cnn_kernel_size,
                mlp_hidden_sizes=tuple(int(x) for x in parse_csv_list(args.mlp_hidden_sizes)),
                physics_mode=args.physics,
                run_prefix=args.run_prefix,
            )
        )
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="preprocess.pkl")
    parser.add_argument("--inverse-pkl", default=None)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--raw-csv", default="Data/Belgium.csv")
    parser.add_argument("--benchmark-csv", default="Data/price_data_2024.csv")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--extreme-date", default=DEFAULT_EXTREME_DATE)
    parser.add_argument("--physics", default="nonlinear", choices=["linear", "nonlinear"])
    parser.add_argument("--ste", default="gumbel", choices=["gumbel", "sparsemax", "soft"])
    parser.add_argument("--dynamics", default="batch", choices=["batch", "step"])

    parser.add_argument("--architectures", default="transformer")
    parser.add_argument("--samplers", default="cluster_balanced")
    parser.add_argument("--batch-sizes", default="32")
    parser.add_argument("--optimizers", default="adamw")
    parser.add_argument("--schedulers", default="warmup_cosine")
    parser.add_argument("--tau-schedules", default="two_stage")
    parser.add_argument("--tau-ends", default="0.08")
    parser.add_argument("--vol-balance-modes", default="squared_asymmetric")
    parser.add_argument("--vol-balance-weights", default="2.0")
    parser.add_argument("--vol-surplus-factors", default="1.5")
    parser.add_argument("--head-penalties", default=f"{HEAD_PENALTY:g}")
    parser.add_argument("--vol-traj-penalties", default=f"{VOL_TRAJ_PENALTY:g}")
    parser.add_argument("--h-lb-scales", default="1.0")
    parser.add_argument("--h-ub-scales", default="1.0")
    parser.add_argument("--vol-lb-scales", default="1.0")
    parser.add_argument("--vol-ub-scales", default="1.0")
    parser.add_argument("--seeds", default="0")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--num-train-samples", type=int, default=NUM_TRAIN_SAMPLES)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--noise-std", type=float, default=PRICE_NOISE_STD)
    parser.add_argument("--min-price", type=float, default=MIN_PRICE)
    parser.add_argument("--sampler-clusters", type=int, default=12)
    parser.add_argument("--shape-clusters", type=int, default=8)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lrs", default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--weight-decays", default=None)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--grad-clips", default=None)
    parser.add_argument("--tau-start", type=float, default=TAU_START)
    parser.add_argument("--tau-decay-ratio", type=float, default=0.75)
    parser.add_argument("--vol-balance-scale", type=float, default=6.2e-8)
    parser.add_argument("--vol-deficit-factor", type=float, default=1.0)
    parser.add_argument("--vol-weight-schedule", default="late_ramp", choices=["constant", "linear_ramp", "late_ramp"])
    parser.add_argument("--c-op", type=float, default=C_OP)
    parser.add_argument("--si-shortage-multiplier", type=float, default=-2.0)
    parser.add_argument("--si-surplus-multiplier", type=float, default=-0.5)
    parser.add_argument("--si-penalty-weight", type=float, default=1.0)
    parser.add_argument("--target-vol-penalty-weight", type=float, default=1.0)
    parser.add_argument("--target-vol-penalty-weights", default=None)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=256)
    parser.add_argument("--cnn-kernel-size", type=int, default=3)
    parser.add_argument("--mlp-hidden-sizes", default="512,512,512")
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--save-all-epochs", action="store_true", default=False)
    args = parser.parse_args()

    refs = load_reference_metrics()
    configs = build_configs(args)
    output_root = getattr(args, "output_root", OUTPUT_ROOT)
    os.makedirs(output_root, exist_ok=True)

    rows = []
    print(f"Running {len(configs)} experiment(s) under {output_root}")
    for cfg in configs:
        rows.append(run_single_experiment(cfg, args, refs))
        write_sweep_summary(rows, output_root)

    print("\nTop configurations:")
    summary_df = pd.DataFrame(rows).sort_values("dev_expost", ascending=False)
    print(summary_df.head(min(10, len(summary_df))).to_string(index=False))


if __name__ == "__main__":
    main()
