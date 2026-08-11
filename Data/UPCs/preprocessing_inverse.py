from __future__ import annotations

import argparse
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
FORWARD_ARTIFACT_PATH = REPO_ROOT / "preprocess.pkl"
PUMP_SURFACE_PATH = MODULE_DIR / "temp" / "Mod_Francis_pump_temp.xlsx"
TURBINE_SURFACE_PATH = MODULE_DIR / "temp" / "Mod_Francis_turbine_temp.xlsx"


def _as_tensor(value, *, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device) if device is not None else value
    return torch.tensor(value, dtype=torch.float32, device=device)


def _broadcast_xy(first, second):
    first_t = _as_tensor(first)
    second_t = _as_tensor(second, device=first_t.device)
    return torch.broadcast_tensors(first_t, second_t)


def _evaluate_polynomial_surface(first, second, coefs, intercept, poly_degree: int):
    first_t = torch.as_tensor(first)
    second_t = torch.as_tensor(second, device=first_t.device)
    first_t, second_t = torch.broadcast_tensors(first_t, second_t)
    device = first_t.device
    target_dtype = torch.promote_types(first_t.dtype, second_t.dtype)
    if isinstance(coefs, torch.Tensor):
        target_dtype = torch.promote_types(target_dtype, coefs.dtype)
    if isinstance(intercept, torch.Tensor):
        target_dtype = torch.promote_types(target_dtype, intercept.dtype)
    first_t = first_t.to(dtype=target_dtype)
    second_t = second_t.to(dtype=target_dtype)
    coefs = torch.as_tensor(coefs, dtype=target_dtype, device=device)
    intercept = torch.as_tensor(intercept, dtype=target_dtype, device=device)

    powers = torch.arange(poly_degree + 1, device=device)
    first_pows = torch.pow(first_t.unsqueeze(-1), powers)
    second_pows = torch.pow(second_t.unsqueeze(-1), powers)

    terms = []
    for total_degree in range(1, poly_degree + 1):
        for first_degree in range(total_degree, -1, -1):
            second_degree = total_degree - first_degree
            terms.append(first_pows[..., first_degree] * second_pows[..., second_degree])

    features = torch.stack(terms, dim=-1)
    return torch.einsum("...f,f->...", features, coefs) + intercept


def make_inverse_upc_function(*, coefs, intercept, poly_degree: int, mode: str, q_min, q_max, p_min, p_max):
    def inverse_upc(q, h):
        q_t = torch.as_tensor(q)
        h_t = torch.as_tensor(h, device=q_t.device)
        q_t, h_t = torch.broadcast_tensors(q_t, h_t)
        q_raw = q_t

        device = q_t.device
        target_dtype = torch.promote_types(q_t.dtype, h_t.dtype)
        if isinstance(coefs, torch.Tensor):
            target_dtype = torch.promote_types(target_dtype, coefs.dtype)
        if isinstance(intercept, torch.Tensor):
            target_dtype = torch.promote_types(target_dtype, intercept.dtype)
        target_dtype = torch.promote_types(target_dtype, torch.float32)

        q_t = q_t.to(dtype=target_dtype)
        h_t = h_t.to(dtype=target_dtype)
        coefs_t = torch.as_tensor(coefs, dtype=target_dtype, device=device)
        intercept_t = torch.as_tensor(intercept, dtype=target_dtype, device=device)
        q_min_t = torch.as_tensor(q_min, dtype=target_dtype, device=device)
        q_max_t = torch.as_tensor(q_max, dtype=target_dtype, device=device)
        p_min_t = torch.as_tensor(p_min, dtype=target_dtype, device=device)
        p_max_t = torch.as_tensor(p_max, dtype=target_dtype, device=device)

        q_t = torch.clamp(q_t, min=q_min_t, max=q_max_t)

        powers = torch.arange(poly_degree + 1, device=device)
        q_pows = torch.pow(q_t.unsqueeze(-1), powers)
        h_pows = torch.pow(h_t.unsqueeze(-1), powers)

        terms = []
        for total_degree in range(1, poly_degree + 1):
            for q_degree in range(total_degree, -1, -1):
                h_degree = total_degree - q_degree
                terms.append(q_pows[..., q_degree] * h_pows[..., h_degree])

        features = torch.stack(terms, dim=-1)
        p = torch.einsum("...f,f->...", features, coefs_t) + intercept_t
        p = torch.clamp(p, min=p_min_t, max=p_max_t)
        if mode == "turbine":
            return torch.where(q_raw > 0, p, torch.zeros_like(p))
        if mode == "pump":
            return torch.where(q_raw < 0, p, torch.zeros_like(p))
        raise ValueError(f"Unsupported inverse UPC mode: {mode}")

    return inverse_upc


def _retarget_pickle_function(fn, device: torch.device):
    defaults = fn.__defaults__ or ()
    retargeted_defaults = []
    for value in defaults:
        if isinstance(value, torch.Tensor):
            retargeted_defaults.append(value.to(device=device))
        else:
            retargeted_defaults.append(value)
    fn.__defaults__ = tuple(retargeted_defaults)
    fn.__globals__["torch"] = torch
    fn.__globals__["device"] = device
    return fn


def _validate_forward_upc_function(fn, name: str):
    if not callable(fn):
        raise TypeError(f"{name} must be callable, got {type(fn)!r}")

    sample_power = torch.tensor([5.0, -5.0], dtype=torch.float32)
    sample_head = torch.tensor([75.0, 75.0], dtype=torch.float32)
    try:
        sample_output = fn(sample_power, sample_head)
    except Exception as exc:  # pragma: no cover - defensive validation
        raise ValueError(f"{name} failed validation call: {exc}") from exc
    if not isinstance(sample_output, torch.Tensor):
        raise TypeError(f"{name} must return a torch.Tensor, got {type(sample_output)!r}")
    if sample_output.shape != sample_power.shape:
        raise ValueError(
            f"{name} returned shape {tuple(sample_output.shape)} for sample input "
            f"{tuple(sample_power.shape)}"
        )
    return fn


def _load_upc_surface(file_path: Path):
    data = pd.read_excel(file_path)
    power = np.array(data.columns[1:], dtype=float)
    head = np.array(data.iloc[:, 0], dtype=float)
    flow = data.iloc[:, 1:].to_numpy(dtype=float)

    valid = ~np.isnan(flow)
    x_valid = np.broadcast_to(power, flow.shape)[valid]
    y_valid = np.broadcast_to(head[:, None], flow.shape)[valid]
    z_valid = flow[valid]
    return x_valid, y_valid, z_valid


def _load_upc_surface_grid(file_path: Path):
    data = pd.read_excel(file_path)
    power = np.array(data.columns[1:], dtype=float)
    head = np.array(data.iloc[:, 0], dtype=float)
    flow = data.iloc[:, 1:].to_numpy(dtype=float)
    return power, head, flow


def _select_inverse_fit_samples(file_path: Path, num_head_samples: int, num_power_samples: int):
    power_grid, head_grid, flow_grid = _load_upc_surface_grid(file_path)
    head_idx = np.unique(np.linspace(0, len(head_grid) - 1, min(num_head_samples, len(head_grid)), dtype=int))
    power_idx = np.unique(np.linspace(0, len(power_grid) - 1, min(num_power_samples, len(power_grid)), dtype=int))

    valid_mask = ~np.isnan(flow_grid)
    selected_mask = np.zeros_like(valid_mask, dtype=bool)
    selected_mask[head_idx, :] = True
    selected_mask[:, power_idx] = True
    selected_mask &= valid_mask

    sampled_power = power_grid[np.where(selected_mask)[1]]
    sampled_head = head_grid[np.where(selected_mask)[0]]
    sampled_flow = flow_grid[selected_mask]

    if sampled_power.size == 0:
        sampled_power, sampled_head, sampled_flow = _load_upc_surface(file_path)

    sampled_power = np.asarray(sampled_power, dtype=float)
    sampled_head = np.asarray(sampled_head, dtype=float)
    sampled_flow = np.asarray(sampled_flow, dtype=float)
    return sampled_power, sampled_head, sampled_flow, len(head_idx), len(power_idx)


def _fit_inverse_surface(file_path: Path, num_head_samples: int, num_power_samples: int, poly_degree: int = 5):
    power_valid, h_valid, flow_valid, sampled_head_count, sampled_power_count = _select_inverse_fit_samples(
        file_path, num_head_samples, num_power_samples
    )
    full_power_valid, _, full_flow_valid = _load_upc_surface(file_path)

    features = np.vstack([flow_valid, h_valid]).T
    model = make_pipeline(PolynomialFeatures(degree=poly_degree, include_bias=False), LinearRegression())
    model.fit(features, power_valid)

    regression = model.named_steps["linearregression"]
    coefs = torch.tensor(regression.coef_, dtype=torch.float32)
    intercept = torch.tensor(regression.intercept_, dtype=torch.float32)

    score = model.score(features, power_valid)
    mode = "turbine" if np.nanmax(power_valid) > 0 else "pump"
    q_min = float(np.nanmin(flow_valid))
    q_max = float(np.nanmax(flow_valid))
    p_min = float(np.nanmin(power_valid))
    p_max = float(np.nanmax(power_valid))
    return {
        "function": make_inverse_upc_function(
            coefs=coefs,
            intercept=intercept,
            poly_degree=poly_degree,
            mode=mode,
            q_min=q_min,
            q_max=q_max,
            p_min=p_min,
            p_max=p_max,
        ),
        "coefs": coefs,
        "intercept": intercept,
        "r2": float(score),
        "num_samples": int(len(power_valid)),
        "sampled_head_count": int(sampled_head_count),
        "sampled_power_count": int(sampled_power_count),
        "source_samples": int(len(full_power_valid)),
        "q_min": q_min,
        "q_max": q_max,
        "p_min": p_min,
        "p_max": p_max,
    }


def load_forward_upc_functions(pkl_path: Path | str = FORWARD_ARTIFACT_PATH, device: torch.device | None = None):
    device = device or torch.device("cpu")
    _torch_load = torch.load

    def _safe_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", device)
        kwargs.setdefault("weights_only", False)
        return _torch_load(*args, **kwargs)

    torch.load = _safe_torch_load
    try:
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
    finally:
        torch.load = _torch_load

    if not isinstance(payload, (tuple, list)):
        raise TypeError(f"Expected preprocess pickle payload to be tuple/list, got {type(payload)!r}")
    if len(payload) != 35:
        raise ValueError(f"Expected preprocess pickle payload with 35 items, got {len(payload)}")

    (
        v_up_to_h,
        h_to_v_up,
        v_low_to_h,
        h_to_v_low,
        coefs_tur_lin,
        intercept_tur_lin,
        coefs_pump_lin,
        intercept_pump_lin,
        UPC_linear_tur,
        UPC_linear_pump,
        UPC_poly_tur,
        UPC_poly_pump,
        neg_min,
        neg_max,
        pos_min,
        pos_max,
        h_fit,
        neg_min_fit,
        neg_max_fit,
        pos_min_fit,
        pos_max_fit,
        DA_price_hour,
        DA_price_quarter,
        prepare_and_fit_model,
        get_UPC_bound,
        LR_UPC_bound,
        max_vol_up,
        max_vol_low,
        max_vol,
        head_min,
        head_max,
        nl_h_to_v_low,
        nl_v_low_to_h,
        nl_h_v_coeffs,
        nl_v_low_h_coeffs,
    ) = payload

    # preprocess.pkl stores the forward UPC functions in the exact schema above.
    if not callable(UPC_poly_tur):
        raise TypeError(f"UPC_poly_tur must be callable, got {type(UPC_poly_tur)!r}")
    if not callable(UPC_poly_pump):
        raise TypeError(f"UPC_poly_pump must be callable, got {type(UPC_poly_pump)!r}")

    upc_poly_tur = _retarget_pickle_function(UPC_poly_tur, device)
    upc_poly_pump = _retarget_pickle_function(UPC_poly_pump, device)
    _validate_forward_upc_function(upc_poly_tur, "UPC_poly_tur")
    _validate_forward_upc_function(upc_poly_pump, "UPC_poly_pump")
    return upc_poly_tur, upc_poly_pump


def build_inverse_upc_artifact(
    output_path: Path | str,
    num_head_samples: int = 51,
    num_power_samples: int = 201,
    poly_degree: int = 5,
):
    output_path = Path(output_path)
    # The sampling arguments are kept for compatibility with the rollout plan and are
    # recorded in metadata. The fit itself uses the feasible samples in the source grids.
    turbine_fit = _fit_inverse_surface(
        TURBINE_SURFACE_PATH,
        num_head_samples=num_head_samples,
        num_power_samples=num_power_samples,
        poly_degree=poly_degree,
    )
    pump_fit = _fit_inverse_surface(
        PUMP_SURFACE_PATH,
        num_head_samples=num_head_samples,
        num_power_samples=num_power_samples,
        poly_degree=poly_degree,
    )

    inv_tur = turbine_fit["function"]
    inv_pump = pump_fit["function"]
    meta = {
        "fit_kind": "polynomial_inverse_upc",
        "poly_degree": poly_degree,
        "num_head_samples": num_head_samples,
        "num_power_samples": num_power_samples,
        "source_files": {
            "turbine": str(TURBINE_SURFACE_PATH),
            "pump": str(PUMP_SURFACE_PATH),
        },
        "diagnostics": {
            "turbine_r2": turbine_fit["r2"],
            "pump_r2": pump_fit["r2"],
            "turbine_samples": turbine_fit["num_samples"],
            "pump_samples": pump_fit["num_samples"],
            "turbine_fit_samples": turbine_fit["num_samples"],
            "pump_fit_samples": pump_fit["num_samples"],
            "turbine_source_samples": turbine_fit["source_samples"],
            "pump_source_samples": pump_fit["source_samples"],
            "turbine_selected_head_samples": turbine_fit["sampled_head_count"],
            "turbine_selected_power_samples": turbine_fit["sampled_power_count"],
            "pump_selected_head_samples": pump_fit["sampled_head_count"],
            "pump_selected_power_samples": pump_fit["sampled_power_count"],
        },
        "ranges": {
            "turbine": {
                "q_min": turbine_fit["q_min"],
                "q_max": turbine_fit["q_max"],
                "p_min": turbine_fit["p_min"],
                "p_max": turbine_fit["p_max"],
            },
            "pump": {
                "q_min": pump_fit["q_min"],
                "q_max": pump_fit["q_max"],
                "p_min": pump_fit["p_min"],
                "p_max": pump_fit["p_max"],
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump((inv_tur, inv_pump, meta), f)

    return output_path, meta


def _parse_args():
    parser = argparse.ArgumentParser(description="Build inverse UPC artifact.")
    parser.add_argument(
        "--output",
        type=Path,
        default=MODULE_DIR / "preprocess_inverse_upc.pkl",
        help="Output pickle path for inverse UPC functions.",
    )
    parser.add_argument("--num-head-samples", type=int, default=51)
    parser.add_argument("--num-power-samples", type=int, default=201)
    parser.add_argument("--poly-degree", type=int, default=5)
    return parser.parse_args()


def main():
    args = _parse_args()
    output_path, meta = build_inverse_upc_artifact(
        output_path=args.output,
        num_head_samples=args.num_head_samples,
        num_power_samples=args.num_power_samples,
        poly_degree=args.poly_degree,
    )
    print(f"Saved inverse UPC artifact to {output_path}")
    print(
        "Fit diagnostics: "
        f"turbine_r2={meta['diagnostics']['turbine_r2']:.6f}, "
        f"pump_r2={meta['diagnostics']['pump_r2']:.6f}, "
        f"turbine_samples={meta['diagnostics']['turbine_samples']}, "
        f"pump_samples={meta['diagnostics']['pump_samples']}"
    )


if __name__ == "__main__":
    main()
