from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import dill as pickle
import numpy as np
import pytest
import torch


def _load_inverse_module():
    module_path = Path(__file__).resolve().parents[2] / "Data" / "UPCs" / "preprocessing_inverse.py"
    spec = importlib.util.spec_from_file_location("preprocessing_inverse", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_config_module():
    module_path = Path(__file__).resolve().parents[2] / "DPC" / "config.py"
    spec = importlib.util.spec_from_file_location("dpc_config", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _broken_inverse_upc(q, h):
    raise RuntimeError("malformed inverse callable")


def test_inverse_upc_recovers_power_on_sampled_feasible_points(tmp_path):
    artifact_path = tmp_path / "preprocess_inverse_upc.pkl"

    module = _load_inverse_module()
    module.build_inverse_upc_artifact(output_path=artifact_path, num_head_samples=5, num_power_samples=9)

    p_tur, h_tur, q_tur = _sample_valid_points(module, module.TURBINE_SURFACE_PATH, 3)
    p_pump, h_pump, q_pump = _sample_valid_points(module, module.PUMP_SURFACE_PATH, 3)

    fresh = _run_inverse_artifact_in_fresh_process(
        artifact_path,
        q_values={"tur": q_tur.tolist(), "pump": q_pump.tolist()},
        h_values={"tur": h_tur.tolist(), "pump": h_pump.tolist()},
    )

    assert fresh["meta"]["fit_kind"] == "polynomial_inverse_upc"
    assert torch.max(torch.abs(torch.tensor(fresh["p_tur"]) - p_tur)).item() < 0.35
    assert torch.max(torch.abs(torch.tensor(fresh["p_pump"]) - p_pump)).item() < 0.35


def _sample_valid_points(module, file_path: Path, count: int):
    power_valid, h_valid, flow_valid = module._load_upc_surface(file_path)
    sample_indices = np.linspace(0, len(flow_valid) - 1, count, dtype=int)
    return (
        torch.tensor(power_valid[sample_indices], dtype=torch.float32),
        torch.tensor(h_valid[sample_indices], dtype=torch.float32),
        torch.tensor(flow_valid[sample_indices], dtype=torch.float32),
    )


def _run_inverse_artifact_in_fresh_process(artifact_path: Path, q_values, h_values, dtype: str = "float32"):
    script = """
import dill as pickle
import json
import sys
import torch

artifact_path = sys.argv[1]
q_values = json.loads(sys.argv[2])
h_values = json.loads(sys.argv[3])
dtype = getattr(torch, sys.argv[4])

with open(artifact_path, "rb") as f:
    inv_tur, inv_pump, meta = pickle.load(f)

q_tur = torch.tensor(q_values["tur"], dtype=dtype)
h_tur = torch.tensor(h_values["tur"], dtype=dtype)
q_pump = torch.tensor(q_values["pump"], dtype=dtype)
h_pump = torch.tensor(h_values["pump"], dtype=dtype)

out = {
    "meta": meta,
    "p_tur": inv_tur(q_tur, h_tur).tolist(),
    "p_pump": inv_pump(q_pump, h_pump).tolist(),
}
print(json.dumps(out))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(artifact_path), json.dumps(q_values), json.dumps(h_values), dtype],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "MKL_THREADING_LAYER": "GNU"},
    )
    return json.loads(result.stdout)


def test_inverse_upc_round_trips_forward_upc_on_sampled_feasible_points(tmp_path):
    module = _load_inverse_module()
    artifact_path = tmp_path / "preprocess_inverse_upc.pkl"
    module.build_inverse_upc_artifact(output_path=artifact_path, num_head_samples=7, num_power_samples=15)

    upc_tur, upc_pump = module.load_forward_upc_functions()

    p_tur, h_tur, q_tur = _sample_valid_points(module, module.TURBINE_SURFACE_PATH, 5)
    p_pump, h_pump, q_pump = _sample_valid_points(module, module.PUMP_SURFACE_PATH, 5)

    q_tur_pred = upc_tur(p_tur, h_tur)
    q_pump_pred = upc_pump(p_pump, h_pump)

    round_trip = _run_inverse_artifact_in_fresh_process(
        artifact_path,
        q_values={"tur": q_tur_pred.tolist(), "pump": q_pump_pred.tolist()},
        h_values={"tur": h_tur.tolist(), "pump": h_pump.tolist()},
    )

    assert torch.max(torch.abs(q_tur_pred - q_tur)).item() < 0.5
    assert torch.max(torch.abs(q_pump_pred - q_pump)).item() < 0.5
    assert round_trip["meta"]["fit_kind"] == "polynomial_inverse_upc"
    assert torch.max(torch.abs(torch.tensor(round_trip["p_tur"]) - p_tur)).item() < 0.35
    assert torch.max(torch.abs(torch.tensor(round_trip["p_pump"]) - p_pump)).item() < 0.35


def test_inverse_upc_accepts_float64_inputs_in_fresh_process(tmp_path):
    module = _load_inverse_module()
    artifact_path = tmp_path / "preprocess_inverse_upc.pkl"
    module.build_inverse_upc_artifact(output_path=artifact_path, num_head_samples=7, num_power_samples=15)

    p_tur, h_tur, q_tur = _sample_valid_points(module, module.TURBINE_SURFACE_PATH, 3)
    p_pump, h_pump, q_pump = _sample_valid_points(module, module.PUMP_SURFACE_PATH, 3)

    fresh = _run_inverse_artifact_in_fresh_process(
        artifact_path,
        q_values={"tur": q_tur.tolist(), "pump": q_pump.tolist()},
        h_values={"tur": h_tur.tolist(), "pump": h_pump.tolist()},
        dtype="float64",
    )

    assert torch.max(torch.abs(torch.tensor(fresh["p_tur"], dtype=torch.float64) - p_tur.to(dtype=torch.float64))).item() < 0.35
    assert torch.max(torch.abs(torch.tensor(fresh["p_pump"], dtype=torch.float64) - p_pump.to(dtype=torch.float64))).item() < 0.35


def test_inverse_upc_clamps_out_of_domain_inputs_in_fresh_process(tmp_path):
    module = _load_inverse_module()
    artifact_path = tmp_path / "preprocess_inverse_upc.pkl"
    module.build_inverse_upc_artifact(output_path=artifact_path, num_head_samples=5, num_power_samples=9)

    extreme = _run_inverse_artifact_in_fresh_process(
        artifact_path,
        q_values={"tur": [1e6], "pump": [-1e6]},
        h_values={"tur": [77.0], "pump": [77.0]},
    )
    tur_ranges = extreme["meta"]["ranges"]["turbine"]
    pump_ranges = extreme["meta"]["ranges"]["pump"]
    boundary = _run_inverse_artifact_in_fresh_process(
        artifact_path,
        q_values={
            "tur": [tur_ranges["q_max"]],
            "pump": [pump_ranges["q_min"]],
        },
        h_values={"tur": [77.0], "pump": [77.0]},
    )

    p_tur = torch.tensor(extreme["p_tur"])
    p_pump = torch.tensor(extreme["p_pump"])
    boundary_tur = torch.tensor(boundary["p_tur"])
    boundary_pump = torch.tensor(boundary["p_pump"])

    assert torch.all(p_tur >= tur_ranges["p_min"] - 1e-4)
    assert torch.all(p_tur <= tur_ranges["p_max"] + 1e-4)
    assert torch.all(p_pump >= pump_ranges["p_min"] - 1e-4)
    assert torch.all(p_pump <= pump_ranges["p_max"] + 1e-4)
    assert torch.allclose(p_tur, boundary_tur, atol=1e-4)
    assert torch.allclose(p_pump, boundary_pump, atol=1e-4)


def test_inverse_upc_preserves_zero_output_for_zero_and_wrong_sign_flow(tmp_path):
    module = _load_inverse_module()
    artifact_path = tmp_path / "preprocess_inverse_upc.pkl"
    module.build_inverse_upc_artifact(output_path=artifact_path, num_head_samples=5, num_power_samples=9)

    fresh = _run_inverse_artifact_in_fresh_process(
        artifact_path,
        q_values={"tur": [0.0, -5.0], "pump": [0.0, 5.0]},
        h_values={"tur": [77.0, 77.0], "pump": [77.0, 77.0]},
    )

    assert fresh["p_tur"] == [0.0, 0.0]
    assert fresh["p_pump"] == [0.0, 0.0]


def test_inverse_upc_records_truthful_sampling_metadata(tmp_path):
    module = _load_inverse_module()
    artifact_path = tmp_path / "preprocess_inverse_upc.pkl"
    module.build_inverse_upc_artifact(output_path=artifact_path, num_head_samples=3, num_power_samples=4)

    fresh = _run_inverse_artifact_in_fresh_process(
        artifact_path,
        q_values={"tur": [3.0], "pump": [-3.0]},
        h_values={"tur": [77.0], "pump": [77.0]},
    )

    diag = fresh["meta"]["diagnostics"]
    assert diag["turbine_fit_samples"] < diag["turbine_source_samples"]
    assert diag["pump_fit_samples"] < diag["pump_source_samples"]
    assert diag["turbine_selected_head_samples"] == 3
    assert diag["turbine_selected_power_samples"] == 4
    assert diag["pump_selected_head_samples"] == 3
    assert diag["pump_selected_power_samples"] == 4


def test_load_forward_upc_functions_rejects_drifted_payload_schema(tmp_path):
    module = _load_inverse_module()
    bad_path = tmp_path / "bad_preprocess.pkl"
    with open(bad_path, "wb") as f:
        pickle.dump(tuple(range(34)), f)

    with pytest.raises(ValueError, match="35 items"):
        module.load_forward_upc_functions(bad_path, device="cpu")


def test_load_forward_upc_functions_rejects_malformed_payload_entries(tmp_path):
    module = _load_inverse_module()
    bad_path = tmp_path / "bad_preprocess_callable.pkl"
    payload = list(range(35))
    payload[10] = 123
    payload[11] = 456
    with open(bad_path, "wb") as f:
        pickle.dump(tuple(payload), f)

    with pytest.raises(TypeError, match="must be callable"):
        module.load_forward_upc_functions(bad_path, device="cpu")


def test_load_forward_upc_functions_restores_torch_load_on_open_failure():
    module = _load_inverse_module()
    original_torch_load = torch.load
    missing_path = Path(__file__).resolve().parent / "does_not_exist.pkl"

    with pytest.raises(FileNotFoundError):
        module.load_forward_upc_functions(missing_path, device="cpu")

    assert torch.load is original_torch_load


def test_load_system_params_exposes_inverse_upc_functions_on_cpu(tmp_path):
    module = _load_config_module()
    inverse_module = _load_inverse_module()
    inverse_path = tmp_path / "preprocess_inverse_upc.pkl"
    inverse_module.build_inverse_upc_artifact(
        output_path=inverse_path,
        num_head_samples=5,
        num_power_samples=9,
    )

    pkl_path = Path(__file__).resolve().parents[2] / "preprocess.pkl"
    params = module.load_system_params(
        pkl_path,
        device="cpu",
        physics_mode="nonlinear",
        inverse_pkl_path=inverse_path,
    )

    q_tur = torch.tensor([3.0], device="cpu")
    h_tur = torch.tensor([77.0], device="cpu")
    q_pump = torch.tensor([-3.0], device="cpu")
    h_pump = torch.tensor([77.0], device="cpu")

    p_tur = params["UPC_inv_tur"](q_tur, h_tur)
    p_pump = params["UPC_inv_pump"](q_pump, h_pump)

    assert callable(params["UPC_inv_tur"])
    assert callable(params["UPC_inv_pump"])
    assert p_tur.device.type == "cpu"
    assert p_pump.device.type == "cpu"
    assert params["inverse_meta"]["fit_kind"] == "polynomial_inverse_upc"
    assert Path(params["inverse_pkl_path"]) == inverse_path


def test_load_system_params_rejects_malformed_inverse_payload(tmp_path):
    module = _load_config_module()
    inverse_path = tmp_path / "bad_inverse.pkl"
    with open(inverse_path, "wb") as f:
        pickle.dump((_broken_inverse_upc, _broken_inverse_upc, {"fit_kind": "polynomial_inverse_upc"}), f)

    pkl_path = Path(__file__).resolve().parents[2] / "preprocess.pkl"
    with pytest.raises(ValueError, match="failed validation call"):
        module.load_system_params(
            pkl_path,
            device="cpu",
            physics_mode="nonlinear",
            inverse_pkl_path=inverse_path,
        )


def test_load_system_params_keeps_inverse_fields_empty_without_artifact():
    module = _load_config_module()
    pkl_path = Path(__file__).resolve().parents[2] / "preprocess.pkl"
    params = module.load_system_params(pkl_path, device="cpu", physics_mode="nonlinear")

    assert params["UPC_inv_tur"] is None
    assert params["UPC_inv_pump"] is None
    assert params["inverse_meta"] is None
    assert params["inverse_pkl_path"] is None
