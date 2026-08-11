from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from DPC.config import DT
from DPC.evaluate import evaluate_day_oneshot
from DPC.dynamics import UPHESDynamicsStep


def _make_system_params(max_vol_low=20.0, head_max=20.0, include_inverse=True):
    def pos_min(h):
        return h

    def pos_max(h):
        return h + 2.0

    def neg_min(h):
        return -(h + 2.0)

    def neg_max(h):
        return -h

    def upc_tur(p, h):
        del h
        return p / DT

    def upc_pump(p, h):
        del h
        return p / DT

    def v_low_to_h(v):
        return v

    params = {
        "pos_min": pos_min,
        "pos_max": pos_max,
        "neg_min": neg_min,
        "neg_max": neg_max,
        "UPC_poly_tur": upc_tur,
        "UPC_poly_pump": upc_pump,
        "v_low_to_h": v_low_to_h,
        "head_min": 0.0,
        "head_max": head_max,
        "max_vol_low": max_vol_low,
        "h_init": 1.0,
        "v_init": 1.0,
        "target_vol_low": max_vol_low,
        "target_head": 1.0,
    }
    if include_inverse:
        def upc_inv_tur(q, h):
            del h
            q_t = torch.as_tensor(q)
            return torch.where(q_t > 0, q_t * DT, torch.zeros_like(q_t))

        def upc_inv_pump(q, h):
            del h
            q_t = torch.as_tensor(q)
            return torch.where(q_t < 0, q_t * DT, torch.zeros_like(q_t))

        params["UPC_inv_tur"] = upc_inv_tur
        params["UPC_inv_pump"] = upc_inv_pump
    return params


def _make_head_sensitive_system_params(max_vol_low=10.0, head_max=10.0):
    params = _make_system_params(max_vol_low=max_vol_low, head_max=head_max)

    def upc_inv_tur(q, h):
        q_t = torch.as_tensor(q)
        h_t = torch.as_tensor(h, device=q_t.device)
        return torch.where(q_t > 0, 2.0 * q_t * DT + 10.0 * h_t, torch.zeros_like(q_t))

    def upc_inv_pump(q, h):
        q_t = torch.as_tensor(q)
        h_t = torch.as_tensor(h, device=q_t.device)
        return torch.where(q_t < 0, 3.0 * q_t * DT - 5.0 * h_t, torch.zeros_like(q_t))

    params["UPC_inv_tur"] = upc_inv_tur
    params["UPC_inv_pump"] = upc_inv_pump
    return params


def _make_exact_eval_system_params():
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
        return p / DT

    return {
        "pos_min": pos_min,
        "pos_max": pos_max,
        "neg_min": neg_min,
        "neg_max": neg_max,
        "UPC_poly_tur": upc,
        "UPC_poly_pump": upc,
        "v_low_to_h": lambda v: v,
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


def test_rollout_step_requires_inverse_upc_functions():
    with pytest.raises(ValueError, match="UPC_inv_tur and UPC_inv_pump"):
        UPHESDynamicsStep(_make_system_params(include_inverse=False))


def test_rollout_step_emits_full_sequential_trajectory_and_aux_contract():
    dynamics = UPHESDynamicsStep(_make_system_params())
    x0 = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)
    u = torch.tensor(
        [[[0.5, 0.0, 1.0],
          [0.5, 0.0, 1.0]]],
        dtype=torch.float32,
    )

    x_full, aux = dynamics(x0, u)

    expected_q = torch.tensor([2.0 / DT, 4.0 / DT], dtype=torch.float32)
    expected_x = torch.tensor(
        [[[1.0, 1.0],
          [3.0, 3.0],
          [7.0, 7.0]]],
        dtype=torch.float32,
    )
    expected_aux = torch.tensor(
        [[[2.0, 0.0, 2.0, 0.0, expected_q[0], expected_q[0], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [4.0, 0.0, 4.0, 0.0, expected_q[1], expected_q[1], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    assert x_full.shape == (1, 3, 2)
    assert aux.shape == (1, 2, 12)
    assert torch.allclose(x_full, expected_x)
    assert torch.allclose(aux, expected_aux)


def test_rollout_step_reconstructs_executed_power_from_q_exec_and_current_head():
    dynamics = UPHESDynamicsStep(_make_head_sensitive_system_params(max_vol_low=10.0, head_max=10.0))
    x0 = torch.tensor([[[9.0, 9.0]]], dtype=torch.float32)
    u = torch.tensor([[[1.0, 0.0, 1.0]]], dtype=torch.float32)

    x_full, aux = dynamics(x0, u)

    assert torch.allclose(x_full, torch.tensor([[[9.0, 9.0], [10.0, 10.0]]], dtype=torch.float32))
    assert torch.isclose(aux[0, 0, 0], torch.tensor(11.0))
    assert torch.isclose(aux[0, 0, 4], torch.tensor(11.0 / DT))
    assert torch.isclose(aux[0, 0, 5], torch.tensor(1.0 / DT))
    assert torch.isclose(aux[0, 0, 2], torch.tensor(92.0))
    assert torch.isclose(aux[0, 0, 3], torch.tensor(0.0))
    assert torch.isclose(aux[0, 0, 9], torch.tensor(10.0))


def test_rollout_step_preserves_gradient_through_clamped_volume_update():
    dynamics = UPHESDynamicsStep(_make_system_params(max_vol_low=10.0, head_max=10.0))
    x0 = torch.tensor([[[9.0, 9.0]]], dtype=torch.float32)
    u = torch.tensor([[[0.5, 0.0, 1.0]]], dtype=torch.float32, requires_grad=True)

    x_full, _ = dynamics(x0, u)
    x_full[0, -1, 1].backward()

    assert u.grad is not None
    assert u.grad[0, 0, 0].item() > 0.0


def test_rollout_step_mixed_mode_horizon_consumes_previous_clamped_state():
    dynamics = UPHESDynamicsStep(_make_system_params(max_vol_low=10.0, head_max=10.0))
    x0 = torch.tensor([[[8.0, 8.0]]], dtype=torch.float32)
    u = torch.tensor(
        [[[1.0, 0.0, 1.0],
          [0.0, 0.0, 0.0],
          [0.0, 1.0, -1.0]]],
        dtype=torch.float32,
    )

    x_full, aux = dynamics(x0, u)

    expected_x = torch.tensor(
        [[[8.0, 8.0],
          [10.0, 10.0],
          [10.0, 10.0],
          [0.0, 0.0]]],
        dtype=torch.float32,
    )

    assert torch.allclose(x_full, expected_x)
    assert torch.isclose(aux[0, 0, 0], torch.tensor(10.0))
    assert torch.isclose(aux[0, 1, 4], torch.tensor(0.0))
    assert torch.isclose(aux[0, 2, 1], torch.tensor(-10.0))
    assert torch.isclose(aux[0, 2, 5], torch.tensor(-10.0 / DT))
    assert torch.isclose(aux[0, 2, 3], torch.tensor(-10.0))


def test_rollout_step_rejects_invalid_input_shapes():
    dynamics = UPHESDynamicsStep(_make_system_params())
    valid_x = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)
    valid_u = torch.tensor([[[0.5, 0.0, 1.0]]], dtype=torch.float32)
    mismatch_u = torch.tensor(
        [[[0.5, 0.0, 1.0]],
         [[0.0, 0.5, -1.0]]],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="x must have shape"):
        dynamics(valid_x[:, 0, :], valid_u)

    with pytest.raises(ValueError, match="u must have shape"):
        dynamics(valid_x, valid_u[:, 0, :])

    with pytest.raises(ValueError, match="batch size"):
        dynamics(valid_x, mismatch_u)


def test_rollout_step_supports_zero_horizon():
    dynamics = UPHESDynamicsStep(_make_system_params())
    x0 = torch.tensor([[[2.0, 3.0]]], dtype=torch.float32)
    u = torch.zeros((1, 0, 3), dtype=torch.float32)

    x_full, aux = dynamics(x0, u)

    assert torch.allclose(x_full, x0)
    assert aux.shape == (1, 0, 12)


def test_evaluate_day_oneshot_prefers_executed_power_for_step_aux():
    prices = [2.0, 3.0]
    x = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
    aux = torch.tensor(
        [[[10.0, 1.0, 8.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0],
          [20.0, 3.0, 12.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    u = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]], dtype=torch.float32)
    problem = _FakeProblem(x, aux, u)

    result = evaluate_day_oneshot(problem, prices, 1.0, 1.0, c_op=0.0, system_params=_make_exact_eval_system_params())

    assert np.allclose(result["p_net"], np.array([11.0, 23.0]))
    assert np.allclose(result["p_sim"], np.array([11.0, 23.0]))
    assert result["revenue"] == pytest.approx(91.0)


def test_evaluate_day_oneshot_can_return_step_trace():
    prices = [2.0, 3.0]
    x = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
    aux = torch.tensor(
        [[[10.0, 1.0, 8.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0],
          [20.0, 3.0, 12.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    u = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]], dtype=torch.float32)
    problem = _FakeProblem(x, aux, u)

    result = evaluate_day_oneshot(
        problem,
        prices,
        1.0,
        1.0,
        c_op=0.0,
        system_params=_make_exact_eval_system_params(),
        return_trace=True,
    )

    assert "trace" in result
    assert {"p_cmd", "p_exec", "q_exec", "h", "v_t", "v_next"}.issubset(result["trace"])
    assert result["profit"] == pytest.approx(91.0)


def test_evaluate_day_oneshot_preserves_legacy_batch_fallback():
    prices = [float(i + 1) for i in range(24)]
    x = torch.full((1, 25, 2), 1.0, dtype=torch.float32)
    aux = torch.tensor(
        [[[10.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 24],
        dtype=torch.float32,
    )
    u = torch.tensor([[[1.0, 0.0, 1.0]] * 24], dtype=torch.float32)
    problem = _FakeProblem(x, aux, u)

    result = evaluate_day_oneshot(problem, prices, 1.0, 1.0, c_op=0.0, system_params=_make_exact_eval_system_params())

    expected_p = torch.tensor([11.0] * 24)
    assert np.allclose(result["p_net"], expected_p.numpy())
    assert np.allclose(result["p_sim"], expected_p.numpy())
    assert result["revenue"] == pytest.approx(11.0 * sum(prices))
    assert result["profit"] == pytest.approx(11.0 * sum(prices))
