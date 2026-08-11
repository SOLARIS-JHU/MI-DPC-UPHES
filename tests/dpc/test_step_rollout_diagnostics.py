from pathlib import Path
import json
import importlib
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_step_rollout_diagnostics_writes_trace_json_for_selected_day(tmp_path):
    module = importlib.import_module("DPC.experiments.step_rollout_diagnostics")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    module.run_diagnostics(run_dir=run_dir, days=["worst_surplus"], output_dir=output_dir)

    trace_files = sorted(output_dir.glob("*.json"))
    assert trace_files


def test_verify_trace_consistency_checks_key_rollout_metrics():
    module = importlib.import_module("DPC.experiments.step_rollout_diagnostics")
    trace_data = {
        "q_exec": [1.0, 2.0, 3.0],
        "q_ref": [1.0, 2.0, 3.0],
        "p_exec": [10.0, 11.0, 12.0],
        "p_ref": [10.0, 11.0, 12.0],
        "p_cmd": [10.0, 11.0, 12.0],
        "p_cmd_ref": [10.0, 11.0, 12.0],
        "v_final": 9.5,
        "v_final_ref": 9.5,
        "profit": 42.0,
        "profit_ref": 42.0,
        "si_penalty": 1.25,
        "si_penalty_ref": 1.25,
        "volume_penalty": 3.5,
        "volume_penalty_ref": 3.5,
    }

    report = module.verify_trace_consistency(trace_data)

    assert report["passed"] is True
    assert report["max_q_error"] < 1e-4
    assert report["max_p_error"] < 1e-3
    assert report["max_p_cmd_error"] < 1e-6
    assert report["v_final_error"] < 1e-6
    assert report["max_profit_error"] < 1e-2
    assert report["max_si_penalty_error"] < 1e-2
    assert report["max_volume_penalty_error"] < 1e-2


def test_ablation_summary_computes_expected_metrics_for_dense_history():
    module = importlib.import_module("DPC.experiments.ablation_summary")
    history = {
        "epoch": [1, 2, 3, 4, 5, 6],
        "grad_norm": [0.0, 0.05, 0.2, 0.4, 0.3, 0.1],
        "dev_expost": [1.0, 1.5, 2.0, 2.6, 2.7, 2.9],
    }

    summary = module.summarize_history(history)

    assert summary["grad_norm_mean"] == pytest.approx(0.175)
    assert summary["grad_norm_std"] == pytest.approx(0.14068285846778444)
    assert summary["grad_norm_cv"] == pytest.approx(0.8039020483873396)
    assert summary["near_zero_grad_frac"] == pytest.approx(1 / 6)
    assert summary["late_to_early_grad_ratio"] == pytest.approx(3.2)
    assert summary["best_dev_epoch"] == 6
    assert summary["dev_expost_slope_tail"] == pytest.approx(0.35)


def test_ablation_summary_handles_nan_and_missing_history_via_eval_results_loader(tmp_path):
    module = importlib.import_module("DPC.experiments.ablation_summary")
    eval_results_path = tmp_path / "eval_results.json"
    payload = {
        "history": [
            {"grad_norm": 0.1, "dev_expost": float("nan")},
            {"grad_norm": None},
            {"grad_norm": float("nan"), "dev_expost": float("nan")},
            {"grad_norm": 0.2, "dev_expost": None},
        ]
    }
    eval_results_path.write_text(json.dumps(payload))

    summary = module.summarize_eval_results(eval_results_path)

    assert summary["grad_norm_mean"] == pytest.approx(0.15)
    assert summary["grad_norm_std"] == pytest.approx(0.05)
    assert summary["grad_norm_cv"] == pytest.approx(1 / 3)
    assert summary["near_zero_grad_frac"] == pytest.approx(0.0)
    assert summary["late_to_early_grad_ratio"] == pytest.approx(2.0)
    assert summary["best_dev_epoch"] == -1
    assert math.isnan(summary["dev_expost_slope_tail"])


def test_ablation_summary_uses_near_zero_threshold_not_eps(tmp_path, capsys):
    module = importlib.import_module("DPC.experiments.ablation_summary")
    eval_results_path = tmp_path / "eval_results.json"
    payload = {
        "history": {
            "epoch": [1, 2],
            "grad_norm": [5e-3, 2e-2],
            "dev_expost": [1.0, 2.0],
        }
    }
    eval_results_path.write_text(json.dumps(payload))

    direct_summary = module.summarize_eval_results(
        eval_results_path,
        eps=1e-1,
        near_zero_threshold=1e-2,
    )
    many_summary = module.summarize_eval_results_many(
        [eval_results_path],
        eps=1e-1,
        near_zero_threshold=1e-2,
    )[0]

    assert direct_summary["near_zero_grad_frac"] == pytest.approx(0.5)
    assert many_summary["near_zero_grad_frac"] == pytest.approx(0.5)

    exit_code = module.main(["--eps", "0.1", "--near-zero-threshold", "0.01", str(eval_results_path)])
    output = capsys.readouterr().out
    parsed = json.loads(output)

    assert exit_code == 0
    assert parsed[0]["near_zero_grad_frac"] == pytest.approx(0.5)


def test_ablation_summary_aggregate_handles_all_nan_columns():
    module = importlib.import_module("DPC.experiments.ablation_summary")
    summaries = [
        {
            "grad_norm_mean": float("nan"),
            "grad_norm_std": float("nan"),
            "best_dev_epoch": -1,
        },
        {
            "grad_norm_mean": float("nan"),
            "grad_norm_std": float("nan"),
            "best_dev_epoch": -1,
        },
    ]

    aggregated = module.aggregate_summaries(summaries)

    assert aggregated["count"] == 2
    assert math.isnan(aggregated["grad_norm_mean_mean"])
    assert math.isnan(aggregated["grad_norm_mean_std"])
    assert math.isnan(aggregated["grad_norm_std_mean"])
    assert math.isnan(aggregated["grad_norm_std_std"])
    assert aggregated["best_dev_epoch_mean"] == pytest.approx(-1.0)
    assert aggregated["best_dev_epoch_std"] == pytest.approx(0.0)
