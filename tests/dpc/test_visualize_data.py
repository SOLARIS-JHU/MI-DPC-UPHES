from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from DPC.visualize.data import (
    filter_runs,
    load_ablation_runs_csv,
    pick_representative_day,
    select_median_seed,
)
import DPC.visualize.fig_training_convergence as fig_training_convergence
from DPC.visualize.fig_dispatch_comparison import BENCHMARK_CSV, load_best_date


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "DPC" / "outputs" / "benchmark_suite"
RUNS_CSV = BENCH_DIR / "ABLATION_47SEED_RUNS.csv"


def test_select_median_seed_returns_retained_transformer_seed_43():
    rows = load_ablation_runs_csv(RUNS_CSV)

    transformer_rows = filter_runs(rows, study="architecture", variant="transformer")

    assert len(transformer_rows) == 47
    assert select_median_seed(rows, study="architecture", variant="transformer") == 43


def test_pick_representative_day_prefers_zero_volume_penalty_then_higher_profit():
    per_day = {
        # Highest ex-post profit overall, but disqualified by nonzero volume
        # penalty: must NOT be picked even though its profit beats every
        # zero-penalty day.
        "2024/01/09": {
            "profit": 1904.19,
            "si_penalty": -50.52,
            "volume_penalty": 1381.25,
            "expost_profit": 9999.0,
            "v_final": 354400.9,
            "n_turbine": 16,
            "n_pump": 8,
            "n_idle": 0,
        },
        # Zero volume penalty, lower profit among the eligible days.
        "2024/03/15": {
            "profit": 800.0,
            "si_penalty": -10.0,
            "volume_penalty": 0.0,
            "expost_profit": 500.0,
            "v_final": 300000.0,
            "n_turbine": 12,
            "n_pump": 6,
            "n_idle": 6,
        },
        # Zero volume penalty, highest profit among the eligible days: this
        # is the expected winner.
        "2024/08/09": {
            "profit": 1200.0,
            "si_penalty": -5.0,
            "volume_penalty": 0.0,
            "expost_profit": 700.0,
            "v_final": 310000.0,
            "n_turbine": 14,
            "n_pump": 7,
            "n_idle": 3,
        },
    }

    assert pick_representative_day(per_day) == "2024/08/09"


def test_training_convergence_uses_supplied_bench_dir_for_runs_csv(tmp_path, monkeypatch):
    seen = {}

    def fake_load_ablation_runs_csv(path):
        seen["path"] = Path(path)
        return []

    def fake_load_config(*args, **kwargs):
        return [], [], []

    monkeypatch.setattr(fig_training_convergence, "load_ablation_runs_csv", fake_load_ablation_runs_csv)
    monkeypatch.setattr(fig_training_convergence, "load_config", fake_load_config)

    out_dir = tmp_path / "out"
    bench_dir = tmp_path / "custom_bench"
    bench_dir.mkdir()

    fig_training_convergence.make_figure(bench_dir, out_dir)

    assert seen["path"] == bench_dir / "ABLATION_47SEED_RUNS.csv"


def test_dispatch_comparison_loads_representative_date_from_benchmark_csv():
    assert load_best_date(BENCHMARK_CSV) == "2024/08/09"
