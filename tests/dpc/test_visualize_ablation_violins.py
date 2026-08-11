from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import DPC.visualize.fig_ablation_violins as fig_ablation_violins


def _sample_data() -> dict[str, dict[str, list[float]]]:
    return {
        "transformer": {"expost": [2480.0, 2510.0, 2550.0], "train_s": [160.0, 162.0, 164.0], "infer_ms": [4.2, 4.4, 4.6]},
        "bilstm": {"expost": [2410.0, 2450.0, 2470.0], "train_s": [150.0, 151.0, 152.0], "infer_ms": [4.8, 5.0, 5.2]},
        "mlp": {"expost": [2310.0, 2360.0, 2390.0], "train_s": [120.0, 121.0, 122.0], "infer_ms": [3.6, 3.7, 3.8]},
        "cnn": {"expost": [2050.0, 2100.0, 2140.0], "train_s": [130.0, 131.0, 132.0], "infer_ms": [3.9, 4.0, 4.1]},
        "annealed": {"expost": [2480.0, 2510.0, 2550.0], "train_s": [160.0, 162.0, 164.0], "infer_ms": [4.2, 4.4, 4.6]},
        "fixed": {"expost": [2240.0, 2290.0, 2320.0], "train_s": [160.0, 161.0, 162.0], "infer_ms": [4.1, 4.2, 4.3]},
        "batch": {"expost": [2480.0, 2510.0, 2550.0], "train_s": [160.0, 162.0, 164.0], "infer_ms": [4.2, 4.4, 4.6]},
        "step": {"expost": [2110.0, 2140.0, 2190.0], "train_s": [1900.0, 1910.0, 1920.0], "infer_ms": [55.0, 56.0, 57.0]},
    }


def test_build_figure_adds_benchmark_lines_and_summary_stats_without_seed_dots():
    fig = fig_ablation_violins._build_figure(_sample_data())

    try:
        assert round(fig.get_size_inches()[1], 2) == 2.0
        first_ax = fig.axes[0]
        assert len(first_ax.lines) == 2
        assert all(line.get_linestyle() == "-" for line in first_ax.lines)

        labels = {text.get_text(): text for text in first_ax.texts}
        assert {"MIQP-PW", "MIQP-GL"} <= set(labels)
        assert labels["MIQP-PW"].get_position()[1] < fig_ablation_violins.MIQP_PW_MEAN_EXPOST
        assert labels["MIQP-GL"].get_position()[1] < fig_ablation_violins.MIQP_GL_MEAN_EXPOST

        scatter_collections = [
            collection
            for collection in first_ax.collections
            if isinstance(collection, PathCollection)
        ]
        assert not scatter_collections
        line_collections = [
            collection
            for collection in first_ax.collections
            if isinstance(collection, LineCollection)
        ]
        assert len(line_collections) == 16
        stat_texts = [text for text in first_ax.texts if text.get_text().startswith("€")]
        assert len(stat_texts) == 4
        stat_texts.sort(key=lambda text: text.get_position()[0])
        for text, vmax in zip(stat_texts, [2550.0, 2470.0, 2390.0, 2140.0], strict=True):
            assert text.get_position()[1] > vmax
            assert "±" in text.get_text()
        timing_texts = [text for text in first_ax.texts if "trained" in text.get_text()]
        assert len(timing_texts) == 4
        timing_texts.sort(key=lambda text: text.get_position()[0])
        for text, vmin in zip(timing_texts, [2480.0, 2410.0, 2310.0, 2050.0], strict=True):
            assert text.get_position()[1] < vmin
            assert "\n" in text.get_text()
            assert "infer" in text.get_text()
            assert "ms" in text.get_text()
        assert first_ax.get_ylim()[0] == 1250
        assert fig.axes[1].get_ylim()[0] == 1250
        assert fig.axes[2].get_ylim()[0] == 1250
        yticks = first_ax.get_yticks()
        assert len(yticks) >= 5
        assert yticks[0] == 1250
        assert yticks[1] - yticks[0] == 200

        assert first_ax._left_title.get_text() == "(a) Architecture"
        assert fig.axes[1]._left_title.get_text() == "(b) Temperature"
        assert fig.axes[2]._left_title.get_text() == "(c) Dynamics"
        assert [tick.get_text() for tick in fig.axes[1].get_xticklabels()] == ["Annealed", "Fixed"]
    finally:
        plt.close(fig)
