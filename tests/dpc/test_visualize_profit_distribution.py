from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import DPC.visualize.fig_profit_distribution as fig_profit_distribution


def _bimodal() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(1500.0, 300.0, 10), rng.normal(4000.0, 300.0, 9)])


def test_density_curve_spans_exactly_the_observed_range():
    values = _bimodal()
    x, density = fig_profit_distribution.density_curve(values)
    assert x[0] == values.min() and x[-1] == values.max()
    assert x.shape == density.shape == (500,)
    assert np.all(density >= 0.0)


def test_density_curve_uses_half_the_scott_bandwidth():
    from scipy.stats import gaussian_kde

    values = _bimodal()
    x, density = fig_profit_distribution.density_curve(values)
    reference = gaussian_kde(values, bw_method="scott")
    reference.set_bandwidth(reference.factor * 0.5)
    assert np.allclose(density, reference(x))
    # Narrower than plain Scott: the two modes are resolved more sharply.
    assert density.max() > gaussian_kde(values, bw_method="scott")(x).max()


def test_build_figure_has_one_panel_per_baseline_with_mean_lines():
    days = _bimodal()
    fig = fig_profit_distribution.build_figure(midpc=days + 50.0, miqp_gl=days - 400.0, miqp_pw=days + 80.0)
    try:
        assert len(fig.axes) == 2
        assert [ax.get_title() for ax in fig.axes] == ["(a) MI-DPC vs. MIQP-GL", "(b) MI-DPC vs. MIQP-PW"]
        for ax in fig.axes:
            labels = [t.get_text() for t in ax.get_legend().get_texts()]
            assert len(labels) == 2 and labels[0].startswith("MI-DPC (€")
            assert all("±" in label for label in labels)
    finally:
        plt.close(fig)
