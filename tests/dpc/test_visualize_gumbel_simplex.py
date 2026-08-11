"""Tests for DPC.visualize.fig_gumbel_simplex."""

import matplotlib

matplotlib.use("Agg")

from DPC.visualize import fig_gumbel_simplex


def test_main_writes_pdf(tmp_path):
    out = tmp_path / "gumbel_simplex.pdf"
    result = fig_gumbel_simplex.main(out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0
