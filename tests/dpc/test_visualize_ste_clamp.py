"""Tests for DPC.visualize.fig_ste_clamp."""

import matplotlib

matplotlib.use("Agg")

from DPC.visualize import fig_ste_clamp


def test_main_writes_pdf(tmp_path):
    out = tmp_path / "ste_clamp.pdf"
    result = fig_ste_clamp.main(out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0
