from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.figure

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import DPC.visualize.fig_gradient_diagnostics as fig_gradient_diagnostics


def test_main_writes_shorter_gradient_diagnostics_figure(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    monkeypatch.setattr(fig_gradient_diagnostics, "load_ablation_runs_csv", lambda _path: [{"run_dir": "dummy"}])
    monkeypatch.setattr(fig_gradient_diagnostics, "filter_runs", lambda runs, **kwargs: runs)
    monkeypatch.setattr(fig_gradient_diagnostics, "_load_series", lambda bench, rows: [[10.0, 5.0, 2.5], [12.0, 6.0, 3.0]])

    original_savefig = matplotlib.figure.Figure.savefig

    def fake_savefig(self, fname, *args, **kwargs):
        seen["size"] = tuple(self.get_size_inches())
        seen["fname"] = Path(fname)
        Path(fname).write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fake_savefig)

    try:
        fig_gradient_diagnostics.main(output_dir=tmp_path)
    finally:
        monkeypatch.setattr(matplotlib.figure.Figure, "savefig", original_savefig)

    assert seen["fname"] == tmp_path / "gradient_diagnostics.pdf"
    width, height = seen["size"]
    assert width == fig_gradient_diagnostics.COL_WIDTH
    assert height <= 1.6
