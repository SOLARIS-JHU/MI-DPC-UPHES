from __future__ import annotations

import importlib
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import DPC.visualize.style as style


@pytest.fixture(autouse=True)
def _reset_theme(monkeypatch):
    monkeypatch.delenv(style.THEME_ENV, raising=False)
    with plt.rc_context():
        yield
    style.apply_theme("paper")


def test_paper_theme_leaves_rcparams_untouched():
    before = dict(plt.rcParams)
    assert style.apply_theme("paper") == "paper"
    assert dict(plt.rcParams) == before


def test_themed_returns_paper_value_by_default():
    style.apply_theme("paper")
    assert style.themed("black", "fg") == "black"
    assert style.grid_kw() == style.GRID_KW


def test_web_light_is_transparent_but_keeps_paper_colours():
    style.apply_theme("web-light")
    assert plt.rcParams["savefig.transparent"] is True
    assert style.themed("black", "fg") == "black"


def test_web_dark_swaps_neutral_colours():
    style.apply_theme("web-dark")
    assert plt.rcParams["text.color"] == "#e8ecf3"
    assert plt.rcParams["axes.edgecolor"] == "#e8ecf3"
    assert style.themed("black", "fg") == "#e8ecf3"
    assert style.themed("0.6", "faint") == "#6b7893"
    assert style.grid_kw()["color"] == "#2a3a5c"
    assert style.theme_background() == "#15223d"


def test_theme_comes_from_environment(monkeypatch):
    monkeypatch.setenv(style.THEME_ENV, "web-dark")
    assert style.apply_theme() == "web-dark"
    assert style.current_theme() == "web-dark"


def test_unknown_theme_is_rejected():
    with pytest.raises(ValueError, match="Unknown figure theme"):
        style.apply_theme("neon")


def test_bench_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MIDPC_BENCH_DIR", str(tmp_path))
    reloaded = importlib.reload(style)
    try:
        assert Path(reloaded.BENCH_DIR) == tmp_path
    finally:
        monkeypatch.delenv("MIDPC_BENCH_DIR")
        importlib.reload(style)


def test_resolve_run_dir_falls_back_to_cache_parent(tmp_path):
    from DPC.visualize.epoch_replay import resolve_run_dir

    cache = tmp_path / "run" / "epoch_dispatch_trace.npz"
    cache.parent.mkdir()
    cache.touch()
    assert resolve_run_dir(cache, {"run_dir": "/nonexistent/old/location"}) == cache.parent
    assert resolve_run_dir(cache, {"run_dir": str(tmp_path)}) == tmp_path


def test_by_theme_picks_the_dark_value_only_under_web_dark():
    style.apply_theme("paper")
    assert style.by_theme("#D62728", "#ff6b6b") == "#D62728"
    style.apply_theme("web-light")
    assert style.by_theme("#D62728", "#ff6b6b") == "#D62728"
    style.apply_theme("web-dark")
    assert style.by_theme("#D62728", "#ff6b6b") == "#ff6b6b"
