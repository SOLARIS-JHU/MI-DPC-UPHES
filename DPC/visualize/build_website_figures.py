"""Render every figure of the project page in the light and dark web themes.

Each figure script runs in a subprocess with MIDPC_FIG_THEME set, writes its
usual PDF into a temporary directory, and the PDF is rasterized to a
transparent PNG with pdftocairo. Animations are written directly as GIFs.

Figures that read per-seed run outputs need --bench-dir (a benchmark_suite
directory that still has the run folders) and --cache (a replay cache).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
THEMES = {"web-light": "", "web-dark": "_dark"}
DPI = 300

# (output stem, PDF written by the script, python -c body; {out} and {cache} are substituted)
PDF_FIGURES = [
    ("ablation_violins", "ablation_violins.pdf",
     "import sys; from DPC.visualize import fig_ablation_violins as m; sys.argv=['x','--output-dir',r'{out}']; m.main()"),
    ("gumbel_simplex", "gumbel_simplex.pdf",
     "from DPC.visualize import fig_gumbel_simplex as m; m.main(r'{out}/gumbel_simplex.pdf')"),
    ("ste_clamp", "ste_clamp.pdf",
     "from DPC.visualize import fig_ste_clamp as m; m.main(r'{out}/ste_clamp.pdf')"),
    ("profit_distribution", "profit_distribution.pdf",
     "import pathlib; from DPC.visualize import fig_profit_distribution as m; m.make_figure(pathlib.Path(r'{out}'))"),
    ("schedule_evolution", "schedule_evolution.pdf",
     "from DPC.visualize import fig_schedule_evolution as m; m.make_figure(r'{out}', cache_path=r'{cache}')"),
]
GIFS = [
    ("schedule_evolution", ["-m", "DPC.visualize.anim_schedule_evolution", "--cache", "{cache}", "--output", "{gif}"]),
    ("gumbel_simplex", ["-m", "DPC.visualize.anim_gumbel_simplex", "--output", "{gif}"]),
    # --still writes the final frame as the reduced-motion fallback
    ("training_run", ["-m", "DPC.visualize.anim_training_run", "--cache", "{cache}", "--output", "{gif}", "--still", "{still}"]),
]


def _run(args: list[str], env: dict[str, str]) -> None:
    subprocess.run([sys.executable, *args], cwd=REPO_ROOT, env=env, check=True)


def build(out_dir: Path, *, bench_dir: Path, cache: Path, only: set[str] | None = None) -> list[Path]:
    if shutil.which("pdftocairo") is None:
        raise RuntimeError("pdftocairo (poppler-utils) is required to rasterize the figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for theme, suffix in THEMES.items():
        env = {**os.environ, "MIDPC_FIG_THEME": theme, "MIDPC_BENCH_DIR": str(bench_dir)}
        with tempfile.TemporaryDirectory() as tmp:
            for stem, pdf_name, body in PDF_FIGURES:
                if only and stem not in only:
                    continue
                _run(["-c", body.format(out=tmp, cache=cache)], env)
                target = out_dir / f"{stem}{suffix}"
                subprocess.run(
                    ["pdftocairo", "-png", "-transp", "-singlefile", "-r", str(DPI), str(Path(tmp) / pdf_name), str(target)],
                    check=True,
                )
                written.append(target.with_suffix(".png"))
        for stem, args in GIFS:
            if only and stem not in only:
                continue
            gif = out_dir / f"{stem}{suffix}.gif"
            _run([a.format(cache=cache, gif=gif, still=gif.with_suffix(".png")) for a in args], env)
            written.append(gif)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "website" / "static")
    parser.add_argument("--bench-dir", type=Path, required=True, help="benchmark_suite directory containing the per-seed run folders")
    parser.add_argument("--cache", type=Path, required=True, help="epoch_dispatch_trace.npz replay cache")
    parser.add_argument("--only", nargs="*", help="Subset of figure stems to rebuild")
    args = parser.parse_args(argv)
    for path in build(args.out, bench_dir=args.bench_dir, cache=args.cache, only=set(args.only) if args.only else None):
        print(f"{path.stat().st_size:>10,}  {path}")


if __name__ == "__main__":
    main()
