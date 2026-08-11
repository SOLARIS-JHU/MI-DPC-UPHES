"""Emit manual command sets for the DPC ablation runs.

The benchmark tuner still runs the actual experiments. This helper only
generates copy/paste-friendly shell commands for the three manual ablation
families used in the corrected benchmark study.
"""

from __future__ import annotations

import argparse
import shlex

from DPC.experiments.benchmark_data import DEFAULT_EXTREME_DATE


DEFAULT_SEEDS = "0,1,2,3,4"
DEFAULT_INVERSE_PKL = "Data/UPCs/preprocess_inverse_upc.pkl"
BASE_COMMAND = [
    "python",
    "-m",
    "DPC.experiments.benchmark_tuner",
]
BASE_FLAGS = [
    "--pkl",
    "preprocess.pkl",
    "--raw-csv",
    "Data/Belgium.csv",
    "--benchmark-csv",
    "Data/price_data_2024.csv",
    "--year",
    "2024",
    "--extreme-date",
    DEFAULT_EXTREME_DATE,
    "--optimizers",
    "adamw",
    "--schedulers",
    "warmup_cosine",
    "--lr",
    "3e-4",
    "--grad-clip",
    "1.0",
    "--c-op",
    "0.4",
]


def _render_command(*, run_prefix: str, seed_csv: str, extra_flags: list[str]) -> str:
    return shlex.join(BASE_COMMAND + BASE_FLAGS + ["--seeds", seed_csv, "--run-prefix", run_prefix] + extra_flags)


def _render_group(title: str, lines: list[str]) -> str:
    body = "\n\n".join(lines)
    return f"# {title}\n{body}"


def build_ablation_command_sets(seed_csv: str = DEFAULT_SEEDS) -> dict[str, str]:
    architecture = _render_group(
        "Architecture ablation: transformer,mlp,cnn,bilstm",
        [
            "# abl_arch: transformer,mlp,cnn,bilstm | batch dynamics | annealed temperature",
            _render_command(
                run_prefix="abl_arch",
                seed_csv=seed_csv,
                extra_flags=[
                    "--architectures",
                    "transformer,mlp,cnn,bilstm",
                    "--samplers",
                    "cluster_balanced",
                    "--batch-sizes",
                    "32",
                    "--dynamics",
                    "batch",
                    "--tau-start",
                    "10.0",
                    "--tau-ends",
                    "0.08",
                    "--tau-schedules",
                    "two_stage",
                ],
            ),
        ],
    )

    temperature = _render_group(
        "Temperature ablation: annealed vs fixed-low",
        [
            "# abl_tau_annealed: tau_start=10.0 tau_end=0.08 tau_schedule=two_stage",
            _render_command(
                run_prefix="abl_tau_annealed",
                seed_csv=seed_csv,
                extra_flags=[
                    "--architectures",
                    "transformer",
                    "--samplers",
                    "cluster_balanced",
                    "--batch-sizes",
                    "32",
                    "--dynamics",
                    "batch",
                    "--tau-start",
                    "10.0",
                    "--tau-ends",
                    "0.08",
                    "--tau-schedules",
                    "two_stage",
                ],
            ),
            "# abl_tau_fixed: tau_start=0.08 tau_end=0.08 tau_schedule=two_stage",
            _render_command(
                run_prefix="abl_tau_fixed",
                seed_csv=seed_csv,
                extra_flags=[
                    "--architectures",
                    "transformer",
                    "--samplers",
                    "cluster_balanced",
                    "--batch-sizes",
                    "32",
                    "--dynamics",
                    "batch",
                    "--tau-start",
                    "0.08",
                    "--tau-ends",
                    "0.08",
                    "--tau-schedules",
                    "two_stage",
                ],
            ),
        ],
    )

    dynamics = _render_group(
        "Dynamics ablation: batch vs step",
        [
            "# abl_dyn_batch: batch dynamics | annealed temperature",
            _render_command(
                run_prefix="abl_dyn_batch",
                seed_csv=seed_csv,
                extra_flags=[
                    "--architectures",
                    "transformer",
                    "--samplers",
                    "cluster_balanced",
                    "--batch-sizes",
                    "32",
                    "--dynamics",
                    "batch",
                    "--tau-start",
                    "10.0",
                    "--tau-ends",
                    "0.08",
                    "--tau-schedules",
                    "two_stage",
                ],
            ),
            "# abl_dyn_step: step dynamics | annealed temperature | inverse UPC artifact",
            _render_command(
                run_prefix="abl_dyn_step",
                seed_csv=seed_csv,
                extra_flags=[
                    "--architectures",
                    "transformer",
                    "--samplers",
                    "cluster_balanced",
                    "--batch-sizes",
                    "32",
                    "--dynamics",
                    "step",
                    "--inverse-pkl",
                    DEFAULT_INVERSE_PKL,
                    "--tau-start",
                    "10.0",
                    "--tau-ends",
                    "0.08",
                    "--tau-schedules",
                    "two_stage",
                ],
            ),
        ],
    )

    return {
        "architecture": architecture,
        "temperature": temperature,
        "dynamics": dynamics,
    }


def render_script(seed_csv: str = DEFAULT_SEEDS) -> str:
    command_sets = build_ablation_command_sets(seed_csv=seed_csv)
    return "\n\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# Manual DPC ablation command sets",
            "",
            command_sets["architecture"],
            "",
            command_sets["temperature"],
            "",
            command_sets["dynamics"],
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate manual DPC ablation command sets.")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated seed CSV to embed in emitted commands")
    args = parser.parse_args(argv)
    print(render_script(seed_csv=args.seeds), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_INVERSE_PKL",
    "DEFAULT_SEEDS",
    "build_ablation_command_sets",
    "main",
    "render_script",
]
