"""CLI for merging real + synthetic (DR) LeRobotDatasets.

Thin argparse wrapper around :func:`merge_utilities.merge_datasets` so the
documented ``python -m lerobot_isaac_synthetic.merge`` command works (B.1
Phase 4 of the DR100 plan).

Example::

    python -m lerobot_isaac_synthetic.merge \\
        --real datasets/kvgork/so101-pickplace1 \\
        --sim  datasets/kvgork/so101-pickplace1-dr100 \\
        --out  datasets/kvgork/so101-pickplace1-dr100-merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lerobot_isaac_synthetic.merge",
        description="Merge a real LeRobotDataset with one or more synthetic (DR) datasets.",
    )
    p.add_argument("--real", required=True, help="Path to the real LeRobotDataset dir.")
    p.add_argument(
        "--sim",
        action="append",
        required=True,
        help="Path to a synthetic dataset dir (repeatable for multiple).",
    )
    p.add_argument("--out", required=True, help="Output path for the merged dataset.")
    p.add_argument(
        "--sim-weight",
        type=float,
        default=0.5,
        help="Fraction of merged episodes that are synthetic (0,1). Default 0.5.",
    )
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable near-duplicate synthetic-episode dropping.",
    )
    p.add_argument("--task-name", default="pick_and_place")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved merge call and exit without writing.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    real = Path(args.real)
    sims = [Path(s) for s in args.sim]
    out = Path(args.out)

    if args.dry_run:
        print("merge dry-run — resolved parameters:")
        print(f"  real       : {real}")
        print(f"  sim        : {[str(s) for s in sims]}")
        print(f"  out        : {out}")
        print(f"  sim_weight : {args.sim_weight}")
        print(f"  dedup      : {not args.no_dedup}")
        print(f"  task_name  : {args.task_name}")
        return 0

    from lerobot_isaac_synthetic.merge_utilities import merge_datasets

    result = merge_datasets(
        real_path=real,
        sim_paths=sims,
        output_path=out,
        sim_weight=args.sim_weight,
        dedup=not args.no_dedup,
        task_name=args.task_name,
        fps=args.fps,
    )
    print(f"merged dataset written → {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
