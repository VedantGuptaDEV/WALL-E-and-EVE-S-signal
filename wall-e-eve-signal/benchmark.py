#!/usr/bin/env python3
"""
benchmark.py
============
Batch benchmark script for the WALL-E EVE Signal mission simulator.

Runs N independent missions with different RNG seeds, collects statistics,
and writes a JSON results file.

Usage
-----
    python benchmark.py --config configs/sample_easy.yaml --runs 20
    python benchmark.py --config configs/sample_hard.yaml --runs 50 --output results/bench.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config, Config
from src.environment import Environment
from src.belief import BeliefGrid
from src.agent import Agent
from src.logger import JSONLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_single(config: Config) -> dict:
    """Run one mission with *config* and return a result dict.

    Parameters
    ----------
    config:
        Fully-validated :class:`~src.config.Config` for this run.

    Returns
    -------
    dict with keys: seed, found_eve, steps, battery_remaining, confidence
    """
    env = Environment(config)
    belief = BeliefGrid(
        width=config.map.width,
        height=config.map.height,
        initial_signal=config.initial_signal.direction,
        seed=config.map.seed,
    )
    # Use a null logger (writes to /dev/null-equivalent temp path) to avoid
    # flooding the filesystem during bulk runs.
    null_trace = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs",
        "_bench_tmp.json",
    )
    logger = JSONLogger(null_trace)

    agent = Agent(config=config, environment=env, belief=belief, logger=logger)
    result = agent.run()

    # Confidence: probability of the top belief cell
    top_cells = belief.top_cells(n=1)
    confidence = top_cells[0][2] if top_cells else 0.0

    return {
        "seed": config.map.seed,
        "found_eve": result["found_eve"],
        "steps": result["steps"],
        "battery_remaining": result["battery_remaining"],
        "confidence": round(confidence, 6),
    }


def _patch_seed(config: Config, new_seed: int) -> Config:
    """Return a shallow-copied Config with map.seed replaced by *new_seed*.

    We don't deep-copy the entire dataclass graph — only the map field is
    touched since that's where the seed lives.
    """
    import dataclasses

    new_map = dataclasses.replace(config.map, seed=new_seed)
    return dataclasses.replace(config, map=new_map)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description="Batch benchmark for the WALL-E EVE Signal mission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmark.py --config configs/sample_easy.yaml --runs 20\n"
            "  python benchmark.py --config configs/sample_hard.yaml --runs 50 "
            "--output results/bench.json\n"
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Base config file (YAML or JSON).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        metavar="N",
        help="Number of independent runs (default: 20).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("outputs", "benchmark_results.json"),
        metavar="PATH",
        help="Output JSON file path (default: outputs/benchmark_results.json).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write results."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path: str = args.config
    num_runs: int    = args.runs
    output_path: str = args.output

    # ------------------------------------------------------------------
    # Load base config
    # ------------------------------------------------------------------
    try:
        base_config = load_config(config_path)
    except FileNotFoundError as exc:
        print(f"[error] Config file not found: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[error] Invalid configuration: {exc}", file=sys.stderr)
        return 1

    base_seed: int = base_config.map.seed

    print(
        f"[benchmark] Config: {config_path}  "
        f"base_seed={base_seed}  runs={num_runs}"
    )
    print(f"[benchmark] Output: {output_path}")
    print("-" * 60)

    # ------------------------------------------------------------------
    # Run benchmark
    # ------------------------------------------------------------------
    results: list[dict] = []

    for i in range(num_runs):
        seed_i = base_seed + i
        config_i = _patch_seed(base_config, seed_i)

        try:
            run_result = _run_single(config_i)
        except Exception as exc:
            print(f"  Run {i+1:3d}/{num_runs}  seed={seed_i}  ERROR: {exc}", file=sys.stderr)
            run_result = {
                "seed": seed_i,
                "found_eve": False,
                "steps": 0,
                "battery_remaining": 0,
                "confidence": 0.0,
                "error": str(exc),
            }

        results.append(run_result)

        status = "[OK]" if run_result["found_eve"] else "[--]"
        print(
            f"  Run {i+1:3d}/{num_runs}  seed={seed_i}  {status}  "
            f"steps={run_result['steps']:4d}  "
            f"battery={run_result['battery_remaining']:4d}  "
            f"conf={run_result['confidence']:.4f}"
        )

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    n_found  = sum(1 for r in results if r.get("found_eve", False))
    n_total  = len(results)

    success_rate      = n_found / n_total if n_total > 0 else 0.0
    avg_steps         = sum(r["steps"] for r in results) / n_total if n_total else 0.0
    avg_battery       = sum(r["battery_remaining"] for r in results) / n_total if n_total else 0.0
    avg_confidence    = sum(r["confidence"] for r in results) / n_total if n_total else 0.0

    summary = {
        "config": config_path,
        "base_seed": base_seed,
        "num_runs": n_total,
        "success_rate":          round(success_rate, 4),
        "avg_steps":             round(avg_steps, 2),
        "avg_battery_remaining": round(avg_battery, 2),
        "avg_confidence":        round(avg_confidence, 6),
        "runs": results,
    }

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Total runs        : {n_total}")
    print(f"  Successful (found): {n_found}  ({success_rate * 100:.1f}%)")
    print(f"  Avg steps         : {avg_steps:.1f}")
    print(f"  Avg battery left  : {avg_battery:.1f}")
    print(f"  Avg confidence    : {avg_confidence:.4f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\n[benchmark] Results saved → {os.path.abspath(output_path)}")
    except OSError as exc:
        print(f"[warning] Could not save results: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
