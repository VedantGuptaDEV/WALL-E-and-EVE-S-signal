#!/usr/bin/env python3
"""
run.py
======
CLI entrypoint for the WALL-E EVE Signal mission simulator.

Usage
-----
    python run.py --config configs/sample_easy.yaml
    python run.py --config configs/sample_hard.yaml --output-dir results --verbose
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.environment import Environment
from src.belief import BeliefGrid
from src.agent import Agent
from src.logger import JSONLogger
from src.report import ReportGenerator
from src.visualize import Visualizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="WALL-E EVE Signal Mission Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run.py --config configs/sample_easy.yaml\n"
            "  python run.py --config configs/sample_hard.yaml --output-dir results --verbose\n"
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML (or JSON) mission configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        metavar="DIR",
        help="Directory where output files are written (default: outputs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress and intermediate snapshots.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run a single mission and save all outputs.

    Parameters
    ----------
    argv:
        Command-line arguments.  Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path: str = args.config
    output_dir: str = args.output_dir
    verbose: bool = args.verbose

    # Derive a clean name from the config filename (no extension)
    config_name: str = Path(config_path).stem

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    try:
        if verbose:
            print(f"[run] Loading config: {config_path}")
        config = load_config(config_path)
    except FileNotFoundError as exc:
        print(f"[error] Config file not found: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[error] Invalid configuration: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[error] Failed to load config: {exc}", file=sys.stderr)
        return 1

    if verbose:
        print(
            f"[run] Config loaded — grid {config.map.width}x{config.map.height}, "
            f"seed={config.map.seed}, battery={config.resources.battery}"
        )

    # ------------------------------------------------------------------
    # 2. Build environment, belief, logger, agent
    # ------------------------------------------------------------------
    try:
        env = Environment(config)
        belief = BeliefGrid(
            width=config.map.width,
            height=config.map.height,
            initial_signal=config.initial_signal.direction,
            seed=config.map.seed,
            signal_strength=config.initial_signal.strength,
        )

        trace_path = os.path.join(output_dir, f"{config_name}_trace.json")
        logger = JSONLogger(trace_path)

        agent = Agent(config=config, environment=env, belief=belief, logger=logger)
    except Exception as exc:
        print(f"[error] Initialisation failed: {exc}", file=sys.stderr)
        return 1

    if verbose:
        print(f"[run] Agent initialised.  Running mission …")

    # ------------------------------------------------------------------
    # 3. Register strategy-change snapshot callbacks
    # ------------------------------------------------------------------
    # We need a reference to the visualizer for callbacks, but we build it
    # after the run.  Instead we collect (step, strategy) pairs during the
    # run and write snapshots after building the Visualizer.
    strategy_changes: list[tuple[int, str]] = []

    def _on_strategy_change(step: int, new_strategy: str) -> None:
        strategy_changes.append((step, new_strategy))

    agent.register_strategy_change_callback(_on_strategy_change)

    # ------------------------------------------------------------------
    # 4. Run agent
    # ------------------------------------------------------------------
    try:
        result = agent.run()
    except Exception as exc:
        print(f"[error] Mission run failed: {exc}", file=sys.stderr)
        return 1

    found_eve:       bool           = result["found_eve"]
    steps:           int            = result["steps"]
    battery_left:    int            = result["battery_remaining"]
    eve_estimate:    list | None    = result.get("eve_location_estimate")

    if verbose:
        print(
            f"[run] Mission complete — "
            f"found_eve={found_eve}, steps={steps}, battery_remaining={battery_left}"
        )

    # ------------------------------------------------------------------
    # 5. Save JSON trace
    # ------------------------------------------------------------------
    try:
        saved_trace = logger.save()
        if verbose:
            print(f"[run] Trace saved -> {saved_trace}")
    except Exception as exc:
        print(f"[warning] Could not save trace: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 6. Generate and save Markdown report
    # ------------------------------------------------------------------
    try:
        report_path = os.path.join(output_dir, f"{config_name}_report.md")
        reporter = ReportGenerator(
            config_name=config_name,
            config=config,
            trace=logger.entries,
        )
        reporter.generate(report_path)
        if verbose:
            print(f"[run] Report saved -> {os.path.abspath(report_path)}")
    except Exception as exc:
        print(f"[warning] Could not generate report: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 7. Reconstruct agent path from trace
    # ------------------------------------------------------------------
    agent_path: list[list[int]] = []
    for entry in logger.entries:
        pos = entry.get("observation", {}).get("agent_pos")
        if pos:
            agent_path.append(pos)

    if not agent_path:
        agent_path = [env.agent_pos]

    # ------------------------------------------------------------------
    # 8. Build Visualizer — pass discovered clue/decoy positions from agent
    # ------------------------------------------------------------------
    try:
        viz = Visualizer(
            environment=env,
            belief=belief,
            agent_path=agent_path,
            config=config,
            discovered_clue_positions=agent.discovered_clue_positions,
            discovered_decoy_positions=agent.discovered_decoy_positions,
        )
    except Exception as exc:
        print(f"[warning] Visualizer init failed: {exc}", file=sys.stderr)
        viz = Visualizer(
            environment=env,
            belief=belief,
            agent_path=agent_path,
            config=config,
        )

    # ------------------------------------------------------------------
    # 9. Save and print the final ASCII map (Issue 4: shows @ and &)
    # ------------------------------------------------------------------
    try:
        map_path = os.path.join(output_dir, f"{config_name}_map.txt")
        viz.save(map_path)

        print("\n" + "=" * (config.map.width + 6))
        print("  FINAL MAP")
        print("=" * (config.map.width + 6))
        print(viz.render())
        print("=" * (config.map.width + 6) + "\n")

        if verbose:
            print(f"[run] ASCII map saved -> {os.path.abspath(map_path)}")
    except Exception as exc:
        print(f"[warning] Visualization failed: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 10. Print and save belief heatmap (Issue 3)
    # ------------------------------------------------------------------
    try:
        belief_path = os.path.join(output_dir, f"{config_name}_belief.txt")
        viz.save_belief_heatmap(belief_path)

        print("=" * (config.map.width + 6))
        print("  BELIEF HEATMAP (final)")
        print("=" * (config.map.width + 6))
        print(viz.render_belief_heatmap())
        print("=" * (config.map.width + 6) + "\n")

        if verbose:
            print(f"[run] Belief heatmap saved -> {os.path.abspath(belief_path)}")
    except Exception as exc:
        print(f"[warning] Belief heatmap failed: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 11. Save strategy-change belief snapshots (Issue 3 — snapshots)
    # ------------------------------------------------------------------
    try:
        for snap_step, snap_strategy in strategy_changes:
            snap_path = os.path.join(
                output_dir,
                f"{config_name}_snapshot_step{snap_step:04d}.txt",
            )
            viz.save_snapshot(snap_path, step=snap_step, strategy=snap_strategy)
            if verbose:
                print(
                    f"[run] Snapshot saved -> {snap_path}  "
                    f"(step={snap_step} strategy={snap_strategy})"
                )
        if strategy_changes:
            print(
                f"[run] {len(strategy_changes)} strategy-change snapshot(s) saved "
                f"to {output_dir}/"
            )
    except Exception as exc:
        print(f"[warning] Snapshot saving failed: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 12. Print mission summary to stdout
    # ------------------------------------------------------------------
    import math as _math
    top_cells = belief.top_cells(n=1)
    if found_eve:
        confidence = 1.0
        confidence_label = "1.0000 [CONFIRMED]"
    else:
        max_prob = top_cells[0][2] if top_cells else 0.0
        entropy  = belief.entropy()
        n        = config.map.width * config.map.height
        uniform  = 1.0 / max(n, 1)
        prob_comp = min(1.0, max_prob / (10.0 * uniform))
        max_H     = _math.log(max(n, 2))
        ent_comp  = max(0.0, 1.0 - entropy / max_H)
        confidence = round(0.5 * prob_comp + 0.5 * ent_comp, 4)
        confidence_label = f"{confidence:.4f}"

    # Report clue scan statistics
    clue_scans = agent.discovered_clue_positions
    decoy_scans = agent.discovered_decoy_positions
    total_scans = sum(1 for e in logger.entries if e.get("action") == "scan")

    print("=" * 50)
    print("  MISSION SUMMARY")
    print("=" * 50)
    print(f"  Found EVE        : {'YES [FOUND]' if found_eve else 'NO [NOT FOUND]'}")
    print(f"  Steps taken      : {steps}")
    print(f"  Battery remaining: {battery_left}")
    print(f"  Confidence score : {confidence_label}")
    print(f"  Total scans      : {total_scans}")
    print(f"  Clue signals     : {len(clue_scans)} scan positions returned clue")
    print(f"  Decoys found     : {len(decoy_scans)} positions identified as decoys")
    print(f"  Strategy changes : {len(strategy_changes)}")
    if eve_estimate:
        label = "EVE location     :"
        if found_eve:
            print(f"  {label} {eve_estimate} [CONFIRMED]")
        else:
            print(f"  {label} {eve_estimate} (estimate)")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
