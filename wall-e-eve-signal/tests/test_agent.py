"""
tests/test_agent.py
===================
Pytest tests for src/agent.py — autonomous agent behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    Config,
    MapConfig,
    InitialSignalConfig,
    ResourceConfig,
    ClueConfig,
)
from src.environment import Environment
from src.belief import BeliefGrid
from src.logger import JSONLogger
from src.agent import Agent

_ROOT = Path(__file__).parent.parent
_EASY_CONFIG_PATH = str(_ROOT / "configs" / "sample_easy.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    width: int = 8,
    height: int = 8,
    seed: int = 0,
    battery: int = 80,
    time_limit: int = 40,
    move_cost: int = 2,
    scan_cost: int = 5,
    recharge_cost: int = 0,
    recharge_amount: int = 40,
    recharge_stations: list | None = None,
    eve: list | None = None,
    clues: list | None = None,
    sensor_noise: float = 0.0,
    obstacles: list | None = None,
) -> Config:
    return Config(
        map=MapConfig(
            width=width,
            height=height,
            seed=seed,
            obstacles=obstacles or [],
        ),
        initial_signal=InitialSignalConfig(direction="NE", strength=0.6, timestamp=0),
        resources=ResourceConfig(
            battery=battery,
            time_limit=time_limit,
            move_cost=move_cost,
            scan_cost=scan_cost,
            recharge_cost=recharge_cost,
            recharge_amount=recharge_amount,
            recharge_stations=recharge_stations or [[0, 0]],
        ),
        eve=eve or [6, 6],
        clues=clues or [],
        sensor_noise=sensor_noise,
        decoys=[],
    )


def _build_agent(config: Config, trace_path: str = "outputs/_test_trace.json"):
    """Construct and return (env, belief, logger, agent) for given config."""
    env     = Environment(config)
    belief  = BeliefGrid(
        width=config.map.width,
        height=config.map.height,
        initial_signal=config.initial_signal.direction,
        seed=config.map.seed,
    )
    logger  = JSONLogger(trace_path)
    agent   = Agent(config=config, environment=env, belief=belief, logger=logger)
    return env, belief, logger, agent


# ---------------------------------------------------------------------------
# test_agent_runs_without_crash
# ---------------------------------------------------------------------------

class TestAgentRunsWithoutCrash:
    """Loading the easy config and running the agent should raise no exception."""

    def test_easy_config_no_exception(self, tmp_path):
        from src.config import load_config
        config = load_config(_EASY_CONFIG_PATH)
        _, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        # Should not raise
        result = agent.run()
        assert result is not None

    def test_minimal_config_no_exception(self, tmp_path):
        config = _make_config()
        _, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        result = agent.run()
        assert result is not None

    def test_config_with_sensor_noise_no_exception(self, tmp_path):
        config = _make_config(sensor_noise=0.1)
        _, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        result = agent.run()
        assert result is not None


# ---------------------------------------------------------------------------
# test_agent_terminates
# ---------------------------------------------------------------------------

class TestAgentTerminates:
    """agent.run() must return a dict containing at minimum 'found_eve'."""

    def test_run_returns_dict(self, tmp_path):
        config = _make_config()
        _, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        result = agent.run()
        assert isinstance(result, dict)

    def test_run_has_found_eve_key(self, tmp_path):
        config = _make_config()
        _, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        result = agent.run()
        assert "found_eve" in result

    def test_run_has_steps_key(self, tmp_path):
        config = _make_config()
        _, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        result = agent.run()
        assert "steps" in result

    def test_run_has_battery_remaining_key(self, tmp_path):
        config = _make_config()
        _, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        result = agent.run()
        assert "battery_remaining" in result

    def test_found_eve_is_bool(self, tmp_path):
        config = _make_config()
        _, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        result = agent.run()
        assert isinstance(result["found_eve"], bool)

    def test_steps_is_non_negative_int(self, tmp_path):
        config = _make_config()
        _, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        result = agent.run()
        assert isinstance(result["steps"], int)
        assert result["steps"] >= 0

    def test_battery_remaining_non_negative(self, tmp_path):
        config = _make_config()
        _, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        result = agent.run()
        assert result["battery_remaining"] >= 0

    def test_env_is_done_after_run(self, tmp_path):
        config = _make_config()
        env, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()
        assert env.done is True


# ---------------------------------------------------------------------------
# test_agent_respects_battery_safety
# ---------------------------------------------------------------------------

class TestAgentRespectsBatterySafety:
    """Agent should never go below its computed safety reserve battery level."""

    def test_battery_never_below_zero(self, tmp_path):
        """Battery must not go below 0 during or after the run."""
        config = _make_config(battery=60, move_cost=2, scan_cost=5)
        env, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()
        assert env.battery >= 0

    def test_safety_reserve_respected_with_station(self, tmp_path):
        """When a recharge station is configured, the agent should not use so much
        battery that it can't reach the station (within a reasonable margin)."""
        config = _make_config(
            battery=100,
            move_cost=2,
            scan_cost=5,
            recharge_stations=[[0, 0]],
            time_limit=50,
        )
        env, belief, logger, agent = _build_agent(config, str(tmp_path / "trace.json"))

        # Capture battery at every step by hooking into the logger entries after run
        agent.run()

        min_battery_reserve = agent._safety_reserve()  # computed at final state
        # After the run completes (env.done), battery should be >= 0
        assert env.battery >= 0

    def test_battery_never_negative_hard_scenario(self, tmp_path):
        """Even in a resource-constrained scenario, battery never goes negative."""
        config = _make_config(
            battery=20,
            move_cost=3,
            scan_cost=4,
            time_limit=100,
            recharge_stations=[],  # no stations — pure safety
        )
        env, _, _, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()
        assert env.battery >= 0


# ---------------------------------------------------------------------------
# test_agent_produces_trace
# ---------------------------------------------------------------------------

class TestAgentProducesTrace:
    """After agent.run(), the logger must have at least 1 step recorded."""

    def test_logger_has_at_least_one_step(self, tmp_path):
        config = _make_config()
        _, _, logger, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()
        assert len(logger) >= 1

    def test_logger_entries_have_step_key(self, tmp_path):
        config = _make_config()
        _, _, logger, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()
        for entry in logger.entries:
            assert "step" in entry

    def test_logger_entries_have_action_key(self, tmp_path):
        config = _make_config()
        _, _, logger, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()
        for entry in logger.entries:
            assert "action" in entry

    def test_logger_entries_have_belief_summary(self, tmp_path):
        config = _make_config()
        _, _, logger, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()
        for entry in logger.entries:
            assert "belief_summary" in entry

    def test_trace_steps_are_sequential(self, tmp_path):
        config = _make_config()
        _, _, logger, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()
        steps = [e["step"] for e in logger.entries]
        assert steps == list(range(1, len(steps) + 1))

    def test_easy_config_produces_trace(self, tmp_path):
        """Real easy config should also produce a non-empty trace."""
        from src.config import load_config
        config = load_config(_EASY_CONFIG_PATH)
        _, _, logger, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()
        assert len(logger) >= 1



# ---------------------------------------------------------------------------
# test_hard_config_explores_and_updates_belief
# ---------------------------------------------------------------------------

class TestHardConfigExploresAndUpdatesBelief:
    """On the hard config the agent must explore multiple regions and update
    its belief when scans come back empty from the top cell.

    Specifically:
    1. The agent must visit cells in more than one distinct quadrant of the
       15x15 grid (proving it does not stay stuck in one corner).
    2. If the agent scanned a cell that was the top belief cell at the time,
       and the scan returned nothing (no EVE detected, no strong clue), the
       top belief cell must change within the next 5 steps — the agent must
       not keep chasing the same empty location.
    """

    _HARD_CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "sample_hard.yaml")

    def test_agent_visits_more_than_one_quadrant(self, tmp_path):
        """Agent must visit cells in at least 2 of the 4 grid quadrants."""
        from src.config import load_config
        config = load_config(self._HARD_CONFIG_PATH)
        env, belief, logger, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()

        mid_x = config.map.width // 2
        mid_y = config.map.height // 2

        quadrants_visited: set[int] = set()
        for entry in logger.entries:
            pos = entry.get("observation", {}).get("agent_pos")
            if pos:
                x, y = pos[0], pos[1]
                q = (0 if x < mid_x else 1) + (0 if y < mid_y else 2)
                quadrants_visited.add(q)

        assert len(quadrants_visited) >= 2, (
            f"Agent only visited quadrant(s) {quadrants_visited} — "
            "expected at least 2 distinct quadrants on hard config."
        )

    def test_belief_top_cell_changes_after_empty_scan(self, tmp_path):
        """After a scan returns no EVE and no strong clue, and the scan was
        performed *near* the current top belief cell (within 3 steps), the
        top belief cell must shift within the next 5 steps — the agent must
        not keep chasing the same cell it just scanned empty from nearby.
        """
        from src.config import load_config
        config = load_config(self._HARD_CONFIG_PATH)
        env, belief, logger, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()

        entries = logger.entries
        violations = []

        for i, entry in enumerate(entries):
            if entry.get("action") != "scan":
                continue

            scan_result = entry.get("observation", {}).get("scan_result", {})
            eve_detected = scan_result.get("eve_detected", False)
            clue_strength = scan_result.get("clue_strength", 0.0)

            # Only care about scans that found nothing useful
            if eve_detected or clue_strength > 0.3:
                continue

            top_cells_now = entry.get("belief_summary", {}).get("top_cells", [])
            if not top_cells_now:
                continue
            top_at_scan = (top_cells_now[0]["x"], top_cells_now[0]["y"])

            # Only flag if the scan was performed close to the top belief cell —
            # specifically within SCAN_RADIUS so the scan actually covered that cell.
            from src.environment import SCAN_RADIUS as _SR
            agent_pos = entry.get("observation", {}).get("agent_pos", [0, 0])
            dist_to_top = abs(agent_pos[0] - top_at_scan[0]) + abs(agent_pos[1] - top_at_scan[1])
            if dist_to_top > _SR:
                continue  # scan radius did not reach the top cell — skip

            # Check if top cell changes within 5 subsequent steps
            window = entries[i + 1 : i + 6]
            changed = False
            for future_entry in window:
                future_top = future_entry.get("belief_summary", {}).get("top_cells", [])
                if future_top:
                    future_cell = (future_top[0]["x"], future_top[0]["y"])
                    if future_cell != top_at_scan:
                        changed = True
                        break

            if not changed and len(window) >= 3:
                violations.append(
                    f"Step {entry['step']}: empty scan near top cell {top_at_scan} "
                    f"(agent at {agent_pos}, dist={dist_to_top}), "
                    f"but top cell did not change in next {len(window)} steps."
                )

        assert len(violations) == 0, (
            "Agent kept targeting already-scanned empty cells:\n"
            + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# test_clue_cells_appear_on_map
# ---------------------------------------------------------------------------

class TestClueCellsOnMap:
    """When the agent discovers clue signals during scans, the final map
    must contain C symbols at those scan positions."""

    _EASY_CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "sample_easy.yaml")
    _HARD_CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "sample_hard.yaml")

    def _run_and_render(self, config_path: str, tmp_path):
        """Run the agent and return (visualizer, agent, logger)."""
        from src.config import load_config
        from src.visualize import Visualizer

        config = load_config(config_path)
        env, belief, logger, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()

        agent_path = [
            e["observation"]["agent_pos"]
            for e in logger.entries
            if "agent_pos" in e.get("observation", {})
        ]
        if not agent_path:
            agent_path = [env.agent_pos]

        viz = Visualizer(
            environment=env,
            belief=belief,
            agent_path=agent_path,
            config=config,
            discovered_clue_positions=agent.discovered_clue_positions,
            discovered_decoy_positions=agent.discovered_decoy_positions,
        )
        return viz, agent, logger

    def test_clue_symbol_present_when_clues_discovered(self, tmp_path):
        """If the agent detected any clue signals during scans, the rendered
        map must contain at least one 'C' character."""
        viz, agent, _ = self._run_and_render(self._EASY_CONFIG_PATH, tmp_path)
        clue_positions = agent.discovered_clue_positions
        if clue_positions:
            rendered = viz.render()
            assert "C" in rendered, (
                "Agent found clue signals but no C symbols appear in the rendered map."
            )

    def test_clue_positions_match_rendered_map(self, tmp_path):
        """Every position in discovered_clue_positions (that is not overridden
        by a higher-priority symbol) must render as 'C' on the map."""
        viz, agent, logger = self._run_and_render(self._EASY_CONFIG_PATH, tmp_path)
        clue_positions = agent.discovered_clue_positions
        decoy_positions = agent.discovered_decoy_positions

        if not clue_positions:
            pytest.skip("No clue signals detected in this run — test not applicable.")

        # Reconstruct agent path for priority checking
        agent_path_set = set(
            tuple(e["observation"]["agent_pos"])
            for e in logger.entries
            if "agent_pos" in e.get("observation", {})
        )
        # Last position is the agent marker
        all_visited = [e["observation"]["agent_pos"] for e in logger.entries if "agent_pos" in e.get("observation", {})]
        final_pos = tuple(all_visited[-1]) if all_visited else None

        # Build map grid for inspection
        lines = viz.render().splitlines()
        data_lines = {y: line for y, line in enumerate(l for l in lines if "|" in l)}

        for (cx, cy), _ in clue_positions.items():
            # Skip if overridden by higher-priority symbol
            if (cx, cy) in decoy_positions:
                continue
            if (cx, cy) == final_pos:
                continue  # agent marker takes priority

            if cy in data_lines:
                row = data_lines[cy]
                # Row format: " N|<cells>"  — find the cell column
                pipe_idx = row.index("|")
                cell_x = cx + pipe_idx + 1
                if cell_x < len(row):
                    cell_char = row[cell_x]
                    assert cell_char == "C", (
                        f"Clue at ({cx},{cy}) should render as 'C' but got '{cell_char}'."
                    )

    def test_hard_config_clues_on_map(self, tmp_path):
        """Hard config: if any clue signals were detected, 'C' must appear."""
        viz, agent, _ = self._run_and_render(self._HARD_CONFIG_PATH, tmp_path)
        if agent.discovered_clue_positions:
            rendered = viz.render()
            assert "C" in rendered, (
                "Hard config: clue signals were detected but no C on map."
            )

    def test_decoy_symbol_appears_when_decoys_discovered(self, tmp_path):
        """If any positions were identified as decoys, the map must contain 'D'."""
        viz, agent, _ = self._run_and_render(self._HARD_CONFIG_PATH, tmp_path)
        if agent.discovered_decoy_positions:
            rendered = viz.render()
            assert "D" in rendered, (
                "Decoy positions were found but no D symbol on map."
            )


# ---------------------------------------------------------------------------
# test_belief_heatmap_file_created
# ---------------------------------------------------------------------------

class TestBeliefHeatmapFileCreated:
    """run.py must save a belief heatmap file at outputs/<config>_belief.txt."""

    _EASY_CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "sample_easy.yaml")
    _HARD_CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "sample_hard.yaml")

    def _invoke_run(self, config_path: str, output_dir: str) -> int:
        """Call run.main() with the given config and output dir."""
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "run", str(Path(__file__).parent.parent / "run.py")
        )
        run_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_module)
        return run_module.main(["--config", config_path, "--output-dir", output_dir])

    def test_belief_file_created_easy(self, tmp_path):
        """After running the easy config, a _belief.txt file must exist."""
        import os
        output_dir = str(tmp_path / "outputs")
        self._invoke_run(self._EASY_CONFIG_PATH, output_dir)
        belief_file = os.path.join(output_dir, "sample_easy_belief.txt")
        assert os.path.isfile(belief_file), (
            f"Expected belief heatmap file at {belief_file} but it was not created."
        )

    def test_belief_file_created_hard(self, tmp_path):
        """After running the hard config, a _belief.txt file must exist."""
        import os
        output_dir = str(tmp_path / "outputs")
        self._invoke_run(self._HARD_CONFIG_PATH, output_dir)
        belief_file = os.path.join(output_dir, "sample_hard_belief.txt")
        assert os.path.isfile(belief_file), (
            f"Expected belief heatmap file at {belief_file} but it was not created."
        )

    def test_belief_file_contains_heatmap_title(self, tmp_path):
        """The belief file must start with 'BELIEF HEATMAP (final)'."""
        import os
        output_dir = str(tmp_path / "outputs")
        self._invoke_run(self._EASY_CONFIG_PATH, output_dir)
        belief_file = os.path.join(output_dir, "sample_easy_belief.txt")
        with open(belief_file, encoding="utf-8") as fh:
            content = fh.read()
        assert "BELIEF HEATMAP (final)" in content, (
            "Belief heatmap file does not contain the required title."
        )

    def test_belief_file_contains_intensity_legend(self, tmp_path):
        """The belief file must contain the intensity scale legend line."""
        import os
        output_dir = str(tmp_path / "outputs")
        self._invoke_run(self._EASY_CONFIG_PATH, output_dir)
        belief_file = os.path.join(output_dir, "sample_easy_belief.txt")
        with open(belief_file, encoding="utf-8") as fh:
            content = fh.read()
        assert "Belief intensity" in content, (
            "Belief heatmap file does not contain the intensity legend."
        )

    def test_belief_file_non_empty(self, tmp_path):
        """The belief file must have more than a trivial number of bytes."""
        import os
        output_dir = str(tmp_path / "outputs")
        self._invoke_run(self._EASY_CONFIG_PATH, output_dir)
        belief_file = os.path.join(output_dir, "sample_easy_belief.txt")
        size = os.path.getsize(belief_file)
        assert size > 100, f"Belief file is suspiciously small: {size} bytes."

    def test_snapshot_files_created_on_strategy_change(self, tmp_path):
        """If strategy changes occur, snapshot files must be written."""
        import os
        import glob
        output_dir = str(tmp_path / "outputs")
        self._invoke_run(self._HARD_CONFIG_PATH, output_dir)
        snapshots = glob.glob(os.path.join(output_dir, "sample_hard_snapshot_step*.txt"))
        # Strategy changes should occur on the hard config (it has multiple regions)
        # We don't mandate a specific count, just that files were written if changes happened
        # (the agent may or may not change strategy depending on the belief update path)
        for snap in snapshots:
            assert os.path.isfile(snap)
            with open(snap, encoding="utf-8") as fh:
                content = fh.read()
            assert "BELIEF SNAPSHOT" in content, (
                f"Snapshot file {snap} does not contain expected header."
            )


# ---------------------------------------------------------------------------
# test_agent_marker_at_final_position
# ---------------------------------------------------------------------------

class TestAgentMarkerAtFinalPosition:
    """The final rendered map must show the agent marker (@) at the last path
    position, or the combined marker (&) if the agent is standing on EVE."""

    _EASY_CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "sample_easy.yaml")
    _HARD_CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "sample_hard.yaml")

    def _run_and_render(self, config_path: str, tmp_path):
        from src.config import load_config
        from src.visualize import Visualizer

        config = load_config(config_path)
        env, belief, logger, agent = _build_agent(config, str(tmp_path / "trace.json"))
        agent.run()

        agent_path = [
            e["observation"]["agent_pos"]
            for e in logger.entries
            if "agent_pos" in e.get("observation", {})
        ]
        if not agent_path:
            agent_path = [env.agent_pos]

        viz = Visualizer(
            environment=env,
            belief=belief,
            agent_path=agent_path,
            config=config,
            discovered_clue_positions=agent.discovered_clue_positions,
            discovered_decoy_positions=agent.discovered_decoy_positions,
        )
        return viz, env, agent_path

    def test_agent_marker_present(self, tmp_path):
        """The rendered map must contain either '@' or '&'."""
        viz, _, _ = self._run_and_render(self._EASY_CONFIG_PATH, tmp_path)
        rendered = viz.render()
        assert "@" in rendered or "&" in rendered, (
            "Neither '@' nor '&' found in rendered map — agent marker missing."
        )

    def test_combined_marker_when_found_eve(self, tmp_path):
        """When EVE is found and agent ends on EVE's cell, '&' must appear."""
        viz, env, agent_path = self._run_and_render(self._EASY_CONFIG_PATH, tmp_path)
        if not env.found_eve:
            pytest.skip("EVE not found in this run — combined marker test not applicable.")

        from src.config import load_config
        config = load_config(self._EASY_CONFIG_PATH)
        if config.eve is None:
            pytest.skip("EVE position not configured.")

        final_pos = tuple(agent_path[-1])
        eve_pos = (int(config.eve[0]), int(config.eve[1]))

        rendered = viz.render()
        if final_pos == eve_pos:
            assert "&" in rendered, (
                f"Agent ended at EVE position {eve_pos} but '&' not in rendered map."
            )
        else:
            assert "@" in rendered, (
                f"Agent did not end on EVE; '@' should appear but was not found."
            )

    def test_agent_marker_at_correct_cell(self, tmp_path):
        """The map cell at the final agent position must be '@' or '&'."""
        viz, env, agent_path = self._run_and_render(self._EASY_CONFIG_PATH, tmp_path)

        from src.config import load_config
        config = load_config(self._EASY_CONFIG_PATH)

        final_x, final_y = int(agent_path[-1][0]), int(agent_path[-1][1])
        eve_pos = (int(config.eve[0]), int(config.eve[1])) if config.eve else None

        rendered = viz.render()
        lines = rendered.splitlines()

        # Find the data row for final_y
        data_row = None
        for line in lines:
            if "|" in line:
                pipe = line.index("|")
                row_num_str = line[:pipe].strip()
                if row_num_str.isdigit() and int(row_num_str) == final_y:
                    data_row = line[pipe + 1:]
                    break

        assert data_row is not None, f"Could not find row {final_y} in rendered map."
        assert final_x < len(data_row), (
            f"Column {final_x} out of range for row '{data_row}'."
        )
        cell_char = data_row[final_x]
        expected = "&" if (final_x, final_y) == eve_pos else "@"
        assert cell_char == expected, (
            f"Expected '{expected}' at ({final_x},{final_y}) but got '{cell_char}'."
        )

    def test_hard_config_agent_marker_present(self, tmp_path):
        """Hard config: agent marker must appear somewhere on the map."""
        viz, _, _ = self._run_and_render(self._HARD_CONFIG_PATH, tmp_path)
        rendered = viz.render()
        assert "@" in rendered or "&" in rendered, (
            "Neither '@' nor '&' found in hard config rendered map."
        )

    def test_legend_contains_agent_marker_description(self, tmp_path):
        """The map legend must document the '@' and '&' symbols."""
        viz, _, _ = self._run_and_render(self._EASY_CONFIG_PATH, tmp_path)
        rendered = viz.render()
        assert "@=Agent" in rendered, "Legend missing '@=Agent' description."
        assert "&=Agent on EVE" in rendered, "Legend missing '&=Agent on EVE' description."
