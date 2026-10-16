"""
tests/test_coordinates.py
=========================
Coordinate-consistency tests for the (x, y) = (column, row) convention.

All components — Environment, BeliefGrid, Visualizer, ReportGenerator, and
the run.py summary dict — must use the same (x, y) system where:
    x = column index (increases East)
    y = row   index  (increases South)

EVE is placed at the asymmetric position (x=7, y=2) so that any x/y swap
produces a visibly different result (7 ≠ 2).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    Config,
    ClueConfig,
    MapConfig,
    InitialSignalConfig,
    ResourceConfig,
)
from src.environment import Environment
from src.belief import BeliefGrid
from src.visualize import Visualizer
from src.logger import JSONLogger
from src.agent import Agent
from src.report import ReportGenerator

# ---------------------------------------------------------------------------
# EVE's asymmetric position — x=7 (column 7), y=2 (row 2)
# ---------------------------------------------------------------------------
EVE_X = 7
EVE_Y = 2


# ---------------------------------------------------------------------------
# Shared config factory
# ---------------------------------------------------------------------------

def _make_asym_config() -> Config:
    """10×10 grid with EVE at (x=7, y=2) and a clue nearby at (x=7, y=1)."""
    return Config(
        map=MapConfig(width=10, height=10, seed=42, obstacles=[]),
        initial_signal=InitialSignalConfig(direction="NE", strength=0.8, timestamp=0),
        resources=ResourceConfig(
            battery=200,
            time_limit=100,
            move_cost=2,
            scan_cost=5,
            recharge_cost=0,
            recharge_amount=50,
            recharge_stations=[[1, 1]],
        ),
        eve=[EVE_X, EVE_Y],
        clues=[ClueConfig(position=[EVE_X, EVE_Y - 1], type="audio", strength=0.9)],
        sensor_noise=0.0,
        decoys=[],
    )


# ---------------------------------------------------------------------------
# 1. Environment: EVE stored at the correct (x, y)
# ---------------------------------------------------------------------------

class TestEnvironmentCoordinates:
    """Environment must store EVE at exactly (x=7, y=2)."""

    def test_eve_pos_matches_config(self):
        config = _make_asym_config()
        env = Environment(config)
        # Internal _eve_pos is (x, y) tuple
        assert env._eve_pos == (EVE_X, EVE_Y), (
            f"env._eve_pos={env._eve_pos!r}, expected ({EVE_X}, {EVE_Y})"
        )

    def test_agent_finds_eve_at_correct_cell(self, tmp_path):
        """Agent must report confirmed_eve_pos = [EVE_X, EVE_Y] after finding EVE."""
        config = _make_asym_config()
        env = Environment(config)
        belief = BeliefGrid(
            width=10, height=10,
            initial_signal="NE", seed=42, signal_strength=0.8,
        )
        logger = JSONLogger(str(tmp_path / "trace.json"))
        agent = Agent(config=config, environment=env, belief=belief, logger=logger)
        result = agent.run()

        assert result["found_eve"], "Agent should have found EVE in this easy scenario"
        confirmed = result["confirmed_eve_pos"]
        assert confirmed == [EVE_X, EVE_Y], (
            f"confirmed_eve_pos={confirmed!r}, expected [{EVE_X}, {EVE_Y}]"
        )


# ---------------------------------------------------------------------------
# 2. BeliefGrid: collapse_on_found and top_cells honour (x, y)
# ---------------------------------------------------------------------------

class TestBeliefCoordinates:
    """BeliefGrid must use (x=col, y=row) consistently."""

    def test_collapse_on_found_spikes_correct_cell(self):
        bg = BeliefGrid(width=10, height=10, initial_signal="NE", seed=42)
        bg.collapse_on_found([EVE_X, EVE_Y])
        # grid is indexed [row, col] == [y, x]
        assert bg._grid[EVE_Y, EVE_X] > 0.99, (
            f"grid[{EVE_Y},{EVE_X}] should be ~1.0 after collapse, "
            f"got {bg._grid[EVE_Y, EVE_X]:.6f}"
        )
        # The swapped cell should be near zero
        assert bg._grid[EVE_X, EVE_Y] < 1e-6, (
            f"grid[{EVE_X},{EVE_Y}] (swapped) should be ~0, "
            f"got {bg._grid[EVE_X, EVE_Y]:.6f}"
        )

    def test_top_cells_returns_correct_xy(self):
        bg = BeliefGrid(width=10, height=10, initial_signal="NE", seed=42)
        bg.collapse_on_found([EVE_X, EVE_Y])
        top = bg.top_cells(n=1)
        assert len(top) == 1
        x, y, prob = top[0]
        assert x == EVE_X and y == EVE_Y, (
            f"top_cells returned ({x}, {y}), expected ({EVE_X}, {EVE_Y})"
        )
        assert prob > 0.99


# ---------------------------------------------------------------------------
# 3. Visualizer: 'E' appears at row EVE_Y, column EVE_X
# ---------------------------------------------------------------------------

class TestVisualizerCoordinates:
    """The rendered ASCII map must show 'E' at row=EVE_Y, col=EVE_X."""

    def test_eve_marker_at_correct_row_and_column(self):
        config = _make_asym_config()
        env = Environment(config)
        env._found_eve = True  # reveal EVE for rendering

        belief = BeliefGrid(width=10, height=10, initial_signal="NE", seed=42)
        belief.collapse_on_found([EVE_X, EVE_Y])

        viz = Visualizer(
            environment=env,
            belief=belief,
            agent_path=[[0, 0]],
            config=config,
        )
        rendered = viz.render()
        lines = rendered.split("\n")

        # Find the data rows: they start with "{row:2d}|..."
        data_rows = {
            int(line.split("|")[0].strip()): line.split("|", 1)[1]
            for line in lines
            if "|" in line and line.split("|")[0].strip().isdigit()
        }

        assert EVE_Y in data_rows, f"Row {EVE_Y} not found in rendered map"
        row_str = data_rows[EVE_Y]

        assert len(row_str) > EVE_X, (
            f"Row {EVE_Y} is only {len(row_str)} chars wide; "
            f"cannot check column {EVE_X}"
        )
        assert row_str[EVE_X] == "E", (
            f"Expected 'E' at row {EVE_Y}, col {EVE_X} but got "
            f"{row_str[EVE_X]!r}. Full row: {row_str!r}"
        )

        # Also confirm 'E' is NOT at the swapped position (row=EVE_X, col=EVE_Y)
        if EVE_X in data_rows:
            swapped_row = data_rows[EVE_X]
            if len(swapped_row) > EVE_Y:
                assert swapped_row[EVE_Y] != "E", (
                    f"'E' found at swapped position row={EVE_X}, col={EVE_Y} — "
                    "coordinate swap detected!"
                )


# ---------------------------------------------------------------------------
# 4. ReportGenerator: confirmed position shows (EVE_X, EVE_Y)
# ---------------------------------------------------------------------------

class TestReportCoordinates:
    """Report must confirm EVE at (x=EVE_X, y=EVE_Y)."""

    def _run_and_report(self, tmp_path):
        config = _make_asym_config()
        env = Environment(config)
        belief = BeliefGrid(
            width=10, height=10,
            initial_signal="NE", seed=42, signal_strength=0.8,
        )
        logger = JSONLogger(str(tmp_path / "trace.json"))
        agent = Agent(config=config, environment=env, belief=belief, logger=logger)
        result = agent.run()
        return result, logger, config, belief

    def test_report_confirmed_pos_is_correct(self, tmp_path):
        result, logger, config, belief = self._run_and_report(tmp_path)
        assert result["found_eve"], "Precondition: EVE must be found"

        reporter = ReportGenerator(
            config_name="asym_test",
            config=config,
            trace=logger.entries,
        )
        # Internal confirmed_pos should be [EVE_X, EVE_Y]
        assert reporter._confirmed_pos == [EVE_X, EVE_Y], (
            f"ReportGenerator._confirmed_pos={reporter._confirmed_pos!r}, "
            f"expected [{EVE_X}, {EVE_Y}]"
        )

    def test_report_markdown_contains_correct_coords(self, tmp_path):
        result, logger, config, belief = self._run_and_report(tmp_path)
        assert result["found_eve"], "Precondition: EVE must be found"

        reporter = ReportGenerator(
            config_name="asym_test",
            config=config,
            trace=logger.entries,
        )
        out_path = str(tmp_path / "report.md")
        reporter.generate(out_path)

        with open(out_path, encoding="utf-8") as fh:
            content = fh.read()

        expected_coord = f"({EVE_X}, {EVE_Y})"
        assert expected_coord in content, (
            f"Expected '{expected_coord}' in report but not found.\n"
            f"Report content:\n{content}"
        )

    def test_report_confidence_is_100_when_found(self, tmp_path):
        result, logger, config, belief = self._run_and_report(tmp_path)
        assert result["found_eve"], "Precondition: EVE must be found"

        reporter = ReportGenerator(
            config_name="asym_test",
            config=config,
            trace=logger.entries,
        )
        _, max_prob, entropy = reporter._extract_final_belief()
        confidence = reporter._confidence_score(max_prob, entropy)
        assert confidence == 1.0, (
            f"Confidence should be 1.0 when EVE is found, got {confidence}"
        )


# ---------------------------------------------------------------------------
# 5. run.py summary: eve_location_estimate agrees with all components
# ---------------------------------------------------------------------------

class TestSummaryCoordinates:
    """agent.run() summary must report EVE at (x=EVE_X, y=EVE_Y)."""

    def test_eve_estimate_matches_confirmed_pos(self, tmp_path):
        config = _make_asym_config()
        env = Environment(config)
        belief = BeliefGrid(
            width=10, height=10,
            initial_signal="NE", seed=42, signal_strength=0.8,
        )
        logger = JSONLogger(str(tmp_path / "trace.json"))
        agent = Agent(config=config, environment=env, belief=belief, logger=logger)
        result = agent.run()

        assert result["found_eve"]
        assert result["eve_location_estimate"] == [EVE_X, EVE_Y], (
            f"eve_location_estimate={result['eve_location_estimate']!r}, "
            f"expected [{EVE_X}, {EVE_Y}]"
        )
        assert result["confirmed_eve_pos"] == [EVE_X, EVE_Y], (
            f"confirmed_eve_pos={result['confirmed_eve_pos']!r}, "
            f"expected [{EVE_X}, {EVE_Y}]"
        )

    def test_belief_top_cell_agrees_with_summary(self, tmp_path):
        """After a found run, belief top_cell and summary must agree."""
        config = _make_asym_config()
        env = Environment(config)
        belief = BeliefGrid(
            width=10, height=10,
            initial_signal="NE", seed=42, signal_strength=0.8,
        )
        logger = JSONLogger(str(tmp_path / "trace.json"))
        agent = Agent(config=config, environment=env, belief=belief, logger=logger)
        result = agent.run()

        assert result["found_eve"]
        top = belief.top_cells(n=1)
        assert top, "Belief grid returned no top cells"
        bx, by, _ = top[0]
        assert [bx, by] == result["eve_location_estimate"], (
            f"belief top_cell=({bx},{by}) disagrees with "
            f"eve_location_estimate={result['eve_location_estimate']!r}"
        )
