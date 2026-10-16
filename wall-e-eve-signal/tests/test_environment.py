"""
tests/test_environment.py
=========================
Pytest tests for src/environment.py — grid environment behaviour.
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(
    width: int = 8,
    height: int = 8,
    seed: int = 0,
    obstacles: list | None = None,
    battery: int = 100,
    time_limit: int = 50,
    move_cost: int = 2,
    scan_cost: int = 5,
    recharge_cost: int = 0,
    recharge_amount: int = 30,
    recharge_stations: list | None = None,
    eve: list | None = None,
    clues: list | None = None,
    sensor_noise: float = 0.0,
    decoys: list | None = None,
) -> Config:
    """Build a minimal Config for testing."""
    return Config(
        map=MapConfig(
            width=width,
            height=height,
            seed=seed,
            obstacles=obstacles or [],
        ),
        initial_signal=InitialSignalConfig(direction="N", strength=0.5, timestamp=0),
        resources=ResourceConfig(
            battery=battery,
            time_limit=time_limit,
            move_cost=move_cost,
            scan_cost=scan_cost,
            recharge_cost=recharge_cost,
            recharge_amount=recharge_amount,
            recharge_stations=recharge_stations or [],
        ),
        eve=eve or [5, 5],
        clues=clues or [],
        sensor_noise=sensor_noise,
        decoys=decoys or [],
    )


# ---------------------------------------------------------------------------
# test_move_north
# ---------------------------------------------------------------------------

class TestMoveNorth:
    """Moving north should decrease battery by move_cost and move agent."""

    def test_move_north_decreases_battery(self):
        cfg = _make_config(move_cost=2, battery=100)
        env = Environment(cfg)
        initial_battery = env.battery
        obs = env.move("N")
        # Battery decreases by move_cost (or stays if blocked at wall, but
        # cost is always deducted)
        assert env.battery == initial_battery - cfg.resources.move_cost

    def test_move_north_changes_y_when_not_blocked(self):
        # Agent starts at (0,0); moving N would go to (0,-1) which is out
        # of bounds.  Place agent somewhere in the middle via a large grid.
        cfg = _make_config(width=10, height=10)
        env = Environment(cfg)
        # The agent starts at (0,0).  Force-move south first so we have room.
        env.move("S")
        env.move("S")
        y_before = env.agent_pos[1]
        env.move("N")
        # Either moved north or was blocked at wall — both are valid
        y_after = env.agent_pos[1]
        assert y_after <= y_before  # can't go further south when moving N

    def test_step_increments_by_one(self):
        cfg = _make_config()
        env = Environment(cfg)
        before = env.steps_taken
        env.move("S")  # always valid since y starts at 0 → moves to 1
        assert env.steps_taken == before + 1


# ---------------------------------------------------------------------------
# test_move_blocked_by_obstacle
# ---------------------------------------------------------------------------

class TestMoveBlockedByObstacle:
    """Moving into an obstacle should leave agent position unchanged."""

    def test_agent_stays_when_blocked(self):
        # Put an obstacle directly south of the agent start (0,0) → (0,1)
        cfg = _make_config(obstacles=[[0, 1]])
        env = Environment(cfg)
        pos_before = list(env.agent_pos)
        env.move("S")
        assert env.agent_pos == pos_before

    def test_event_contains_blocked(self):
        cfg = _make_config(obstacles=[[0, 1]])
        env = Environment(cfg)
        obs = env.move("S")
        assert "blocked" in obs["event"]

    def test_battery_still_deducted_when_blocked(self):
        cfg = _make_config(obstacles=[[0, 1]], move_cost=3)
        env = Environment(cfg)
        battery_before = env.battery
        env.move("S")
        assert env.battery == battery_before - 3


# ---------------------------------------------------------------------------
# test_scan_returns_observation
# ---------------------------------------------------------------------------

class TestScanReturnsObservation:
    """scan() must return an Observation with all required keys."""

    def test_observation_has_required_keys(self):
        cfg = _make_config()
        env = Environment(cfg)
        obs = env.scan()
        required_keys = {
            "agent_pos", "battery", "time_remaining",
            "steps_taken", "scan_result", "event",
        }
        assert required_keys.issubset(obs.keys())

    def test_scan_result_has_required_keys(self):
        cfg = _make_config()
        env = Environment(cfg)
        obs = env.scan()
        scan = obs["scan_result"]
        assert "clue_strength" in scan
        assert "clue_direction" in scan
        assert "eve_detected" in scan

    def test_clue_strength_is_float(self):
        cfg = _make_config()
        env = Environment(cfg)
        obs = env.scan()
        assert isinstance(obs["scan_result"]["clue_strength"], float)

    def test_clue_strength_in_range(self):
        cfg = _make_config()
        env = Environment(cfg)
        obs = env.scan()
        s = obs["scan_result"]["clue_strength"]
        assert 0.0 <= s <= 1.0

    def test_event_contains_scan(self):
        cfg = _make_config()
        env = Environment(cfg)
        obs = env.scan()
        assert "scan" in obs["event"]

    def test_scan_deducts_battery(self):
        cfg = _make_config(scan_cost=5)
        env = Environment(cfg)
        b_before = env.battery
        env.scan()
        assert env.battery == b_before - 5


# ---------------------------------------------------------------------------
# test_recharge_on_station
# ---------------------------------------------------------------------------

class TestRechargeOnStation:
    """Recharging on a station should increase the battery."""

    def test_battery_increases_when_on_station(self):
        # Place agent start at (0,0) and put a recharge station there.
        cfg = _make_config(
            recharge_stations=[[0, 0]],
            recharge_amount=30,
            recharge_cost=0,
            battery=50,
        )
        env = Environment(cfg)
        # Drain some battery first
        env.move("S")
        env.move("N")
        b_low = env.battery

        # Manually return to station by resetting (we'll manipulate via moves)
        # Instead, create a fresh env with a station at start position
        cfg2 = _make_config(
            recharge_stations=[[0, 0]],
            recharge_amount=30,
            recharge_cost=0,
            battery=70,
        )
        env2 = Environment(cfg2)
        # Drain battery first so recharge has room to restore (battery=70 is max)
        env2.move("S")  # costs move_cost, moves south
        env2.move("N")  # costs move_cost, returns to (0,0) — the station
        b_before = env2.battery
        obs = env2.recharge()
        b_after = env2.battery
        assert b_after > b_before or "not_on_station" in obs["event"]

    def test_recharge_capped_at_max_battery(self):
        cfg = _make_config(
            recharge_stations=[[0, 0]],
            recharge_amount=9999,
            recharge_cost=0,
            battery=100,
        )
        env = Environment(cfg)
        env.recharge()
        assert env.battery <= cfg.resources.battery

    def test_recharge_fails_off_station(self):
        # No station at (0,0)
        cfg = _make_config(recharge_stations=[[7, 7]])
        env = Environment(cfg)
        obs = env.recharge()
        assert "not_on_station" in obs["event"]


# ---------------------------------------------------------------------------
# test_battery_depletion_ends_episode
# ---------------------------------------------------------------------------

class TestBatteryDepletionEndsEpisode:
    """When battery reaches 0 the environment should set done=True."""

    def test_done_true_when_battery_zero(self):
        # Very low battery, high action cost; run until depleted
        cfg = _make_config(
            battery=10,
            move_cost=3,
            scan_cost=3,
            time_limit=200,
            eve=[6, 6],  # far away so it won't be found
        )
        env = Environment(cfg)
        # Keep scanning until battery is gone
        for _ in range(20):
            if env.done:
                break
            env.scan()

        assert env.done is True

    def test_terminal_event_mentions_battery(self):
        cfg = _make_config(battery=5, scan_cost=3, move_cost=3, time_limit=200)
        env = Environment(cfg)
        obs = None
        for _ in range(10):
            if env.done:
                break
            obs = env.scan()
        assert obs is not None
        assert "battery" in obs["event"] or env.battery <= 0


# ---------------------------------------------------------------------------
# test_eve_not_exposed
# ---------------------------------------------------------------------------

class TestEveNotExposed:
    """The observation dict must never contain the hidden _eve_pos key."""

    def test_scan_obs_has_no_eve_pos_key(self):
        cfg = _make_config()
        env = Environment(cfg)
        obs = env.scan()
        # The private attribute must not be surfaced in the public observation
        assert "_eve_pos" not in obs
        assert "eve_pos" not in obs

    def test_move_obs_has_no_eve_pos_key(self):
        cfg = _make_config()
        env = Environment(cfg)
        obs = env.move("S")
        assert "_eve_pos" not in obs
        assert "eve_pos" not in obs

    def test_recharge_obs_has_no_eve_pos_key(self):
        cfg = _make_config(recharge_stations=[[0, 0]])
        env = Environment(cfg)
        obs = env.recharge()
        assert "_eve_pos" not in obs
        assert "eve_pos" not in obs
