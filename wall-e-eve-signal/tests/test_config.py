"""
tests/test_config.py
====================
Pytest tests for src/config.py — configuration loading and validation.
"""

from __future__ import annotations

import os
import textwrap
import tempfile

import pytest

# Add project root to path so imports resolve without installation
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, Config, MapConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(content: str) -> str:
    """Write *content* to a temp YAML file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(textwrap.dedent(content))
    tmp.close()
    return tmp.name


# Paths to real config files (relative to project root)
_ROOT = Path(__file__).parent.parent
_EASY_CONFIG = str(_ROOT / "configs" / "sample_easy.yaml")
_HARD_CONFIG = str(_ROOT / "configs" / "sample_hard.yaml")


# ---------------------------------------------------------------------------
# test_load_easy_config
# ---------------------------------------------------------------------------

class TestLoadEasyConfig:
    """Load sample_easy.yaml and check all top-level fields."""

    @pytest.fixture(scope="class")
    def config(self):
        return load_config(_EASY_CONFIG)

    def test_returns_config_instance(self, config):
        assert isinstance(config, Config)

    def test_map_dimensions(self, config):
        assert config.map.width == 10
        assert config.map.height == 10

    def test_map_seed(self, config):
        assert config.map.seed == 42

    def test_map_obstacles(self, config):
        obs = [list(o) for o in config.map.obstacles]
        assert [2, 3] in obs
        assert [2, 4] in obs
        assert [5, 5] in obs
        assert [6, 5] in obs

    def test_eve_position(self, config):
        assert config.eve == [7, 8]

    def test_initial_signal_direction(self, config):
        assert config.initial_signal.direction == "NE"

    def test_initial_signal_strength(self, config):
        assert config.initial_signal.strength == pytest.approx(0.8)

    def test_initial_signal_timestamp(self, config):
        assert config.initial_signal.timestamp == 0

    def test_battery(self, config):
        assert config.resources.battery == 150

    def test_time_limit(self, config):
        assert config.resources.time_limit == 80

    def test_move_cost(self, config):
        assert config.resources.move_cost == 2

    def test_scan_cost(self, config):
        assert config.resources.scan_cost == 5

    def test_recharge_cost(self, config):
        assert config.resources.recharge_cost == 1

    def test_recharge_amount(self, config):
        assert config.resources.recharge_amount == 50

    def test_recharge_stations(self, config):
        stations = [list(s) for s in config.resources.recharge_stations]
        assert [1, 1] in stations
        assert [8, 2] in stations
        assert [5, 9] in stations

    def test_clues_is_list(self, config):
        assert isinstance(config.clues, list)
        assert len(config.clues) == 4

    def test_clue_types(self, config):
        types = {c.type for c in config.clues}
        assert "energy_trace" in types
        assert "sound" in types
        assert "heat_signature" in types
        assert "movement" in types

    def test_sensor_noise(self, config):
        assert config.sensor_noise == pytest.approx(0.05)

    def test_decoys_empty(self, config):
        assert config.decoys == []


# ---------------------------------------------------------------------------
# test_load_hard_config
# ---------------------------------------------------------------------------

class TestLoadHardConfig:
    """Load sample_hard.yaml and check key fields."""

    @pytest.fixture(scope="class")
    def config(self):
        return load_config(_HARD_CONFIG)

    def test_returns_config_instance(self, config):
        assert isinstance(config, Config)

    def test_map_dimensions(self, config):
        assert config.map.width == 15
        assert config.map.height == 15

    def test_map_seed(self, config):
        assert config.map.seed == 99

    def test_map_has_many_obstacles(self, config):
        # Hard config has 8+ obstacles
        assert len(config.map.obstacles) >= 8

    def test_eve_position(self, config):
        assert config.eve == [12, 11]

    def test_initial_signal_direction(self, config):
        assert config.initial_signal.direction == "E"

    def test_initial_signal_strength(self, config):
        assert config.initial_signal.strength == pytest.approx(0.4)

    def test_battery(self, config):
        assert config.resources.battery == 220

    def test_time_limit(self, config):
        assert config.resources.time_limit == 100

    def test_clues_is_seed(self, config):
        # Hard config uses integer clue seed
        assert isinstance(config.clues, int)
        assert config.clues == 99

    def test_sensor_noise(self, config):
        assert config.sensor_noise == pytest.approx(0.15)

    def test_decoys(self, config):
        assert len(config.decoys) == 2
        positions = [list(d.position) for d in config.decoys]
        assert [5, 5] in positions
        assert [10, 3] in positions


# ---------------------------------------------------------------------------
# test_missing_required_field
# ---------------------------------------------------------------------------

class TestMissingRequiredField:
    """Config should raise ValueError when 'map' key is absent."""

    def test_raises_value_error_when_map_missing(self):
        path = _write_yaml("""
            initial_signal:
              direction: 'N'
              strength: 0.5
              timestamp: 0
            resources:
              battery: 100
              time_limit: 50
              move_cost: 2
              scan_cost: 5
              recharge_cost: 0
              recharge_amount: 30
        """)
        try:
            with pytest.raises(ValueError, match="map"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_raises_value_error_when_initial_signal_missing(self):
        path = _write_yaml("""
            map:
              width: 5
              height: 5
              seed: 1
            resources:
              battery: 100
              time_limit: 50
              move_cost: 2
              scan_cost: 5
              recharge_cost: 0
              recharge_amount: 30
        """)
        try:
            with pytest.raises(ValueError, match="initial_signal"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_raises_value_error_when_resources_missing(self):
        path = _write_yaml("""
            map:
              width: 5
              height: 5
              seed: 1
            initial_signal:
              direction: 'N'
              strength: 0.5
              timestamp: 0
        """)
        try:
            with pytest.raises(ValueError):
                load_config(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# test_invalid_battery
# ---------------------------------------------------------------------------

class TestInvalidBattery:
    """Config should raise ValueError when battery <= 0."""

    def _make_yaml(self, battery_value: int) -> str:
        return _write_yaml(f"""
            map:
              width: 5
              height: 5
              seed: 1
            initial_signal:
              direction: 'N'
              strength: 0.5
              timestamp: 0
            resources:
              battery: {battery_value}
              time_limit: 50
              move_cost: 2
              scan_cost: 5
              recharge_cost: 0
              recharge_amount: 30
        """)

    def test_zero_battery_raises(self):
        path = self._make_yaml(0)
        try:
            with pytest.raises(ValueError, match="battery"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_negative_battery_raises(self):
        path = self._make_yaml(-10)
        try:
            with pytest.raises(ValueError, match="battery"):
                load_config(path)
        finally:
            os.unlink(path)

    def test_positive_battery_accepted(self):
        path = self._make_yaml(100)
        try:
            cfg = load_config(path)
            assert cfg.resources.battery == 100
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# test_default_sensor_noise
# ---------------------------------------------------------------------------

class TestDefaultSensorNoise:
    """When 'sensor_noise' is absent from YAML, default should be 0.0."""

    def test_default_is_zero(self):
        path = _write_yaml("""
            map:
              width: 5
              height: 5
              seed: 1
            initial_signal:
              direction: 'N'
              strength: 0.5
              timestamp: 0
            resources:
              battery: 100
              time_limit: 50
              move_cost: 2
              scan_cost: 5
              recharge_cost: 0
              recharge_amount: 30
        """)
        try:
            cfg = load_config(path)
            assert cfg.sensor_noise == pytest.approx(0.0)
        finally:
            os.unlink(path)
