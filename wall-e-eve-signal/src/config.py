"""
config.py
=========
Configuration dataclasses and YAML/JSON loader for the WALL-E EVE Signal project.

All mission parameters are validated here before any simulation begins.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Union

import yaml


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MapConfig:
    """Configuration for the simulation grid/map.

    Attributes:
        width:     Number of columns in the grid.
        height:    Number of rows in the grid.
        seed:      RNG seed for reproducibility.
        obstacles: List of [x, y] obstacle positions.
    """

    width: int
    height: int
    seed: int
    obstacles: list[list[int]] = field(default_factory=list)


@dataclass
class InitialSignalConfig:
    """Configuration for EVE's initial signal.

    Attributes:
        direction:  Cardinal direction string ('N','S','E','W','NE',…)
                    OR a coordinate pair [x, y] pointing to the signal origin.
        strength:   Signal strength in [0, 1].
        timestamp:  Mission timestamp at which the signal was recorded.
    """

    direction: Union[str, list[int]]
    strength: float
    timestamp: int


@dataclass
class ResourceConfig:
    """Configuration for agent resources.

    Attributes:
        battery:           Starting battery level.
        time_limit:        Maximum number of steps allowed.
        move_cost:         Battery consumed per move action.
        scan_cost:         Battery consumed per scan action.
        recharge_cost:     Battery consumed per recharge action (usually 0).
        recharge_amount:   Battery restored per recharge action.
        recharge_stations: List of [x, y] recharge station positions.
    """

    battery: int
    time_limit: int
    move_cost: int
    scan_cost: int
    recharge_cost: int
    recharge_amount: int
    recharge_stations: list[list[int]] = field(default_factory=list)


@dataclass
class ClueConfig:
    """A single clue definition.

    Attributes:
        position: [x, y] location of the clue on the grid.
        type:     Clue category string (e.g. 'audio', 'visual', 'thermal').
        strength: Clue intensity in [0, 1].
    """

    position: list[int]
    type: str
    strength: float


@dataclass
class DecoyConfig:
    """A single decoy definition.

    Attributes:
        position: [x, y] location of the decoy.
        strength: False signal strength in [0, 1].
    """

    position: list[int]
    strength: float


# ---------------------------------------------------------------------------
# Top-level Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Top-level configuration for a WALL-E EVE Signal mission.

    Attributes:
        map:            Grid / map parameters.
        initial_signal: Parameters of the initial signal from EVE.
        resources:      Agent resource parameters.
        eve:            Known [x, y] position of EVE, or None to derive from seed.
        clues:          Either a list of ClueConfig objects or an integer seed
                        used to generate clues procedurally.
        sensor_noise:   Standard deviation of Gaussian noise added to sensor
                        readings.  0.0 means perfect sensors.
        decoys:         List of decoy signal sources that can mislead the agent.
    """

    map: MapConfig
    initial_signal: InitialSignalConfig
    resources: ResourceConfig
    eve: list[int] | None = None
    clues: list[ClueConfig] | int = field(default_factory=list)
    sensor_noise: float = 0.0
    decoys: list[DecoyConfig] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require(condition: bool, message: str) -> None:
    """Raise ValueError with *message* unless *condition* is True."""
    if not condition:
        raise ValueError(f"Config validation error: {message}")


def _validate_xy(value: Any, label: str) -> list[int]:
    """Validate that *value* is a two-element list of non-negative integers."""
    _require(
        isinstance(value, (list, tuple)) and len(value) == 2,
        f"{label} must be a list of two integers, got {value!r}",
    )
    x, y = int(value[0]), int(value[1])
    _require(x >= 0 and y >= 0, f"{label} coordinates must be non-negative, got [{x},{y}]")
    return [x, y]


def _validate_map(raw: dict) -> MapConfig:
    """Parse and validate the 'map' section."""
    _require(isinstance(raw, dict), "'map' must be a mapping")
    _require("width" in raw, "'map.width' is required")
    _require("height" in raw, "'map.height' is required")
    _require("seed" in raw, "'map.seed' is required")

    width = int(raw["width"])
    height = int(raw["height"])
    seed = int(raw["seed"])
    _require(width > 0, f"'map.width' must be positive, got {width}")
    _require(height > 0, f"'map.height' must be positive, got {height}")

    obstacles: list[list[int]] = []
    for i, obs in enumerate(raw.get("obstacles", [])):
        obstacles.append(_validate_xy(obs, f"map.obstacles[{i}]"))

    return MapConfig(width=width, height=height, seed=seed, obstacles=obstacles)


def _validate_initial_signal(raw: dict) -> InitialSignalConfig:
    """Parse and validate the 'initial_signal' section."""
    _require(isinstance(raw, dict), "'initial_signal' must be a mapping")
    _require("direction" in raw, "'initial_signal.direction' is required")
    _require("strength" in raw, "'initial_signal.strength' is required")
    _require("timestamp" in raw, "'initial_signal.timestamp' is required")

    direction_raw = raw["direction"]
    valid_directions = {"N", "S", "E", "W", "NE", "NW", "SE", "SW"}
    if isinstance(direction_raw, str):
        _require(
            direction_raw.upper() in valid_directions,
            f"'initial_signal.direction' string must be one of {valid_directions}, "
            f"got {direction_raw!r}",
        )
        direction: Union[str, list[int]] = direction_raw.upper()
    else:
        direction = _validate_xy(direction_raw, "initial_signal.direction")

    strength = float(raw["strength"])
    _require(0.0 <= strength <= 1.0, f"'initial_signal.strength' must be in [0,1], got {strength}")

    timestamp = int(raw["timestamp"])
    _require(timestamp >= 0, f"'initial_signal.timestamp' must be non-negative, got {timestamp}")

    return InitialSignalConfig(direction=direction, strength=strength, timestamp=timestamp)


def _validate_resources(raw: dict) -> ResourceConfig:
    """Parse and validate the 'resources' section."""
    _require(isinstance(raw, dict), "'resources' must be a mapping")

    required_ints = ("battery", "time_limit", "move_cost", "scan_cost",
                     "recharge_cost", "recharge_amount")
    for key in required_ints:
        _require(key in raw, f"'resources.{key}' is required")

    battery = int(raw["battery"])
    time_limit = int(raw["time_limit"])
    move_cost = int(raw["move_cost"])
    scan_cost = int(raw["scan_cost"])
    recharge_cost = int(raw["recharge_cost"])
    recharge_amount = int(raw["recharge_amount"])

    _require(battery > 0, f"'resources.battery' must be positive, got {battery}")
    _require(time_limit > 0, f"'resources.time_limit' must be positive, got {time_limit}")
    _require(move_cost >= 0, f"'resources.move_cost' must be non-negative, got {move_cost}")
    _require(scan_cost >= 0, f"'resources.scan_cost' must be non-negative, got {scan_cost}")
    _require(recharge_cost >= 0, f"'resources.recharge_cost' must be non-negative, got {recharge_cost}")
    _require(recharge_amount > 0, f"'resources.recharge_amount' must be positive, got {recharge_amount}")

    stations: list[list[int]] = []
    for i, st in enumerate(raw.get("recharge_stations", [])):
        stations.append(_validate_xy(st, f"resources.recharge_stations[{i}]"))

    return ResourceConfig(
        battery=battery,
        time_limit=time_limit,
        move_cost=move_cost,
        scan_cost=scan_cost,
        recharge_cost=recharge_cost,
        recharge_amount=recharge_amount,
        recharge_stations=stations,
    )


def _validate_clues(raw: Any) -> list[ClueConfig] | int:
    """Parse and validate the 'clues' field (list of clues or integer seed)."""
    if isinstance(raw, int):
        _require(raw >= 0, f"'clues' seed must be non-negative, got {raw}")
        return raw
    if not isinstance(raw, list):
        raise ValueError(
            f"Config validation error: 'clues' must be a list of clue objects or an integer seed, "
            f"got {type(raw).__name__}"
        )
    result: list[ClueConfig] = []
    for i, item in enumerate(raw):
        _require(isinstance(item, dict), f"'clues[{i}]' must be a mapping")
        _require("position" in item, f"'clues[{i}].position' is required")
        _require("type" in item, f"'clues[{i}].type' is required")
        _require("strength" in item, f"'clues[{i}].strength' is required")
        pos = _validate_xy(item["position"], f"clues[{i}].position")
        clue_type = str(item["type"])
        strength = float(item["strength"])
        _require(0.0 <= strength <= 1.0, f"'clues[{i}].strength' must be in [0,1], got {strength}")
        result.append(ClueConfig(position=pos, type=clue_type, strength=strength))
    return result


def _validate_decoys(raw: list) -> list[DecoyConfig]:
    """Parse and validate the 'decoys' list."""
    result: list[DecoyConfig] = []
    for i, item in enumerate(raw):
        _require(isinstance(item, dict), f"'decoys[{i}]' must be a mapping")
        _require("position" in item, f"'decoys[{i}].position' is required")
        _require("strength" in item, f"'decoys[{i}].strength' is required")
        pos = _validate_xy(item["position"], f"decoys[{i}].position")
        strength = float(item["strength"])
        _require(0.0 <= strength <= 1.0, f"'decoys[{i}].strength' must be in [0,1], got {strength}")
        result.append(DecoyConfig(position=pos, strength=strength))
    return result


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> Config:
    """Load and validate a mission configuration from a YAML or JSON file.

    Parameters
    ----------
    path:
        Filesystem path to the configuration file.  Files ending in ``.json``
        are parsed as JSON; everything else is treated as YAML.

    Returns
    -------
    Config
        A fully-validated :class:`Config` instance ready for use by the
        simulation.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If any required field is missing or fails validation.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        if path.lower().endswith(".json"):
            raw: dict = json.load(fh)
        else:
            raw = yaml.safe_load(fh)

    _require(isinstance(raw, dict), "Configuration root must be a mapping")

    # Required sections
    _require("map" in raw, "Top-level key 'map' is required")
    _require("initial_signal" in raw, "Top-level key 'initial_signal' is required")
    _require("resources" in raw, "Top-level key 'resources' is required")

    map_cfg = _validate_map(raw["map"])
    signal_cfg = _validate_initial_signal(raw["initial_signal"])
    resource_cfg = _validate_resources(raw["resources"])

    # Optional: eve position
    eve: list[int] | None = None
    if "eve" in raw and raw["eve"] is not None:
        eve = _validate_xy(raw["eve"], "eve")
        _require(
            0 <= eve[0] < map_cfg.width and 0 <= eve[1] < map_cfg.height,
            f"'eve' position {eve} is outside grid bounds "
            f"({map_cfg.width}x{map_cfg.height})",
        )

    # Optional: clues
    clues: list[ClueConfig] | int = []
    if "clues" in raw and raw["clues"] is not None:
        clues = _validate_clues(raw["clues"])

    # Optional: sensor_noise (default 0.0)
    sensor_noise = float(raw.get("sensor_noise", 0.0))
    _require(sensor_noise >= 0.0, f"'sensor_noise' must be non-negative, got {sensor_noise}")

    # Optional: decoys (default empty)
    decoys: list[DecoyConfig] = []
    if "decoys" in raw and raw["decoys"]:
        decoys = _validate_decoys(raw["decoys"])

    return Config(
        map=map_cfg,
        initial_signal=signal_cfg,
        resources=resource_cfg,
        eve=eve,
        clues=clues,
        sensor_noise=sensor_noise,
        decoys=decoys,
    )
