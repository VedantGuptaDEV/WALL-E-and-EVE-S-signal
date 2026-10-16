"""
environment.py
==============
Simulated grid environment for the WALL-E EVE Signal mission.

The environment manages the hidden state of the world (EVE's position, clue
positions) while exposing only safe observations to the agent.
"""

from __future__ import annotations

import math
import random
from typing import Optional, TypedDict

from .config import ClueConfig, Config


# ---------------------------------------------------------------------------
# Observation TypedDict
# ---------------------------------------------------------------------------

class ScanResult(TypedDict):
    """Result returned by a scan action."""

    clue_strength: float
    """Strength of the nearest detected clue (0.0 if none detected)."""

    clue_direction: Optional[str]
    """Cardinal direction toward the nearest clue, or None."""

    eve_detected: bool
    """True if EVE is within scan_radius of the agent."""


class Observation(TypedDict):
    """Full observation returned after every environment action."""

    agent_pos: list[int]
    """Current [x, y] position of the agent."""

    battery: int
    """Remaining battery level."""

    time_remaining: int
    """Steps remaining before the time limit is reached."""

    steps_taken: int
    """Total steps the agent has taken so far."""

    scan_result: ScanResult
    """Populated when action was 'scan'; zeros/None otherwise."""

    event: str
    """Human-readable description of what happened this step."""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

SCAN_RADIUS: int = 2
"""Default scan radius in grid cells (Chebyshev distance)."""

_DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}

_COMPASS: list[str] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _angle_to_compass(dx: float, dy: float) -> str:
    """Convert a (dx, dy) vector to the nearest compass direction string."""
    angle = math.degrees(math.atan2(-dy, dx))  # screen coords: y increases downward
    angle = (angle + 360) % 360
    idx = int((angle + 22.5) / 45) % 8
    return _COMPASS[idx]


class Environment:
    """Grid-based simulation environment for the WALL-E EVE signal mission.

    The environment holds the *ground truth* that is hidden from the agent:
    EVE's position and exact clue positions.  The agent receives noisy
    observations through :meth:`scan`, movement feedback through
    :meth:`move`, and battery top-ups through :meth:`recharge`.

    Parameters
    ----------
    config:
        Validated :class:`~config.Config` instance describing the mission.
    """

    def __init__(self, config: Config) -> None:
        """Initialise the environment from *config*."""
        self._config = config
        self._rng = random.Random(config.map.seed)

        # Grid dimensions
        self._width: int = config.map.width
        self._height: int = config.map.height

        # Obstacles — stored as a set of (x, y) tuples for O(1) lookup
        self._obstacles: set[tuple[int, int]] = {
            (int(o[0]), int(o[1])) for o in config.map.obstacles
        }

        # Recharge stations
        self._recharge_stations: set[tuple[int, int]] = {
            (int(s[0]), int(s[1])) for s in config.resources.recharge_stations
        }

        # Agent starts at (0, 0) unless it is an obstacle
        start = self._find_free_cell(preferred=(0, 0))
        self._agent_pos: list[int] = list(start)

        # EVE position (hidden)
        if config.eve is not None:
            self._eve_pos: tuple[int, int] = (int(config.eve[0]), int(config.eve[1]))
        else:
            self._eve_pos = self._generate_hidden_position(exclude={tuple(self._agent_pos)})

        # Clues (hidden)
        self._clue_positions: list[tuple[tuple[int, int], ClueConfig]] = []
        self._init_clues(config)

        # Decoys (treated like weak clues from the environment perspective)
        self._decoy_positions: list[tuple[tuple[int, int], float]] = [
            ((int(d.position[0]), int(d.position[1])), d.strength)
            for d in config.decoys
        ]

        # Resources
        self._max_battery: int = config.resources.battery
        self._battery: int = config.resources.battery
        self._time_remaining: int = config.resources.time_limit
        self._steps_taken: int = 0

        # Terminal flags
        self._done: bool = False
        self._found_eve: bool = False
        self._mission_ended: bool = False

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _find_free_cell(self, preferred: tuple[int, int]) -> tuple[int, int]:
        """Return *preferred* if free, otherwise the first non-obstacle cell."""
        if preferred not in self._obstacles:
            px, py = preferred
            if 0 <= px < self._width and 0 <= py < self._height:
                return preferred
        for y in range(self._height):
            for x in range(self._width):
                if (x, y) not in self._obstacles:
                    return (x, y)
        raise RuntimeError("No free cell found in grid — check obstacle configuration")

    def _generate_hidden_position(
        self, exclude: set[tuple[int, int]] | None = None
    ) -> tuple[int, int]:
        """Pick a random non-obstacle cell, excluding *exclude*."""
        exclude = exclude or set()
        candidates = [
            (x, y)
            for y in range(self._height)
            for x in range(self._width)
            if (x, y) not in self._obstacles and (x, y) not in exclude
        ]
        if not candidates:
            raise RuntimeError("No valid cell for hidden position generation")
        return self._rng.choice(candidates)

    def _init_clues(self, config: Config) -> None:
        """Place clues on the grid from config or by seeded generation."""
        if isinstance(config.clues, int):
            # Generate clues procedurally using the clue seed
            clue_rng = random.Random(config.clues)
            num_clues = clue_rng.randint(3, max(3, (self._width * self._height) // 10))
            types = ["audio", "visual", "thermal", "magnetic"]
            used: set[tuple[int, int]] = set()
            for _ in range(num_clues):
                pos = self._generate_hidden_position(
                    exclude=self._obstacles | used | {self._eve_pos}
                )
                used.add(pos)
                clue = ClueConfig(
                    position=list(pos),
                    type=clue_rng.choice(types),
                    strength=round(clue_rng.uniform(0.2, 1.0), 2),
                )
                self._clue_positions.append((pos, clue))
        else:
            for clue in config.clues:
                pos = (int(clue.position[0]), int(clue.position[1]))
                self._clue_positions.append((pos, clue))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Grid width (number of columns)."""
        return self._width

    @property
    def height(self) -> int:
        """Grid height (number of rows)."""
        return self._height

    @property
    def agent_pos(self) -> list[int]:
        """Current agent position as [x, y]."""
        return list(self._agent_pos)

    @property
    def battery(self) -> int:
        """Remaining battery level."""
        return self._battery

    @property
    def time_remaining(self) -> int:
        """Steps remaining before mission timeout."""
        return self._time_remaining

    @property
    def steps_taken(self) -> int:
        """Total steps taken so far."""
        return self._steps_taken

    @property
    def done(self) -> bool:
        """True when the mission has terminated for any reason."""
        return self._done

    @property
    def found_eve(self) -> bool:
        """True when the agent has located EVE."""
        return self._found_eve

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _make_empty_scan(self) -> ScanResult:
        """Return an empty / null ScanResult."""
        return ScanResult(clue_strength=0.0, clue_direction=None, eve_detected=False)

    def _make_observation(self, event: str, scan: ScanResult | None = None) -> Observation:
        """Construct an :class:`Observation` for the current state."""
        return Observation(
            agent_pos=list(self._agent_pos),
            battery=self._battery,
            time_remaining=self._time_remaining,
            steps_taken=self._steps_taken,
            scan_result=scan if scan is not None else self._make_empty_scan(),
            event=event,
        )

    def _add_noise(self, value: float) -> float:
        """Add Gaussian noise to *value* and clamp to [0, 1]."""
        noise = self._rng.gauss(0.0, self._config.sensor_noise)
        return max(0.0, min(1.0, value + noise))

    def _noisy_direction(self, true_direction: str) -> str:
        """Possibly rotate the compass direction by ±1 step when noise is high."""
        if self._config.sensor_noise > 0.3:
            # With high noise, sometimes return an adjacent compass bearing
            if self._rng.random() < self._config.sensor_noise * 0.5:
                idx = _COMPASS.index(true_direction) if true_direction in _COMPASS else 0
                offset = self._rng.choice([-1, 1])
                return _COMPASS[(idx + offset) % len(_COMPASS)]
        return true_direction

    def _check_terminal(self) -> None:
        """Set _done if any terminal condition is met."""
        if self._battery <= 0:
            self._done = True
        if self._time_remaining <= 0:
            self._done = True
        if self._found_eve:
            self._done = True
        if self._mission_ended:
            self._done = True

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def move(self, direction: str) -> Observation:
        """Move the agent one cell in *direction*.

        Parameters
        ----------
        direction:
            One of ``'N'``, ``'S'``, ``'E'``, ``'W'``.

        Returns
        -------
        Observation
            Current state after the move attempt.

        Notes
        -----
        - Costs :attr:`~config.ResourceConfig.move_cost` battery and 1 time step.
        - Blocked by walls and obstacles; reports ``'blocked'`` event in that case.
        - If the agent steps onto EVE's cell, ``found_eve`` is set to ``True``.
        """
        if self._done:
            return self._make_observation("mission_already_done")

        direction = direction.upper()
        if direction not in _DIRECTION_DELTAS:
            return self._make_observation(f"invalid_direction:{direction}")

        dx, dy = _DIRECTION_DELTAS[direction]
        nx, ny = self._agent_pos[0] + dx, self._agent_pos[1] + dy

        # Check wall/obstacle
        if not (0 <= nx < self._width and 0 <= ny < self._height):
            event = "blocked:wall"
        elif (nx, ny) in self._obstacles:
            event = "blocked:obstacle"
        else:
            self._agent_pos = [nx, ny]
            # Battery & time costs always apply whether blocked or not
            event = f"moved:{direction}"

            # Check if agent landed on EVE
            if (nx, ny) == self._eve_pos:
                self._found_eve = True
                event = f"found_eve:{direction}"

        # Deduct costs regardless (attempting to move costs resources)
        self._battery -= self._config.resources.move_cost
        self._time_remaining -= 1
        self._steps_taken += 1
        self._battery = max(0, self._battery)

        self._check_terminal()
        obs = self._make_observation(event)

        # Append terminal event text
        if self._done and "terminal" not in event:
            obs["event"] = event + "|" + self._terminal_reason()

        return obs

    def scan(self) -> Observation:
        """Scan the surroundings for clues and EVE.

        Returns
        -------
        Observation
            Observation with :attr:`~Observation.scan_result` populated.

        Notes
        -----
        - Costs :attr:`~config.ResourceConfig.scan_cost` battery and 1 step.
        - Clue strength and direction are noisy when ``sensor_noise > 0``.
        - ``scan_result.eve_detected`` is True if EVE is within *SCAN_RADIUS*
          cells (Chebyshev distance).  If the agent is **on** EVE's cell,
          :attr:`found_eve` is set to ``True``.
        """
        if self._done:
            return self._make_observation("mission_already_done")

        ax, ay = self._agent_pos

        # -- Check if agent is standing on EVE --
        if (ax, ay) == self._eve_pos:
            self._found_eve = True

        # -- Find nearest clue within scan radius --
        best_strength = 0.0
        best_direction: Optional[str] = None

        for (cx, cy), clue in self._clue_positions:
            dist = max(abs(cx - ax), abs(cy - ay))  # Chebyshev distance
            if dist <= SCAN_RADIUS:
                raw_strength = clue.strength / max(1, dist)  # attenuate by distance
                if self._config.sensor_noise > 0:
                    raw_strength = self._add_noise(raw_strength)
                if raw_strength > best_strength:
                    best_strength = raw_strength
                    dx_c, dy_c = cx - ax, cy - ay
                    true_dir = _angle_to_compass(float(dx_c), float(dy_c))
                    best_direction = self._noisy_direction(true_dir)

        # -- Check decoys --
        for (dx_d, dy_d), decoy_strength in self._decoy_positions:
            dist = max(abs(dx_d - ax), abs(dy_d - ay))
            if dist <= SCAN_RADIUS:
                raw_strength = decoy_strength / max(1, dist)
                if self._config.sensor_noise > 0:
                    raw_strength = self._add_noise(raw_strength)
                if raw_strength > best_strength:
                    best_strength = raw_strength
                    dx_c, dy_c = dx_d - ax, dy_d - ay
                    true_dir = _angle_to_compass(float(dx_c), float(dy_c))
                    best_direction = self._noisy_direction(true_dir)

        # -- Eve detection --
        ex, ey = self._eve_pos
        eve_dist = max(abs(ex - ax), abs(ey - ay))
        eve_detected = eve_dist <= SCAN_RADIUS

        scan: ScanResult = ScanResult(
            clue_strength=round(best_strength, 4),
            clue_direction=best_direction,
            eve_detected=eve_detected,
        )

        # Costs
        self._battery -= self._config.resources.scan_cost
        self._time_remaining -= 1
        self._steps_taken += 1
        self._battery = max(0, self._battery)

        self._check_terminal()
        event = "scan"
        if self._found_eve:
            event = "scan|found_eve"
        obs = self._make_observation(event, scan=scan)

        if self._done and "terminal" not in event:
            obs["event"] = obs["event"] + "|" + self._terminal_reason()

        return obs

    def recharge(self) -> Observation:
        """Attempt to recharge the agent's battery.

        Returns
        -------
        Observation
            Current state after recharge attempt.

        Notes
        -----
        - Only effective when the agent is on a recharge station cell.
        - Adds :attr:`~config.ResourceConfig.recharge_amount` battery
          (capped at starting battery max).
        - Always costs 1 time step.
        - Costs :attr:`~config.ResourceConfig.recharge_cost` battery
          (typically 0).
        """
        if self._done:
            return self._make_observation("mission_already_done")

        pos_tuple = (self._agent_pos[0], self._agent_pos[1])
        if pos_tuple in self._recharge_stations:
            old_battery = self._battery
            self._battery = min(
                self._max_battery,
                self._battery + self._config.resources.recharge_amount,
            )
            self._battery -= self._config.resources.recharge_cost
            self._battery = max(0, min(self._max_battery, self._battery))
            gained = self._battery - old_battery + self._config.resources.recharge_cost
            event = f"recharged:+{max(0, gained)}"
        else:
            event = "recharge_failed:not_on_station"

        self._time_remaining -= 1
        self._steps_taken += 1

        self._check_terminal()
        obs = self._make_observation(event)

        if self._done and "terminal" not in event:
            obs["event"] = obs["event"] + "|" + self._terminal_reason()

        return obs

    def end_mission(self) -> Observation:
        """Terminate the mission immediately.

        Returns
        -------
        Observation
            Final observation with event ``'mission_ended'``.
        """
        self._mission_ended = True
        self._done = True
        return self._make_observation("mission_ended")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _terminal_reason(self) -> str:
        """Return a string describing why the mission ended."""
        if self._found_eve:
            return "terminal:found_eve"
        if self._battery <= 0:
            return "terminal:battery_depleted"
        if self._time_remaining <= 0:
            return "terminal:time_limit_reached"
        if self._mission_ended:
            return "terminal:mission_ended"
        return "terminal:unknown"

    def is_obstacle(self, x: int, y: int) -> bool:
        """Return True if cell (x, y) is an obstacle or out of bounds."""
        if not (0 <= x < self._width and 0 <= y < self._height):
            return True
        return (x, y) in self._obstacles

    def is_recharge_station(self, x: int, y: int) -> bool:
        """Return True if cell (x, y) is a recharge station."""
        return (x, y) in self._recharge_stations

    def get_clue_at(self, x: int, y: int) -> ClueConfig | None:
        """Return the clue at (x, y) if the agent is standing on it.

        This is used internally and does NOT expose hidden EVE position.
        """
        for (cx, cy), clue in self._clue_positions:
            if cx == x and cy == y:
                return clue
        return None
