"""
visualize.py
============
ASCII visualization for the WALL-E EVE Signal mission.

The :class:`Visualizer` renders the grid state (obstacles, agent path,
clues, recharge stations, EVE if found) with a belief-heatmap overlay and
exports the result to a text file.

Cell priority (highest → lowest):
  Combined agent+EVE (&) > EVE (E) > Agent (@) > Discovered clue (C)
  > Discovered decoy (D) > Recharge (R) > Obstacle (#) > Visited (*) >
  Belief heatmap overlay > Empty (.)

Symbols
-------
@  Agent's final position
E  EVE (only shown when found)
&  Agent is standing on EVE
C  Scan position where a clue signal was detected (discovered clue)
D  Position that showed a clue signal but was a decoy (suppressed later)
R  Recharge station
#  Obstacle / wall
*  Visited cell (in agent path)
.  Empty / unvisited
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .belief import BeliefGrid
from .config import Config
from .environment import Environment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters used to represent the belief-probability quantile (low → high)
_HEATMAP_CHARS: str = " .:-=+*#%@"

# Map cell characters
_EMPTY      = "."   # unvisited, passable
_AGENT      = "@"   # agent's current position
_AGENT_EVE  = "&"   # agent is standing on EVE
_VISITED    = "*"   # cell in the agent path (previously visited)
_CLUE       = "C"   # discovered clue location (scan returned positive signal)
_DECOY      = "D"   # discovered decoy (signal was later ruled out)
_RECHARGE   = "R"   # recharge station
_OBSTACLE   = "#"   # obstacle / wall
_EVE        = "E"   # EVE (only shown when found)

# Legend text shown below the map
_LEGEND_LINES = [
    "Legend:",
    f"  {_AGENT}=Agent (final pos)   {_AGENT_EVE}=Agent on EVE   {_EVE}=EVE(found)",
    f"  {_VISITED}=Visited   {_CLUE}=Clue discovered   {_DECOY}=Decoy ruled out",
    f"  {_RECHARGE}=Recharge   {_OBSTACLE}=Obstacle   {_EMPTY}=Empty",
]

_BELIEF_LEGEND = (
    "Belief intensity (low->high): " + "  ".join(_HEATMAP_CHARS.strip())
)


class Visualizer:
    """Renders the mission state as an ASCII map.

    Parameters
    ----------
    environment:
        The live (or finished) :class:`~environment.Environment` instance.
    belief:
        The current :class:`~belief.BeliefGrid` probability map.
    agent_path:
        Ordered list of ``[x, y]`` positions visited by the agent.  The
        *last* element is treated as the current position.
    config:
        Validated mission :class:`~config.Config`.
    discovered_clue_positions:
        Dict of ``{(x, y): max_strength}`` — scan positions where a positive
        clue signal was returned.  Supplied by the agent after run().
    discovered_decoy_positions:
        Set of ``(x, y)`` positions that had a clue signal but turned out to
        be decoys (agent visited them without finding EVE after strong signal).
    """

    def __init__(
        self,
        environment: Environment,
        belief: BeliefGrid,
        agent_path: list[list[int]],
        config: Config,
        discovered_clue_positions: Optional[dict[tuple[int, int], float]] = None,
        discovered_decoy_positions: Optional[set[tuple[int, int]]] = None,
    ) -> None:
        """Store references for later rendering."""
        self._env = environment
        self._belief = belief
        self._path = agent_path
        self._config = config
        self._clue_positions: dict[tuple[int, int], float] = (
            discovered_clue_positions or {}
        )
        self._decoy_positions: set[tuple[int, int]] = (
            discovered_decoy_positions or set()
        )

    # ------------------------------------------------------------------
    # Public rendering API
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Render the full grid as an annotated ASCII string.

        Returns
        -------
        str
            Multi-line string containing the map, a coordinate ruler, and a
            legend.
        """
        width  = self._env.width
        height = self._env.height

        # Build sets for fast lookup
        obstacle_set  = {(int(o[0]), int(o[1])) for o in self._config.map.obstacles}
        recharge_set  = {
            (int(s[0]), int(s[1]))
            for s in self._config.resources.recharge_stations
        }

        # Visited cells (all except the last one)
        visited_set: set[tuple[int, int]] = set()
        for pos in self._path[:-1]:
            visited_set.add((int(pos[0]), int(pos[1])))

        # Current agent position (last path entry)
        if self._path:
            agent_pos: Optional[tuple[int, int]] = (
                int(self._path[-1][0]), int(self._path[-1][1])
            )
        else:
            ap = self._env.agent_pos
            agent_pos = (int(ap[0]), int(ap[1]))

        # EVE position (only revealed when found)
        eve_pos: Optional[tuple[int, int]] = None
        if self._env.found_eve and self._config.eve is not None:
            eve_pos = (int(self._config.eve[0]), int(self._config.eve[1]))

        # Belief probability array for heatmap
        belief_array = self._belief.as_array()  # shape (height, width)
        prob_min = float(belief_array.min())
        prob_max = float(belief_array.max())
        prob_range = max(prob_max - prob_min, 1e-12)

        lines: list[str] = []

        # Column header
        col_header = "   " + "".join(str(x % 10) for x in range(width))
        lines.append(col_header)
        lines.append("   " + "-" * width)

        for y in range(height):
            row_chars: list[str] = []
            for x in range(width):
                cell = (x, y)

                if cell == eve_pos and cell == agent_pos:
                    row_chars.append(_AGENT_EVE)
                elif cell == eve_pos:
                    row_chars.append(_EVE)
                elif cell == agent_pos:
                    row_chars.append(_AGENT)
                elif cell in self._decoy_positions:
                    row_chars.append(_DECOY)
                elif cell in self._clue_positions:
                    row_chars.append(_CLUE)
                elif cell in recharge_set:
                    row_chars.append(_RECHARGE)
                elif cell in obstacle_set:
                    row_chars.append(_OBSTACLE)
                elif cell in visited_set:
                    row_chars.append(_VISITED)
                else:
                    # Belief heatmap overlay on unvisited cells
                    prob = float(belief_array[y, x])
                    norm = (prob - prob_min) / prob_range
                    idx = min(int(norm * len(_HEATMAP_CHARS)), len(_HEATMAP_CHARS) - 1)
                    row_chars.append(_HEATMAP_CHARS[idx])

            lines.append(f"{y:2d}|{''.join(row_chars)}")

        lines.append("")
        lines.extend(_LEGEND_LINES)
        lines.append(_BELIEF_LEGEND)

        return "\n".join(lines)

    def render_belief_heatmap(self) -> str:
        """Render a pure belief-probability ASCII heatmap titled 'BELIEF HEATMAP (final)'.

        Each cell shows one character from ``' .:-=+*#%@'`` scaled from the
        minimum to the maximum probability in the grid.  No game-layer
        symbols are shown.

        Returns
        -------
        str
            Multi-line string of the belief-only map.
        """
        belief_array = self._belief.as_array()
        height, width = belief_array.shape

        prob_min = float(belief_array.min())
        prob_max = float(belief_array.max())
        prob_range = max(prob_max - prob_min, 1e-12)

        lines: list[str] = []
        lines.append("BELIEF HEATMAP (final)")
        lines.append("  Intensity scale: ' '=low  '@'=high")
        lines.append("   " + "".join(str(x % 10) for x in range(width)))
        lines.append("   " + "-" * width)

        for y in range(height):
            row_chars: list[str] = []
            for x in range(width):
                prob = float(belief_array[y, x])
                norm = (prob - prob_min) / prob_range
                idx = min(int(norm * len(_HEATMAP_CHARS)), len(_HEATMAP_CHARS) - 1)
                row_chars.append(_HEATMAP_CHARS[idx])
            lines.append(f"{y:2d}|{''.join(row_chars)}")

        lines.append("")
        lines.append(f"  min={prob_min:.6f}  max={prob_max:.6f}")
        lines.append(_BELIEF_LEGEND)

        return "\n".join(lines)

    def save(self, output_path: str) -> str:
        """Render the map and save it to *output_path*.

        Parameters
        ----------
        output_path:
            Filesystem path for the output ``.txt`` file.  Parent directories
            are created automatically.

        Returns
        -------
        str
            The absolute path of the written file.
        """
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        rendered = self.render()
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(rendered)
            fh.write("\n")

        return os.path.abspath(output_path)

    def save_belief_heatmap(self, output_path: str) -> str:
        """Render and save the belief heatmap to *output_path*.

        Parameters
        ----------
        output_path:
            Filesystem path for the belief heatmap ``.txt`` file.

        Returns
        -------
        str
            The absolute path of the written file.
        """
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        rendered = self.render_belief_heatmap()
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(rendered)
            fh.write("\n")

        return os.path.abspath(output_path)

    def save_snapshot(self, output_path: str, step: int, strategy: str) -> str:
        """Save a labeled snapshot of the current belief heatmap.

        Parameters
        ----------
        output_path:
            Where to write the snapshot file.
        step:
            Current mission step number.
        strategy:
            Current strategy name.

        Returns
        -------
        str
            Absolute path of the written file.
        """
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        belief_array = self._belief.as_array()
        height, width = belief_array.shape

        prob_min = float(belief_array.min())
        prob_max = float(belief_array.max())
        prob_range = max(prob_max - prob_min, 1e-12)

        lines: list[str] = []
        lines.append(f"BELIEF SNAPSHOT — step {step}  strategy={strategy}")
        lines.append("   " + "".join(str(x % 10) for x in range(width)))
        lines.append("   " + "-" * width)

        for y in range(height):
            row_chars: list[str] = []
            for x in range(width):
                prob = float(belief_array[y, x])
                norm = (prob - prob_min) / prob_range
                idx = min(int(norm * len(_HEATMAP_CHARS)), len(_HEATMAP_CHARS) - 1)
                row_chars.append(_HEATMAP_CHARS[idx])
            lines.append(f"{y:2d}|{''.join(row_chars)}")

        lines.append("")
        lines.append(f"  min={prob_min:.6f}  max={prob_max:.6f}")

        content = "\n".join(lines) + "\n"
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        return os.path.abspath(output_path)

    def print_snapshot(self, label: str) -> None:
        """Print a labeled snapshot of the current map to stdout.

        Parameters
        ----------
        label:
            A short description shown as a header above the map.
        """
        separator = "=" * (self._env.width + 6)
        print(separator)
        print(f"  SNAPSHOT: {label}")
        print(separator)
        print(self.render())
        print(separator)
