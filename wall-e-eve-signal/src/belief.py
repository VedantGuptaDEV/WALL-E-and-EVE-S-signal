"""
belief.py
=========
Probabilistic belief grid for the WALL-E EVE Signal mission.

The :class:`BeliefGrid` maintains a 2-D numpy probability map representing
the agent's current best estimate of EVE's location.  It supports Bayesian
updates driven by scan observations and direct clue detections.

Key design choices
------------------
* **Initialisation** – the prior is a uniform base plus a Gaussian lobe
  pointed in the signal direction, scaled by ``initial_signal.strength``.
  A strong signal (0.8) creates a tight, prominent lobe; a weak one (0.4)
  barely perturbs the uniform base.
* **Positive scan update** – when a clue is detected in direction *d* with
  strength *s*, cells along that bearing receive a multiplicative likelihood
  boost that falls off with angular distance from *d*.
* **Negative scan update** – when nothing is found within scan_radius, cells
  inside that radius are suppressed (multiplied by a factor < 1).
* **Movement negative update** – stepping onto a cell without finding EVE
  removes that cell (probability → near-zero).
* **Clue update** – finding a physical clue adds a Gaussian boost centred
  on the clue position, weighted by clue strength.
* Every update ends with :meth:`normalize` to keep probabilities summing to 1.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Direction → unit vector mapping (dx right, dy down in screen coords)
# ---------------------------------------------------------------------------

_DIR_VECTORS: dict[str, tuple[float, float]] = {
    "N":  (0.0, -1.0),
    "S":  (0.0,  1.0),
    "E":  (1.0,  0.0),
    "W":  (-1.0, 0.0),
    "NE": (0.7071, -0.7071),
    "NW": (-0.7071, -0.7071),
    "SE": (0.7071,  0.7071),
    "SW": (-0.7071,  0.7071),
}


class BeliefGrid:
    """2-D probabilistic belief map over a grid.

    Probabilities represent the agent's estimate of the likelihood that EVE
    is located at each cell.  The grid is kept normalised so values sum to 1.

    Parameters
    ----------
    width:
        Number of grid columns.
    height:
        Number of grid rows.
    initial_signal:
        Either a compass direction string (e.g. ``'NE'``) or a coordinate
        pair ``[x, y]`` indicating where the initial EVE signal came from.
    seed:
        RNG seed for reproducibility.
    signal_strength:
        Strength of the initial signal in [0, 1].  Scales how strongly the
        prior is biased toward the signal direction.  Defaults to 0.5.
    """

    def __init__(
        self,
        width: int,
        height: int,
        initial_signal: Union[str, list[int]],
        seed: int,
        signal_strength: float = 0.5,
    ) -> None:
        """Initialise the belief grid from the initial signal."""
        self._width = width
        self._height = height
        self._rng = np.random.default_rng(seed)
        self._signal_strength = float(signal_strength)

        # Start with a uniform prior — every cell equally likely
        self._grid: np.ndarray = np.ones((height, width), dtype=np.float64)

        # Agent always starts at (0, 0)
        start_x, start_y = 0, 0

        # Apply the initial signal bias
        self._apply_initial_signal(initial_signal, start_x, start_y)
        self.normalize()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _apply_initial_signal(
        self,
        initial_signal: Union[str, list[int]],
        start_x: int,
        start_y: int,
    ) -> None:
        """Bias the prior toward the signal direction/location.

        The bias weight is scaled by ``self._signal_strength`` so that a
        weak signal barely perturbs the uniform base while a strong signal
        (near 1.0) creates a concentrated lobe.

        For a direction string, we use a Gaussian lobe centred on the far
        corner in that direction rather than a simple dot-product, so that
        cells far from the start in the signal direction receive the highest
        probability (not just cells adjacent to the start).

        For coordinate hints, a Gaussian centred on the given point is added.
        """
        # How much the signal can multiply the uniform base
        # strength=1.0 → max_boost=9× (dominant lobe)
        # strength=0.4 → max_boost=3.6× (moderate lean)
        max_boost = 1.0 + 8.0 * self._signal_strength

        if isinstance(initial_signal, str):
            direction = initial_signal.upper()
            vec = _DIR_VECTORS.get(direction, (1.0, 0.0))

            # Use the projection of each cell onto the signal direction as the
            # score.  The Gaussian lobe is centred at the midpoint of the far
            # edge in the signal direction so that, e.g., direction='E' peaks
            # along the *entire* right column, not just its top corner.
            #
            # mid_x / mid_y: the centre of the grid
            mid_x = (self._width - 1) / 2.0
            mid_y = (self._height - 1) / 2.0

            # Far-edge midpoint in the signal direction
            tx = mid_x + vec[0] * (max(self._width, self._height) / 2.0)
            ty = mid_y + vec[1] * (max(self._width, self._height) / 2.0)

            # Gaussian sigma: covers ~1/3 of the grid
            sigma = max(self._width, self._height) / 3.0

            # Vectorised computation using numpy
            xs = np.arange(self._width, dtype=np.float64)
            ys = np.arange(self._height, dtype=np.float64)
            xx, yy = np.meshgrid(xs, ys)  # shape (H, W)

            # Project each cell onto the signal direction vector
            # cells_dot = (xx - mid_x) * vec[0] + (yy - mid_y) * vec[1]
            # Then the Gaussian is centred at the max-projection distance
            proj = (xx - mid_x) * vec[0] + (yy - mid_y) * vec[1]
            max_proj = abs(vec[0]) * (self._width / 2.0) + abs(vec[1]) * (self._height / 2.0)

            # Gaussian weight based on projection (favours the far half-space)
            # Normalise projection to [0,1] then apply Gaussian around 1.0
            norm_proj = proj / max(max_proj, 1e-6)
            dist_sq = (norm_proj - 1.0) ** 2
            sigma_norm = 0.5  # spread in normalised projection space
            gauss = np.exp(-dist_sq / (2.0 * sigma_norm ** 2))

            # Scale: add between 0 and max_boost-1 to the uniform 1.0
            self._grid += (max_boost - 1.0) * gauss

        else:
            # Coordinate hint — Gaussian centred on signal origin
            sx, sy = float(initial_signal[0]), float(initial_signal[1])
            sigma = max(self._width, self._height) / 4.0

            xs = np.arange(self._width, dtype=np.float64)
            ys = np.arange(self._height, dtype=np.float64)
            xx, yy = np.meshgrid(xs, ys)

            dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
            gauss = np.exp(-dist_sq / (2.0 * sigma ** 2))
            self._grid += (max_boost - 1.0) * gauss

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def update_from_scan(
        self,
        agent_pos: list[int],
        scan_result: dict,
        scan_radius: int,
        sensor_noise: float,
    ) -> None:
        """Bayesian update from a scan observation.

        * **eve_detected** – concentrate probability inside scan_radius
          (boost × 8) and suppress everything outside (× 0.1).
        * **clue detected** – directional cosine-similarity boost.
          Cells in the bearing half-plane at distance >= 1 receive a
          multiplicative boost of ``1 + reliability * dot * distance_weight``
          where ``distance_weight`` falls off as 1/(1+dist/scan_radius) so
          closer cells along the bearing get a stronger boost.
        * **nothing found** – gentle suppression inside scan_radius only.
        """
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        clue_strength: float = float(scan_result.get("clue_strength", 0.0))
        clue_direction: Optional[str] = scan_result.get("clue_direction")
        eve_detected: bool = bool(scan_result.get("eve_detected", False))

        if eve_detected:
            # Hard concentrate: EVE is definitely nearby
            for y in range(max(0, ay - scan_radius), min(self._height, ay + scan_radius + 1)):
                for x in range(max(0, ax - scan_radius), min(self._width, ax + scan_radius + 1)):
                    self._grid[y, x] *= 8.0
            for y in range(self._height):
                for x in range(self._width):
                    if max(abs(x - ax), abs(y - ay)) > scan_radius:
                        self._grid[y, x] *= 0.1
            self.normalize()
            return

        if clue_strength > 0.0 and clue_direction is not None:
            vec = _DIR_VECTORS.get(clue_direction.upper(), (1.0, 0.0))
            # reliability: higher strength + lower noise = more trustworthy direction
            reliability = clue_strength * (1.0 - sensor_noise * 0.5)

            # Vectorised: compute cosine similarity of each cell relative to agent
            xs = np.arange(self._width, dtype=np.float64) - ax
            ys = np.arange(self._height, dtype=np.float64) - ay
            dx, dy = np.meshgrid(xs, ys)           # shape (H, W)
            lengths = np.sqrt(dx ** 2 + dy ** 2)
            lengths[lengths == 0] = 1.0             # agent cell: no update

            dot = (dx * vec[0] + dy * vec[1]) / lengths   # cosine similarity

            # Distance weight: cells close to scan_radius (the "edge" of sensor
            # reach) are more informative than cells far away.
            dist_weight = 1.0 / (1.0 + lengths / max(1, scan_radius))

            # Boost cells in forward half-plane at distance >= 1
            dist_mask = lengths >= 1.0
            boost = np.where(
                (dot > 0) & dist_mask,
                1.0 + reliability * dot * (1.0 + dist_weight),
                1.0,
            )
            self._grid *= boost

        else:
            # Negative scan: nothing found in radius — gentle suppression only
            suppression = 0.55 + 0.25 * sensor_noise  # 0.55 (no noise) .. 0.80 (noisy)
            for y in range(max(0, ay - scan_radius), min(self._height, ay + scan_radius + 1)):
                for x in range(max(0, ax - scan_radius), min(self._width, ax + scan_radius + 1)):
                    self._grid[y, x] *= suppression

        self.normalize()

    def update_from_clue(self, clue_pos: list[int], clue_strength: float) -> None:
        """Boost belief probability around a discovered clue position.

        A Gaussian kernel centred on *clue_pos* is multiplied into the
        current belief, weighted by *clue_strength*.

        Parameters
        ----------
        clue_pos:
            [x, y] position of the discovered clue.
        clue_strength:
            Strength of the clue in [0, 1].
        """
        cx, cy = float(clue_pos[0]), float(clue_pos[1])
        # Stronger clues → tighter, taller spike
        sigma = 1.5 + (1.0 - clue_strength) * 2.0  # range [1.5, 3.5]
        peak = 1.0 + clue_strength * 4.0            # range [1.0, 5.0]

        xs = np.arange(self._width, dtype=np.float64)
        ys = np.arange(self._height, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)

        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        boost = 1.0 + (peak - 1.0) * np.exp(-dist_sq / (2.0 * sigma ** 2))
        self._grid *= boost
        self.normalize()

    def update_negative(self, agent_pos: list[int], radius: int) -> None:
        """Reduce probability in the area the agent just visited without EVE.

        Parameters
        ----------
        agent_pos:
            Agent's current [x, y] position (centre of cleared area).
        radius:
            Radius of the cleared area in cells.
        """
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        # The cell the agent stands on is nearly zeroed; surrounding cells
        # are suppressed in proportion to their distance
        for y in range(max(0, ay - radius), min(self._height, ay + radius + 1)):
            for x in range(max(0, ax - radius), min(self._width, ax + radius + 1)):
                dist = max(abs(x - ax), abs(y - ay))
                if dist == 0:
                    # Agent's cell: almost certainly not EVE
                    self._grid[y, x] *= 0.02
                else:
                    # Adjacent cells: moderately suppressed
                    self._grid[y, x] *= 0.5
        self.normalize()

    def collapse_on_found(self, eve_pos: list[int]) -> None:
        """Collapse the entire belief mass onto the confirmed EVE cell.

        Called once EVE is physically located.  Sets the probability of
        ``eve_pos`` to 1.0 and all other cells to the minimum floor value,
        then normalises.

        Parameters
        ----------
        eve_pos:
            Confirmed EVE position as ``[x, y]``.
        """
        ex, ey = int(eve_pos[0]), int(eve_pos[1])
        # Set everything to near-zero
        self._grid[:, :] = 1e-12
        # Spike the confirmed cell
        self._grid[ey, ex] = 1.0
        self.normalize()

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def top_cells(self, n: int = 3) -> list[tuple[int, int, float]]:
        """Return the *n* cells with the highest belief probability.

        Parameters
        ----------
        n:
            Number of top cells to return.

        Returns
        -------
        list of (x, y, prob)
            Sorted in descending order of probability.
        """
        flat = self._grid.flatten()
        k = min(n, len(flat))
        top_indices = np.argpartition(flat, -k)[-k:]
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
        result: list[tuple[int, int, float]] = []
        for idx in top_indices:
            y, x = divmod(int(idx), self._width)
            result.append((x, y, float(self._grid[y, x])))
        return result

    def entropy(self) -> float:
        """Compute the Shannon entropy of the belief distribution.

        Returns
        -------
        float
            Entropy in nats.  Lower values indicate higher certainty.
        """
        flat = self._grid.flatten()
        nonzero = flat[flat > 0]
        return float(-np.sum(nonzero * np.log(nonzero)))

    def normalize(self) -> None:
        """Normalise the grid so that all probabilities sum to 1.

        If the total is zero (degenerate case), resets to a uniform
        distribution.  Every cell is clamped to a minimum of 1e-6 so that
        suppressed cells can always recover with future positive evidence.
        """
        # Clamp to minimum floor so no cell is ever permanently ruled out
        self._grid = np.maximum(self._grid, 1e-6)
        total = float(self._grid.sum())
        if total <= 0:
            self._grid = np.ones((self._height, self._width), dtype=np.float64)
            total = float(self._grid.sum())
        self._grid /= total

    def as_array(self) -> np.ndarray:
        """Return a copy of the probability grid as a numpy array.

        Returns
        -------
        np.ndarray
            Shape ``(height, width)``, dtype ``float64``, values in ``[0, 1]``
            summing to 1.
        """
        return self._grid.copy()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Grid width."""
        return self._width

    @property
    def height(self) -> int:
        """Grid height."""
        return self._height
