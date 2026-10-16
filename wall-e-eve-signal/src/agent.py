"""
agent.py
========
Autonomous agent for the WALL-E EVE Signal mission.

Decision policy
---------------
All actions are scored on a common [0, 1] scale so they compete fairly.

Move utility
    U_move = (belief_mass_in_scan_radius(dest) / BFS_cost_to_dest)
             * direction_bonus          (1.2 toward locked target, 0.8 away)
             * (1 - revisit_penalty)    (based on recent path history)
             * battery_frac

    The key insight: belief_mass_in_scan_radius(dest) is the total probability
    mass the agent would cover if it scanned FROM dest.  Dividing by BFS path
    cost converts "information per step" so that a nearby mediocre cell beats
    a distant slightly-better cell only if the BFS cost difference is large
    enough to justify it.

Scan utility
    U_scan = belief_mass(agent, r=SCAN_RADIUS) * battery_frac
             [+ 0.25 bonus if scan is overdue by _SCAN_INTERVAL steps]
    Hard cap: at most _SCAN_CAP total scans from the same cell.

Recharge utility
    U_recharge = (1 - battery_frac) * 0.8   [only on a recharge station]

Target locking
    Once a top belief cell is chosen as the target, the agent commits to
    navigating toward it until either (a) it arrives, (b) the cell's belief
    falls below a threshold, or (c) _TARGET_PATIENCE steps have passed.
    This prevents oscillation when two equi-probable cells flank the agent.

Anti-oscillation
    The previous cell is tracked; returning to it in one step scores 0
    unless it is the only safe option (safety retreat).

Safety constraint
    After every action battery must remain >= safety_reserve.
    safety_reserve = bfs_path_to_nearest_station * move_cost.
    (BFS routing avoids underestimating reserve when obstacles lie between
    the agent and the nearest station.)

Strategy tracking
    The agent reports its current strategy as one of:
      explore          — spreading belief, no strong target
      focused_search   — converging toward a high-probability target
      recharge_detour  — heading toward a recharge station
    Strategy changes are logged and can trigger belief-heatmap snapshots.

Clue tracking
    When a scan returns a positive clue signal, the agent records the
    scan position.  When a scan returns a strong decoy signal that is later
    suppressed (belief drops fast after visiting), the position is marked
    as a discovered decoy.  Both sets are exposed for the visualizer.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

from .belief import BeliefGrid
from .config import Config
from .environment import Environment, Observation
from .logger import JSONLogger

_MOVES: tuple[str, ...] = ("N", "S", "E", "W")
_DELTAS: dict[str, tuple[int, int]] = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}

_RECHARGE_THRESHOLD: float = 0.30   # force-recharge when on station below this
_SCAN_CAP: int = 2                   # max total scans per cell
_HISTORY_LEN: int = 20               # recent steps used for revisit penalty
_SCAN_INTERVAL: int = 4              # periodic forced scan interval
_TARGET_PATIENCE: int = 6            # steps before we reconsider the locked target
_EXPLORE_BONUS: float = 0.15         # small bonus for moving to a completely unvisited cell
_SUPPRESS_RADIUS: int = 3            # radius to suppress when agent arrives at target without finding EVE
_DIAG_STEPS: int = 15                # number of initial steps to print full scoring diagnostics
_BFS_CACHE_SIZE: int = 64            # max cached BFS distances (LRU-style)


class Agent:
    """Utility-based autonomous agent for the WALL-E EVE signal search."""

    def __init__(
        self,
        config: Config,
        environment: Environment,
        belief: BeliefGrid,
        logger: JSONLogger,
    ) -> None:
        self._config = config
        self._env = environment
        self._belief = belief
        self._logger = logger

        # Visit counts and path history for anti-loop
        self._visited: dict[tuple[int, int], int] = {}
        self._path_history: deque[tuple[int, int]] = deque(maxlen=_HISTORY_LEN)
        self._prev_pos: Optional[tuple[int, int]] = None   # immediate previous cell

        # Scan tracking
        self._total_scans_at: dict[tuple[int, int], int] = {}
        self._steps_since_scan: int = _SCAN_INTERVAL  # start ready to scan

        # Locked navigation target
        self._locked_target: Optional[tuple[int, int]] = None
        self._target_age: int = 0

        # Replanning
        self._last_top_cell: Optional[tuple[int, int]] = None
        self._replanned: bool = False

        # EVE status
        self._clue_strengths: list[float] = []
        self._eve_status: str = "unknown"

        # Strategy tracking (for snapshot triggers)
        self._current_strategy: str = "explore"
        self._strategy_change_callbacks: list = []   # (callback(step, strategy))

        # Discovered clue positions (scan positions that returned positive clue signals)
        # Maps position -> max clue strength seen from that scan position
        self._discovered_clue_scans: dict[tuple[int, int], float] = {}

        # Positions that had strong initial clue signal but were later suppressed
        # (potential decoys — visited without finding EVE after strong clue)
        self._discovered_decoys: set[tuple[int, int]] = set()

        # BFS distance cache: (start, goal) -> distance
        self._bfs_cache: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

        # Step counter for diagnostics
        self._step: int = 0

    # ------------------------------------------------------------------
    # Public accessors for visualization
    # ------------------------------------------------------------------

    @property
    def discovered_clue_positions(self) -> dict[tuple[int, int], float]:
        """Scan positions where a positive clue signal was detected.
        Maps (x, y) -> max clue strength seen from there."""
        return dict(self._discovered_clue_scans)

    @property
    def discovered_decoy_positions(self) -> set[tuple[int, int]]:
        """Positions that showed a clue signal but turned out to be decoys."""
        return set(self._discovered_decoys)

    @property
    def current_strategy(self) -> str:
        return self._current_strategy

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run the mission loop until the environment signals done."""
        step = 0
        while not self._env.done:
            step += 1
            self._step = step
            self._replanned = False

            pos: tuple[int, int] = (self._env.agent_pos[0], self._env.agent_pos[1])
            self._visited[pos] = self._visited.get(pos, 0) + 1
            self._path_history.append(pos)

            action, reasoning = self._decide_action()
            obs = self._execute_action(action)
            self._update_belief(obs, action)
            self._update_eve_status(obs)
            self._check_replan()
            self._update_locked_target()

            # Update strategy
            old_strategy = self._current_strategy
            self._update_strategy()
            if self._current_strategy != old_strategy:
                # Notify strategy-change subscribers (used for belief snapshots)
                for cb in self._strategy_change_callbacks:
                    cb(step, self._current_strategy)
                reasoning += f" [STRATEGY->{self._current_strategy}]"

            # Periodically suppress heavily-visited cells
            if step % 10 == 0:
                self._suppress_overvisited_cells()

            # Collapse belief onto confirmed EVE cell the moment it's found
            if self._env.found_eve:
                confirmed_pos = obs.get("agent_pos", self._env.agent_pos)
                self._belief.collapse_on_found(confirmed_pos)

            # Track scan counter + clue recording
            if action == "scan":
                self._total_scans_at[pos] = self._total_scans_at.get(pos, 0) + 1
                self._steps_since_scan = 0
                # Record the clue signal
                scan_result = obs.get("scan_result", {})
                cs = float(scan_result.get("clue_strength", 0.0))
                if cs > 0.0:
                    prev = self._discovered_clue_scans.get(pos, 0.0)
                    self._discovered_clue_scans[pos] = max(prev, cs)
            else:
                self._prev_pos = pos
                self._steps_since_scan += 1

            top_raw = self._belief.top_cells(n=5)
            belief_summary = {
                "top_cells": [
                    {"x": x, "y": y, "prob": round(p, 6)} for x, y, p in top_raw
                ],
                "entropy": round(self._belief.entropy(), 4),
            }

            full_reasoning = reasoning
            if self._replanned:
                full_reasoning += " [REPLAN: belief shifted — target unlocked.]"

            self._logger.log_step(
                step=step,
                action=action,
                observation=dict(obs),
                belief_summary=belief_summary,
                decision=action,
                reasoning=full_reasoning,
            )

        top_cells = self._belief.top_cells(n=1)
        eve_estimate: Optional[list[int]] = (
            [top_cells[0][0], top_cells[0][1]] if top_cells else None
        )
        confirmed_pos: Optional[list[int]] = None
        if self._env.found_eve:
            ap = self._env.agent_pos
            confirmed_pos = [int(ap[0]), int(ap[1])]
            eve_estimate = confirmed_pos

        return {
            "found_eve": self._env.found_eve,
            "steps": self._env.steps_taken,
            "battery_remaining": self._env.battery,
            "eve_location_estimate": eve_estimate,
            "confirmed_eve_pos": confirmed_pos,
        }

    # ------------------------------------------------------------------
    # Strategy tracking
    # ------------------------------------------------------------------

    def _update_strategy(self) -> None:
        """Update the current strategy label."""
        battery_frac = self._env.battery / max(1, self._config.resources.battery)
        safety_res = self._safety_reserve()
        battery_gap = self._env.battery - safety_res

        if battery_gap <= self._config.resources.move_cost * 3:
            self._current_strategy = "recharge_detour"
            return

        top_cells = self._belief.top_cells(n=1)
        if top_cells:
            top_prob = top_cells[0][2]
            entropy = self._belief.entropy()
            # High-probability target and low entropy = focused search
            n = self._config.map.width * self._config.map.height
            uniform_prob = 1.0 / max(n, 1)
            if top_prob > 10 * uniform_prob:
                self._current_strategy = "focused_search"
                return

        self._current_strategy = "explore"

    def register_strategy_change_callback(self, callback) -> None:
        """Register a callback(step, new_strategy) for strategy changes."""
        self._strategy_change_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Target locking
    # ------------------------------------------------------------------

    def _update_locked_target(self) -> None:
        """Maintain the locked navigation target."""
        top_cells = self._belief.top_cells(n=1)
        if not top_cells:
            self._locked_target = None
            return

        new_top: tuple[int, int] = (top_cells[0][0], top_cells[0][1])

        if self._locked_target is not None:
            ax, ay = self._env.agent_pos
            lx, ly = self._locked_target
            at_target = (ax == lx and ay == ly)
            patience_exceeded = self._target_age >= _TARGET_PATIENCE

            if at_target and not self._env.found_eve:
                # Arrived but no EVE — suppress and mark as potential decoy
                self._belief.update_negative(
                    agent_pos=[lx, ly],
                    radius=_SUPPRESS_RADIUS,
                )
                # If we had a clue signal from here, it was a decoy
                if (lx, ly) in self._discovered_clue_scans:
                    self._discovered_decoys.add((lx, ly))
                self._locked_target = None
                self._target_age = 0

            elif patience_exceeded:
                self._locked_target = None
                self._target_age = 0

        if self._locked_target is None:
            self._locked_target = new_top
            self._target_age = 0
        else:
            self._target_age += 1

    def _check_replan(self) -> None:
        """Reset the locked target when belief shifts significantly."""
        top_cells = self._belief.top_cells(n=1)
        if not top_cells:
            return
        new_top: tuple[int, int] = (top_cells[0][0], top_cells[0][1])
        if self._last_top_cell is None:
            self._last_top_cell = new_top
            return
        dist = abs(new_top[0] - self._last_top_cell[0]) + abs(new_top[1] - self._last_top_cell[1])
        if dist > 3:
            self._visited = {k: max(0, v // 2) for k, v in self._visited.items()}
            self._path_history.clear()
            self._locked_target = None
            self._target_age = 0
            self._last_top_cell = new_top
            self._replanned = True

    def _suppress_overvisited_cells(self) -> None:
        """Suppress belief on cells visited many times without finding EVE."""
        for (cx, cy), count in self._visited.items():
            if count >= 3:
                self._belief.update_negative(agent_pos=[cx, cy], radius=1)

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide_action(self) -> tuple[str, str]:
        """Score all candidate actions; return (action, one-line reasoning).

        The key scoring change vs. the old version:
        - Move score = (belief_mass_in_scan_radius(dest) / bfs_cost_to_dest)
                       * direction_bonus * (1 - revisit_penalty) * battery_frac
          This is "information per step" — it normalises by travel cost so
          the agent doesn't just march to the farthest high-belief corner.
        - direction_bonus targets the BFS-nearest high-belief cell (the
          "informed target"), not just the global top cell.
        - explore_bonus is small (0.15) and only kicks in for completely
          unvisited cells — it no longer overwhelms the belief signal.
        """
        battery = self._env.battery
        ax, ay = self._env.agent_pos
        max_battery = self._config.resources.battery
        move_cost = self._config.resources.move_cost
        scan_cost = self._config.resources.scan_cost
        battery_frac = battery / max(1, max_battery)
        pos = (ax, ay)

        if battery <= 0:
            return "end_mission", "Battery depleted — ending mission."

        safety_reserve = self._safety_reserve()

        if battery <= safety_reserve:
            d, r = self._retreat_toward_station()
            if d:
                return d, r
            return "end_mission", (
                f"Battery {battery} at/below safety reserve {safety_reserve} — "
                "ending mission."
            )

        on_station = self._env.is_recharge_station(ax, ay)

        if on_station and battery_frac <= _RECHARGE_THRESHOLD:
            return "recharge", (
                f"On station at ({ax},{ay}), battery {battery_frac*100:.0f}% "
                f"[entropy={self._belief.entropy():.2f}] — recharging before exploring."
            )

        belief_array = self._belief.as_array()
        top_cells = self._belief.top_cells(n=3)
        if not top_cells:
            return "end_mission", "Belief grid empty — ending mission."

        # Use locked target if available; otherwise fall back to top cell
        if self._locked_target is not None:
            target_x, target_y = self._locked_target
            target_prob = float(belief_array[target_y, target_x])
        else:
            target_x, target_y, target_prob = top_cells[0]

        # BFS distance to target (to compute direction bonus correctly)
        target_bfs = self._bfs_distance((ax, ay), (target_x, target_y))

        best_action: Optional[str] = None
        best_score: float = -math.inf
        best_reason: str = ""

        # Collect diagnostic info for first N steps
        diag_lines: list[str] = []
        is_diag = self._step <= _DIAG_STEPS

        if is_diag:
            diag_lines.append(
                f"\n{'='*60}\n"
                f"STEP {self._step} DIAGNOSTIC  agent=({ax},{ay})  "
                f"battery={battery}/{max_battery}  "
                f"entropy={self._belief.entropy():.3f}\n"
                f"  Top-3 belief cells:"
            )
            for i, (bx, by, bp) in enumerate(top_cells):
                bfs_to = self._bfs_distance((ax, ay), (bx, by))
                diag_lines.append(
                    f"    #{i+1}: ({bx:2d},{by:2d}) p={bp:.6f}  "
                    f"bfs={bfs_to}  "
                    f"mass_r2={self._belief_mass_around(bx,by,2,belief_array):.4f}"
                )
            diag_lines.append(
                f"  Locked target: ({target_x},{target_y}) p={target_prob:.6f}  "
                f"bfs={target_bfs}"
            )
            diag_lines.append("  Move scores:")

        # ---- Score moves ----
        for direction, (ddx, ddy) in _DELTAS.items():
            nx, ny = ax + ddx, ay + ddy
            if self._env.is_obstacle(nx, ny):
                if is_diag:
                    diag_lines.append(f"    {direction} -> ({nx:2d},{ny:2d}): BLOCKED")
                continue
            if battery - move_cost < safety_reserve:
                if is_diag:
                    diag_lines.append(f"    {direction} -> ({nx:2d},{ny:2d}): UNSAFE (battery)")
                continue

            if (nx, ny) == self._prev_pos:
                score = -0.5
                if is_diag:
                    diag_lines.append(
                        f"    {direction} -> ({nx:2d},{ny:2d}): prev_pos_penalty  score={score:.4f}"
                    )
            else:
                # Information gain: belief mass within scan radius of destination
                from .environment import SCAN_RADIUS
                info_gain = self._belief_mass_around(nx, ny, radius=SCAN_RADIUS, belief=belief_array)

                # BFS cost to reach destination (always 1 for adjacent cells)
                bfs_cost = self._bfs_distance((ax, ay), (nx, ny))

                # Information per step: normalise by BFS cost
                info_per_step = info_gain / max(1, bfs_cost)

                # Direction bonus: reward moves that close in on the locked target.
                # Use a softer scale (1.2/0.85) so moves away from the target
                # don't score too harshly when local belief is still reasonable.
                new_bfs = self._bfs_distance((nx, ny), (target_x, target_y))
                direction_bonus = 1.2 if new_bfs < target_bfs else 0.85

                # Recent-visit anti-loop penalty
                recent = sum(1 for h in self._path_history if h == (nx, ny))
                revisit_penalty = min(0.80, recent * 0.20)

                # Exploration bonus: small reward for completely unvisited cells
                explore_bonus = _EXPLORE_BONUS if self._visited.get((nx, ny), 0) == 0 else 0.0

                # Absolute belief floor: if info_gain for this destination is
                # extremely low (less than half the uniform expected mass in a
                # SCAN_RADIUS window), penalise — the area has been well-cleared.
                n_cells = self._config.map.width * self._config.map.height
                window_size = (2 * SCAN_RADIUS + 1) ** 2
                expected_uniform_mass = window_size / max(n_cells, 1)
                low_belief_penalty = 0.15 if info_gain < 0.3 * expected_uniform_mass else 0.0

                score = (
                    info_per_step
                    * direction_bonus
                    * (1.0 - revisit_penalty)
                    * battery_frac
                    + explore_bonus
                    - low_belief_penalty
                )

                if is_diag:
                    diag_lines.append(
                        f"    {direction} -> ({nx:2d},{ny:2d}): "
                        f"info={info_gain:.4f}  bfs={bfs_cost}  "
                        f"info/step={info_per_step:.4f}  "
                        f"dir_bonus={direction_bonus}  "
                        f"revisit×{recent}(pen={revisit_penalty:.2f})  "
                        f"explore={explore_bonus:.2f}  "
                        f"low_belief={low_belief_penalty:.2f}  "
                        f"TOTAL={score:.4f}"
                    )

            if score > best_score:
                best_score = score
                best_action = direction
                loop_note = ""
                if (nx, ny) != self._prev_pos:
                    recent_here = sum(1 for h in self._path_history if h == (nx, ny))
                    if recent_here > 0:
                        loop_note = f" (recent×{recent_here})"
                    elif self._visited.get((nx, ny), 0) == 0:
                        loop_note = " (unvisited)"
                best_reason = (
                    f"Moving {direction} to ({nx},{ny}){loop_note}: "
                    f"target=({target_x},{target_y}) p={target_prob:.4f} "
                    f"battery={battery} entropy={self._belief.entropy():.2f} "
                    f"[age={self._target_age}]."
                )

        # ---- Score scan ----
        scans_here = self._total_scans_at.get(pos, 0)
        scan_reserve = safety_reserve + self._config.resources.move_cost
        if battery - scan_cost >= scan_reserve and scans_here < _SCAN_CAP:
            from .environment import SCAN_RADIUS
            p_nearby = self._belief_mass_around(ax, ay, radius=SCAN_RADIUS, belief=belief_array)
            scan_score = p_nearby * battery_frac

            if self._steps_since_scan >= _SCAN_INTERVAL:
                scan_score = _EXPLORE_BONUS + 0.20 + p_nearby
                interval_note = f" [overdue: {self._steps_since_scan} steps — forced]"
            else:
                interval_note = ""

            if is_diag:
                diag_lines.append(
                    f"    SCAN from ({ax:2d},{ay:2d}): "
                    f"p_nearby={p_nearby:.4f}  "
                    f"battery_frac={battery_frac:.3f}  "
                    f"overdue={self._steps_since_scan>= _SCAN_INTERVAL}  "
                    f"TOTAL={scan_score:.4f}"
                )

            if scan_score > best_score:
                best_score = scan_score
                best_action = "scan"
                best_reason = (
                    f"Scanning from ({ax},{ay}){interval_note}: "
                    f"nearby belief mass={p_nearby:.4f}  "
                    f"battery={battery}  entropy={self._belief.entropy():.2f}."
                )

        # ---- Opportunistic recharge ----
        if on_station and battery_frac < 0.65:
            recharge_score = (1.0 - battery_frac) * 0.8
            if is_diag:
                diag_lines.append(
                    f"    RECHARGE at ({ax:2d},{ay:2d}): "
                    f"battery_frac={battery_frac:.3f}  "
                    f"TOTAL={recharge_score:.4f}"
                )
            if recharge_score > best_score:
                best_score = recharge_score
                best_action = "recharge"
                best_reason = (
                    f"On station at ({ax},{ay}), battery {battery_frac*100:.0f}% "
                    f"[entropy={self._belief.entropy():.2f}] — recharging opportunistically."
                )

        if is_diag:
            diag_lines.append(f"  >> CHOSEN: {best_action}  score={best_score:.4f}")
            diag_lines.append("=" * 60)
            print("\n".join(diag_lines))

        if best_action is None:
            return "end_mission", "No safe action available — ending mission."

        return best_action, best_reason

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _belief_mass_around(
        self,
        cx: int,
        cy: int,
        radius: int,
        belief: np.ndarray,
    ) -> float:
        """Total belief probability in a Chebyshev window around (cx, cy)."""
        h, w = belief.shape
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        return float(belief[y0:y1, x0:x1].sum())

    def _retreat_toward_station(self) -> tuple[Optional[str], str]:
        nearest = self._nearest_recharge_station()
        if nearest is None:
            return None, "No recharge station configured."
        ax, ay = self._env.agent_pos
        sx, sy = nearest
        best_dir: Optional[str] = None
        best_dist = self._bfs_distance((ax, ay), nearest)
        for direction, (ddx, ddy) in _DELTAS.items():
            nx, ny = ax + ddx, ay + ddy
            if self._env.is_obstacle(nx, ny):
                continue
            d = self._bfs_distance((nx, ny), nearest)
            if d < best_dist:
                best_dist = d
                best_dir = direction
        if best_dir:
            return best_dir, (
                f"Battery critically low ({self._env.battery}) — retreating "
                f"{best_dir} toward recharge station at ({sx},{sy})."
            )
        return None, "Trapped — no path to nearest recharge station."

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: str) -> Observation:
        if action in _MOVES:
            return self._env.move(action)
        if action == "scan":
            return self._env.scan()
        if action == "recharge":
            return self._env.recharge()
        return self._env.end_mission()

    # ------------------------------------------------------------------
    # Belief management
    # ------------------------------------------------------------------

    def _update_belief(self, obs: Observation, action: str) -> None:
        from .environment import SCAN_RADIUS
        scan_result = obs.get("scan_result", {})
        agent_pos = obs.get("agent_pos", self._env.agent_pos)

        if action == "scan":
            self._belief.update_from_scan(
                agent_pos=agent_pos,
                scan_result=scan_result,
                scan_radius=SCAN_RADIUS,
                sensor_noise=self._config.sensor_noise,
            )
            clue_s = float(scan_result.get("clue_strength", 0.0))
            clue_d = scan_result.get("clue_direction")
            if clue_s > 0.0 and clue_d is not None:
                self._clue_strengths.append(clue_s)
        elif action in _MOVES and not self._env.found_eve:
            self._belief.update_negative(agent_pos, radius=1)

    def _update_eve_status(self, obs: Observation) -> None:
        if not self._clue_strengths:
            self._eve_status = "unknown"
            return
        avg = sum(self._clue_strengths) / len(self._clue_strengths)
        self._eve_status = (
            "likely active" if avg > 0.6
            else "likely dormant" if avg < 0.3
            else "unknown"
        )

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _nearest_recharge_station(self) -> Optional[tuple[int, int]]:
        stations = self._config.resources.recharge_stations
        if not stations:
            return None
        ax, ay = self._env.agent_pos
        best: Optional[tuple[int, int]] = None
        best_dist = math.inf
        for st in stations:
            sx, sy = int(st[0]), int(st[1])
            bfs_d = self._bfs_distance((ax, ay), (sx, sy))
            if bfs_d < best_dist:
                best_dist = bfs_d
                best = (sx, sy)
        return best

    def _bfs_distance(self, start: tuple[int, int], goal: tuple[int, int]) -> int:
        """BFS shortest path distance from start to goal avoiding obstacles.

        Returns the true path length in steps, or a large fallback if no path
        exists.  Results are cached to avoid repeated BFS for the same pair.
        """
        if start == goal:
            return 0

        cache_key = (start, goal)
        if cache_key in self._bfs_cache:
            return self._bfs_cache[cache_key]

        # Evict oldest entries if cache is too large
        if len(self._bfs_cache) >= _BFS_CACHE_SIZE:
            # Remove half the cache entries
            keys = list(self._bfs_cache.keys())
            for k in keys[: len(keys) // 2]:
                del self._bfs_cache[k]

        queue: deque[tuple[tuple[int, int], int]] = deque()
        queue.append((start, 0))
        visited: set[tuple[int, int]] = {start}
        while queue:
            (cx, cy), dist = queue.popleft()
            for ddx, ddy in _DELTAS.values():
                nx, ny = cx + ddx, cy + ddy
                npos = (nx, ny)
                if npos == goal:
                    result = dist + 1
                    self._bfs_cache[cache_key] = result
                    return result
                if not self._env.is_obstacle(nx, ny) and npos not in visited:
                    visited.add(npos)
                    queue.append((npos, dist + 1))

        # No path found — return Manhattan as safe overestimate
        result = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        self._bfs_cache[cache_key] = result
        return result

    def _safety_reserve(self) -> int:
        """Battery floor = BFS_path_to_nearest_station * move_cost."""
        nearest = self._nearest_recharge_station()
        if nearest is None:
            return 0
        ax, ay = self._env.agent_pos
        bfs_dist = self._bfs_distance((ax, ay), nearest)
        return bfs_dist * self._config.resources.move_cost
