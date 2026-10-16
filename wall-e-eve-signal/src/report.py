"""
report.py
=========
Markdown report generator for a completed WALL-E EVE Signal mission run.
"""

from __future__ import annotations

import math
import os
from typing import Any

from .config import Config


class ReportGenerator:
    """Generates a Markdown mission report from a run trace.

    Parameters
    ----------
    config_name:
        Human-readable name of the configuration used.
    config:
        The :class:`~config.Config` instance.
    trace:
        List of step dicts from :meth:`~logger.JSONLogger.entries`.
    """

    def __init__(
        self,
        config_name: str,
        config: Config,
        trace: list[dict[str, Any]],
    ) -> None:
        self._config_name = config_name
        self._config = config
        self._trace = trace
        # Cache found-eve detection so every section sees the same answer
        self._found_eve: bool = any(
            "found_eve" in (e.get("observation", {}).get("event", "") or "")
            for e in self._trace
        )
        # Confirmed position: agent_pos at the step where EVE was found
        self._confirmed_pos: list[int] | None = None
        for e in self._trace:
            if "found_eve" in (e.get("observation", {}).get("event", "") or ""):
                self._confirmed_pos = e.get("observation", {}).get("agent_pos")
                break

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, output_path: str) -> str:
        """Write the Markdown report to *output_path*; return its absolute path."""
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(self._build_markdown())
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _build_markdown(self) -> str:
        return "\n".join([
            "# WALL-E EVE Signal -- Mission Report\n",
            self._section_mission_summary(),
            self._section_evidence_found(),
            self._section_inferred_eve_location(),
            self._section_resources(),
            self._section_recommended_next_step(),
        ])

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _section_mission_summary(self) -> str:
        last  = self._trace[-1] if self._trace else {}
        event = last.get("observation", {}).get("event", "unknown")
        steps = len(self._trace)
        grid  = f"{self._config.map.width} x {self._config.map.height}"
        result = "[FOUND] EVE Located" if self._found_eve else "[NOT FOUND] EVE Not Found"
        replan_count = sum(
            1 for e in self._trace if "[REPLAN" in (e.get("reasoning", "") or "")
        )
        lines = [
            "## 1. Mission Summary\n",
            "| Field            | Value                          |",
            "|------------------|--------------------------------|",
            f"| Config Name      | `{self._config_name}`          |",
            f"| Grid Size        | {grid}                         |",
            f"| Seed             | {self._config.map.seed}        |",
            f"| Total Steps      | {steps}                        |",
            f"| Mission Result   | {result}                       |",
            f"| Terminal Event   | `{event}`                      |",
            f"| Replans          | {replan_count}                 |",
            "",
        ]
        return "\n".join(lines)

    def _section_evidence_found(self) -> str:
        lines = [
            "## 2. Evidence Found\n",
            "| Step | Position (x,y) | Clue Direction | Clue Strength |",
            "|------|----------------|----------------|---------------|",
        ]
        count = 0
        for entry in self._trace:
            scan = entry.get("observation", {}).get("scan_result", {})
            s = float(scan.get("clue_strength") or 0.0)
            if s > 0.0:
                pos = entry.get("observation", {}).get("agent_pos", ["?", "?"])
                d   = scan.get("clue_direction", "?")
                lines.append(
                    f"| {entry.get('step','?')} | ({pos[0]}, {pos[1]}) | {d} | {s:.3f} |"
                )
                count += 1
        if count == 0:
            lines.append("| -- | -- | -- | No clue signals detected |")
        lines.append("")
        return "\n".join(lines)

    def _section_inferred_eve_location(self) -> str:
        top_cells, max_prob, entropy = self._extract_final_belief()
        confidence  = self._confidence_score(max_prob, entropy)
        eve_status  = self._eve_status()

        # When EVE is confirmed found, override everything with 1.0 / "confirmed"
        if self._found_eve and self._confirmed_pos:
            cx, cy = self._confirmed_pos[0], self._confirmed_pos[1]
            location_str   = f"({cx}, {cy}) -- CONFIRMED"
            prob_str        = "100.0%"
            confidence_pct  = "100.0%"
            status_str      = f"{eve_status} [confirmed]"
            entropy_str     = "0.0000"
        else:
            if top_cells:
                best = top_cells[0]
                location_str = f"({best['x']}, {best['y']})"
                prob_str      = f"{best['prob'] * 100:.1f}%"
            else:
                location_str = "Unknown"
                prob_str      = "0.0%"
            confidence_pct = f"{confidence * 100:.1f}%"
            status_str     = eve_status
            entropy_str    = f"{entropy:.4f}"

        lines = [
            "## 3. Inferred EVE Location and Status\n",
            "| Field                | Value                |",
            "|----------------------|----------------------|",
            f"| Most Likely Cell     | {location_str}       |",
            f"| Cell Probability     | {prob_str}           |",
            f"| Confidence Score     | {confidence_pct}     |",
            f"| Belief Entropy       | {entropy_str}        |",
            f"| EVE Status Estimate  | {status_str}         |",
            "",
        ]

        if top_cells and not self._found_eve:
            lines += [
                "### Top Belief Cells\n",
                "| Rank | Position (x,y) | Probability |",
                "|------|----------------|-------------|",
            ]
            for rank, cell in enumerate(top_cells, start=1):
                lines.append(
                    f"| {rank} | ({cell['x']}, {cell['y']}) | {cell['prob'] * 100:.2f}% |"
                )
            lines.append("")
        elif self._found_eve and self._confirmed_pos:
            cx, cy = self._confirmed_pos[0], self._confirmed_pos[1]
            lines += [
                "### Confirmed Location\n",
                "| Rank | Position (x,y) | Probability |",
                "|------|----------------|-------------|",
                f"| 1    | ({cx}, {cy})   | 100.00%     |",
                "",
            ]

        return "\n".join(lines)

    def _section_resources(self) -> str:
        max_battery = self._config.resources.battery
        time_limit  = self._config.resources.time_limit
        last        = self._trace[-1] if self._trace else {}
        bat_rem     = last.get("battery", max_battery)
        time_rem    = last.get("time_remaining", time_limit)
        bat_used    = max_battery - bat_rem
        time_used   = time_limit  - time_rem
        lines = [
            "## 4. Resources Used vs Remaining\n",
            "| Resource      | Used       | Remaining  | Limit      |",
            "|---------------|------------|------------|------------|",
            f"| Battery       | {bat_used:>10} | {bat_rem:>10} | {max_battery:>10} |",
            f"| Steps / Time  | {time_used:>10} | {time_rem:>10} | {time_limit:>10} |",
            "",
        ]
        return "\n".join(lines)

    def _section_recommended_next_step(self) -> str:
        top_cells, max_prob, entropy = self._extract_final_belief()
        confidence  = self._confidence_score(max_prob, entropy)
        last        = self._trace[-1] if self._trace else {}
        bat_rem     = last.get("battery", self._config.resources.battery)
        bat_pct     = bat_rem / max(1, self._config.resources.battery)

        # Confirmed
        if self._found_eve and self._confirmed_pos:
            cx, cy = self._confirmed_pos[0], self._confirmed_pos[1]
            rec  = f"EVE confirmed at ({cx}, {cy}) -- proceed to that cell"
            body = (
                f"EVE was physically located at ({cx}, {cy}) during this mission. "
                f"WALL-E should navigate directly there.\n\n"
                f"EVE operational status: **{self._eve_status()} [confirmed]**."
            )
        elif bat_pct < 0.20:
            rec  = "Recharge first -- battery critically low"
            body = (
                f"Battery is at {bat_pct * 100:.0f}% of maximum. "
                "Recharging is essential before any further exploration.\n\n"
                f"EVE operational status: **{self._eve_status()}**."
            )
        elif confidence > 0.70 and top_cells:
            best = top_cells[0]
            rec  = f"Navigate to ({best['x']}, {best['y']}) -- high confidence EVE location"
            body = (
                f"Belief grid converged to {confidence * 100:.0f}% confidence. "
                f"Cell ({best['x']}, {best['y']}) holds {best['prob'] * 100:.1f}% probability.\n\n"
                f"EVE operational status: **{self._eve_status()}**."
            )
        else:
            rec  = "Continue scanning -- uncertainty too high to commit"
            body = (
                f"Confidence is {confidence * 100:.0f}%, below the 70% threshold. "
                "Scan more cells to reduce entropy before committing.\n\n"
                f"EVE operational status: **{self._eve_status()}**."
            )

        return "\n".join([
            "## 5. Recommended Next Step for WALL-E\n",
            f"> **{rec}**\n",
            body,
            "",
        ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_final_belief(self) -> tuple[list[dict[str, Any]], float, float]:
        """Return (top_cells, max_prob, entropy) from the last trace entry."""
        if not self._trace:
            return [], 0.0, 0.0
        last       = self._trace[-1]
        belief     = last.get("belief_summary", {})
        top_cells  = belief.get("top_cells", [])
        entropy    = float(belief.get("entropy", 0.0))
        max_prob   = float(top_cells[0]["prob"]) if top_cells else 0.0
        return top_cells, max_prob, entropy

    def _confidence_score(self, max_prob: float, entropy: float) -> float:
        """Confidence in [0, 1].

        * When EVE is confirmed found: always returns 1.0.
        * Otherwise: average of a probability component and an entropy component.

          prob_component    = min(1, max_prob / (10 * uniform_prob))
          entropy_component = max(0, 1 - entropy / log(num_cells))
        """
        if self._found_eve:
            return 1.0

        n = self._config.map.width * self._config.map.height
        if n < 2:
            return 0.0
        uniform     = 1.0 / n
        prob_comp   = min(1.0, max_prob / (10.0 * uniform))
        max_H       = math.log(n)
        ent_comp    = max(0.0, 1.0 - entropy / max_H)
        return round(0.5 * prob_comp + 0.5 * ent_comp, 4)

    def _eve_status(self) -> str:
        """'likely active', 'likely dormant', or 'unknown' from clue strengths."""
        strengths = [
            float(e.get("observation", {}).get("scan_result", {}).get("clue_strength") or 0.0)
            for e in self._trace
            if float(e.get("observation", {}).get("scan_result", {}).get("clue_strength") or 0.0) > 0
        ]
        if not strengths:
            return "unknown"
        avg = sum(strengths) / len(strengths)
        return "likely active" if avg > 0.6 else ("likely dormant" if avg < 0.3 else "unknown")

    def _recommended_action(
        self,
        confidence: float,
        battery_pct: float,
        top_cells: list[dict[str, Any]],
    ) -> str:
        if battery_pct < 0.2:
            return "Recharge first -- battery critically low"
        if confidence > 0.7 and top_cells:
            best = top_cells[0]
            return f"Navigate to cell ({best['x']}, {best['y']}) -- high confidence EVE location"
        return "Continue scanning -- uncertainty still too high to commit to a target"

    def _recommendation_rationale(
        self,
        confidence: float,
        battery_pct: float,
        top_cells: list[dict[str, Any]],
    ) -> str:
        if battery_pct < 0.2:
            return (
                f"Battery is at {battery_pct * 100:.0f}% of maximum -- recharging is the "
                f"safest option before any further exploration.\n\n"
                f"EVE operational status: **{self._eve_status()}**."
            )
        if confidence > 0.7 and top_cells:
            best = top_cells[0]
            return (
                f"The belief grid has converged with {confidence * 100:.0f}% confidence. "
                f"Cell ({best['x']}, {best['y']}) is the most likely position for EVE "
                f"with {best['prob'] * 100:.1f}% probability.\n\n"
                f"EVE operational status: **{self._eve_status()}**."
            )
        return (
            f"Confidence is {confidence * 100:.0f}%, below the 70% threshold. "
            "Additional scanning will reduce uncertainty before committing to a direction.\n\n"
            f"EVE operational status: **{self._eve_status()}**."
        )
