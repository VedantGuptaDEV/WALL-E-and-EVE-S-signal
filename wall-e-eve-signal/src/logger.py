"""
logger.py
=========
Step-level JSON logger for the WALL-E EVE Signal mission.

Each agent step is recorded with full observation detail, belief summary,
and the agent's decision rationale.  The complete trace is flushed to disk
by calling :meth:`JSONLogger.save`.
"""

from __future__ import annotations

import json
import os
from typing import Any


class JSONLogger:
    """Records agent steps and writes them as a JSON trace file.

    Parameters
    ----------
    output_path:
        Filesystem path for the output ``.json`` file.  The parent directory
        is created if it does not already exist.

    Example
    -------
    >>> logger = JSONLogger("output/trace.json")
    >>> logger.log_step(
    ...     step=1,
    ...     action="scan",
    ...     observation={...},
    ...     belief_summary={"top_cells": [...], "entropy": 2.3},
    ...     decision="scan",
    ...     reasoning="High entropy, scanning to reduce uncertainty.",
    ... )
    >>> logger.save()
    'output/trace.json'
    """

    def __init__(self, output_path: str) -> None:
        """Initialise the logger and ensure the output directory exists."""
        self._output_path = output_path
        self._entries: list[dict[str, Any]] = []

        # Create parent directory if needed
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        action: str,
        observation: dict[str, Any],
        belief_summary: dict[str, Any],
        decision: str,
        reasoning: str,
    ) -> None:
        """Append a single agent step to the in-memory log.

        Parameters
        ----------
        step:
            The step number (1-indexed recommended).
        action:
            The action executed this step (e.g. ``'N'``, ``'scan'``,
            ``'recharge'``).
        observation:
            The full :class:`~environment.Observation` dict returned by the
            environment after the action.
        belief_summary:
            Dictionary with:

            - ``top_cells``: list of ``{x, y, prob}`` dicts for top belief
              cells.
            - ``entropy``: current Shannon entropy of the belief grid.
        decision:
            The chosen action label (mirrors *action*; kept separate for
            traceability).
        reasoning:
            Human-readable explanation of why this action was chosen.
        """
        entry: dict[str, Any] = {
            "step": step,
            "action": action,
            "observation": observation,
            "battery": observation.get("battery"),
            "time_remaining": observation.get("time_remaining"),
            "belief_summary": {
                "top_cells": belief_summary.get("top_cells", []),
                "entropy": belief_summary.get("entropy", 0.0),
            },
            "decision": decision,
            "reasoning": reasoning,
        }
        self._entries.append(entry)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> str:
        """Write all logged entries to the output JSON file.

        Returns
        -------
        str
            The absolute path of the written file.

        Notes
        -----
        The JSON is pretty-printed with 2-space indentation for readability.
        Calling :meth:`save` multiple times overwrites the previous file.
        """
        with open(self._output_path, "w", encoding="utf-8") as fh:
            json.dump(self._entries, fh, indent=2, default=str)
        return os.path.abspath(self._output_path)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[dict[str, Any]]:
        """Read-only view of all logged entries."""
        return list(self._entries)

    @property
    def output_path(self) -> str:
        """Target output path for the JSON trace file."""
        return self._output_path

    def __len__(self) -> int:
        """Return the number of logged steps."""
        return len(self._entries)
