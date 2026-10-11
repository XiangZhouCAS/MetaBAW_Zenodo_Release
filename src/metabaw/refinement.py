"""Explicit, relocation-safe evidence for a validated zero-bin refinement."""
from __future__ import annotations

import json
from pathlib import Path


EMPTY_REFINEMENT_MARKER = ".metabaw.refinement-empty.json"
EMPTY_REASONS = {
    "das_tool": "no_bins_above_score_threshold",
    "metawrap": "no_bins_passing_final_quality_filter",
}


def read_empty_refinement(directory: Path) -> dict[str, object] | None:
    marker = directory / EMPTY_REFINEMENT_MARKER
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        valid = (
            isinstance(payload, dict)
            and type(payload.get("schema")) is int and payload["schema"] == 1
            and type(payload.get("bin_count")) is int and payload["bin_count"] == 0
            and isinstance(payload.get("tool"), str)
            and payload["tool"] in EMPTY_REASONS
            and payload.get("reason") == EMPTY_REASONS[payload["tool"]]
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid empty-refinement evidence: {marker}") from exc
    if not valid:
        raise ValueError(f"Invalid empty-refinement evidence: {marker}")
    return payload


def write_empty_refinement(directory: Path, tool: str) -> None:
    """Call only after validating fresh native no-bin evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "bin_count": 0, "tool": tool,
               "reason": EMPTY_REASONS[tool]}
    (directory / EMPTY_REFINEMENT_MARKER).write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8",
    )
