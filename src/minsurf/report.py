"""JSON report writer for minsurf solve results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minsurf.mesh import Mesh


def write_report(
    path: str | Path,
    *,
    preset: str = "",
    boundary: str = "",
    params: dict | None = None,
    mesh: Mesh,
    initial_area: float,
    iterations: int,
    residual_max_H: float,
    converged: bool,
    validation: dict | None = None,
    stability: dict | None = None,
    outputs: dict | None = None,
) -> dict:
    """Write a JSON report and return the report dict.

    Parameters
    ----------
    path : output file path.
    preset : preset name used.
    boundary : boundary type description.
    params : solver parameters.
    mesh : final (converged) mesh.
    initial_area : area of seed mesh before flow.
    iterations : number of flow iterations run.
    residual_max_H : final max|H| residual.
    converged : whether the tol criterion was met.
    validation : optional validation sub-dict.
    stability : optional stability sub-dict.
    outputs : optional output file paths dict.
    """
    final_area = mesh.total_area()
    area_reduction_pct = 100.0 * (initial_area - final_area) / max(initial_area, 1e-30)

    report: dict[str, Any] = {
        "input": {
            "preset": preset,
            "boundary": boundary,
            "params": params or {},
        },
        "mesh": {
            "n_vertices": mesh.n_vertices(),
            "n_faces": mesh.n_faces(),
            "euler_characteristic": mesh.euler_characteristic(),
        },
        "result": {
            "final_area": final_area,
            "initial_area": initial_area,
            "area_reduction_pct": area_reduction_pct,
            "iterations": iterations,
            "residual_max_H": residual_max_H,
            "converged": converged,
        },
        "validation": validation or {},
        "stability": stability or {},
        "outputs": outputs or {},
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)

    return report


def load_report(path: str | Path) -> dict:
    """Load a JSON report produced by write_report."""
    with open(path) as fh:
        return json.load(fh)
