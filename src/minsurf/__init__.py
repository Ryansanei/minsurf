"""minsurf — discrete minimal surface toolkit.

Computes, validates, and visualizes discrete minimal surfaces (soap films)
via cotangent-Laplacian mean-curvature flow with exact-solution validation.

Quick start
-----------
>>> from minsurf import boundaries, flow, metrics
>>> seed = boundaries.two_rings(separation=1.0, radius=1.0)
>>> mesh, hist = flow.solve(seed)
>>> print(f"area={mesh.total_area():.4f}  max|H|={metrics.max_mean_curvature(mesh):.2e}")
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Ryan Sanei"
__license__ = "MIT"

from minsurf import boundaries, exact, flow, io, metrics, operators, stability, tpms, visualize
from minsurf.flow import History, solve
from minsurf.mesh import Mesh
from minsurf.metrics import l2_error_to_catenoid, max_mean_curvature, neck_radius, total_area
from minsurf.operators import (
    cotangent_laplacian,
    gaussian_curvature,
    mean_curvature,
    mean_curvature_vector,
    vertex_areas,
)

__all__ = [
    "Mesh",
    "solve",
    "History",
    "cotangent_laplacian",
    "vertex_areas",
    "mean_curvature_vector",
    "mean_curvature",
    "gaussian_curvature",
    "total_area",
    "max_mean_curvature",
    "neck_radius",
    "l2_error_to_catenoid",
    "boundaries",
    "exact",
    "flow",
    "io",
    "metrics",
    "operators",
    "stability",
    "tpms",
    "visualize",
]
