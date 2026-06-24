"""Surface metrics: area, curvature residuals, L2 error vs exact solutions, etc."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from minsurf.mesh import Mesh
from minsurf.operators import mean_curvature

_EPS = 1e-14


def total_area(mesh: Mesh) -> float:
    """Total triangle area of the mesh."""
    return mesh.total_area()


def max_mean_curvature(mesh: Mesh) -> float:
    """Maximum |H| over all interior vertices."""
    H = mean_curvature(mesh)
    interior = ~mesh.boundary
    if not interior.any():
        return 0.0
    return float(H[interior].max())


def neck_radius(mesh: Mesh) -> float:
    """Minimum radial distance from the z-axis among all interior vertices.

    For a catenoid (tube topology), this is an estimate of the neck radius a.
    """
    interior = ~mesh.boundary
    if not interior.any():
        return float("nan")
    V = mesh.V[interior]
    r = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
    return float(r.min())


def l2_error_to_catenoid(mesh: Mesh, a: float) -> float:
    """Radial L2 error of a tube-topology mesh vs the exact catenoid r = a cosh(z/a).

    Compares the computed radial distance r_i = sqrt(x_i²+y_i²) to the exact
    value a cosh(z_i/a) at each interior vertex.

    Parameters
    ----------
    mesh : converged mesh (tube topology).
    a : catenoid neck parameter.

    Returns
    -------
    L2 error normalised by the number of interior vertices (RMS error).
    """
    interior = ~mesh.boundary
    if not interior.any():
        return float("nan")
    V = mesh.V[interior]
    r_computed = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
    z = V[:, 2]
    r_exact = a * np.cosh(z / a)
    diff = r_computed - r_exact
    return float(np.sqrt(np.mean(diff**2)))


def linf_error_to_catenoid(mesh: Mesh, a: float) -> float:
    """L∞ radial error vs exact catenoid."""
    interior = ~mesh.boundary
    if not interior.any():
        return float("nan")
    V = mesh.V[interior]
    r_computed = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
    z = V[:, 2]
    r_exact = a * np.cosh(z / a)
    return float(np.max(np.abs(r_computed - r_exact)))


def convergence_study(
    boundary_fn: Callable[..., Mesh],
    exact_fn: Callable[[Mesh], tuple[float, Mesh]],
    resolutions: list[dict],
    flow_kwargs: dict | None = None,
) -> list[dict]:
    """Run the flow at multiple resolutions and report errors.

    Parameters
    ----------
    boundary_fn : callable(**res_params) → seed Mesh.
    exact_fn : callable(solved_mesh) → (error, exact_mesh).
    resolutions : list of dicts; each dict is passed as **kwargs to boundary_fn.
    flow_kwargs : extra kwargs for flow.solve.

    Returns
    -------
    List of dicts, one per resolution, with keys: resolution_params, area,
    error, ratio (error_{i-1}/error_i), empirical_order.
    """
    from minsurf.flow import solve

    if flow_kwargs is None:
        flow_kwargs = {}

    results: list[dict] = []
    prev_error: float | None = None

    for res in resolutions:
        seed = boundary_fn(**res)
        solved, hist = solve(seed, **flow_kwargs)
        error, _ = exact_fn(solved)
        ratio = (prev_error / error) if (prev_error is not None and error > _EPS) else None
        order = (np.log2(ratio) if ratio is not None else None)
        results.append(
            {
                "resolution_params": res,
                "n_vertices": solved.n_vertices(),
                "n_faces": solved.n_faces(),
                "area": solved.total_area(),
                "error": error,
                "ratio": ratio,
                "empirical_order": order,
                "iterations": len(hist.area),
                "final_residual": hist.residual[-1] if hist.residual else float("nan"),
            }
        )
        prev_error = error

    return results


def gauss_bonnet_residual(mesh: Mesh) -> float:
    """Check Gauss–Bonnet: |Σ K_i A_i − 2π χ| / |2π χ|.

    Returns the relative error.  Should be < 1% for reasonable meshes.
    """
    from minsurf.operators import gaussian_curvature, vertex_areas

    K = gaussian_curvature(mesh)
    A = vertex_areas(mesh)
    chi = mesh.euler_characteristic()
    integrated = float(np.sum(K * A))
    expected = 2.0 * np.pi * chi
    if abs(expected) < _EPS:
        return abs(integrated)
    return abs(integrated - expected) / abs(expected)
