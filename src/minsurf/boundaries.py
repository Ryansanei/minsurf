"""Boundary-curve generators that return seed Mesh objects.

Each function builds a triangulated interior seeded from the boundary
curve and marks boundary vertices as pinned (boundary=True).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from minsurf.mesh import Mesh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fan_triangulate(ring_idx: np.ndarray, interior_idx: np.ndarray, F: list[list[int]]) -> None:
    """Connect interior grid to a boundary ring using a structured quad-fan."""
    # Not used directly; topology handled per-function below.
    pass


def _make_disk_mesh(
    boundary_xyz: np.ndarray,
    n_radial: int,
    center: np.ndarray | None = None,
) -> Mesh:
    """Build a disk-topology triangulation from a closed boundary curve.

    Parameters
    ----------
    boundary_xyz : (nb, 3) array of ordered boundary vertices.
    n_radial : number of radial layers (rings between boundary and center).
    center : fixed center position; defaults to centroid.
    """
    nb = boundary_xyz.shape[0]
    if center is None:
        center = boundary_xyz.mean(axis=0)

    V_list: list[np.ndarray] = [boundary_xyz[k] for k in range(nb)]
    boundary_flag = [True] * nb

    # Build concentric rings: t=1 is boundary, t=0 is center
    ring_idx: list[list[int]] = []  # ring_idx[0] = boundary ring indices
    ring_idx.append(list(range(nb)))

    for r in range(1, n_radial):
        t = 1.0 - r / n_radial  # t=0 → center
        ring_pts = (1 - t) * center[np.newaxis, :] + t * boundary_xyz
        start = len(V_list)
        for pt in ring_pts:
            V_list.append(pt)
            boundary_flag.append(False)
        ring_idx.append(list(range(start, start + nb)))

    # Center vertex
    center_idx = len(V_list)
    V_list.append(center.copy())
    boundary_flag.append(False)

    V = np.stack(V_list)
    F_list: list[list[int]] = []

    # Between consecutive rings
    for r in range(len(ring_idx) - 1):
        outer = ring_idx[r]
        inner = ring_idx[r + 1]
        for k in range(nb):
            k1 = (k + 1) % nb
            # Quad split into 2 triangles
            F_list.append([outer[k], outer[k1], inner[k]])
            F_list.append([outer[k1], inner[k1], inner[k]])

    # Fan from innermost ring to center
    inner = ring_idx[-1]
    for k in range(nb):
        k1 = (k + 1) % nb
        F_list.append([inner[k], inner[k1], center_idx])

    F = np.array(F_list, dtype=np.int64)
    boundary = np.array(boundary_flag, dtype=bool)
    return Mesh(V=V, F=F, boundary=boundary)


def _make_tube_mesh(
    ring_a: np.ndarray,
    ring_b: np.ndarray,
    n_z: int,
) -> Mesh:
    """Build a tube (cylinder-topology) mesh between two coplanar rings.

    Parameters
    ----------
    ring_a, ring_b : (n_theta, 3) arrays of ordered boundary vertices.
    n_z : number of intermediate z-layers (interior only).
    """
    n_theta = ring_a.shape[0]
    assert ring_b.shape[0] == n_theta

    V_list: list[np.ndarray] = list(ring_a)
    boundary_flag = [True] * n_theta

    interior_rings: list[list[int]] = []
    for iz in range(1, n_z + 1):
        t = iz / (n_z + 1)
        pts = (1.0 - t) * ring_a + t * ring_b
        start = len(V_list)
        for pt in pts:
            V_list.append(pt)
            boundary_flag.append(False)
        interior_rings.append(list(range(start, start + n_theta)))

    start_b = len(V_list)
    for pt in ring_b:
        V_list.append(pt)
        boundary_flag.append(True)
    bottom_ring = list(range(start_b, start_b + n_theta))

    # All rings in order: ring_a, interior_rings..., ring_b
    all_rings: list[list[int]] = [list(range(n_theta))] + interior_rings + [bottom_ring]

    V = np.stack(V_list)
    F_list: list[list[int]] = []
    for r in range(len(all_rings) - 1):
        outer = all_rings[r]
        inner = all_rings[r + 1]
        for k in range(n_theta):
            k1 = (k + 1) % n_theta
            F_list.append([outer[k], outer[k1], inner[k]])
            F_list.append([outer[k1], inner[k1], inner[k]])

    F = np.array(F_list, dtype=np.int64)
    boundary = np.array(boundary_flag, dtype=bool)
    return Mesh(V=V, F=F, boundary=boundary)


# ---------------------------------------------------------------------------
# Public boundary generators
# ---------------------------------------------------------------------------


def disk_boundary(
    z_fun: Callable[[np.ndarray], np.ndarray],
    n_radial: int = 16,
    n_theta: int = 48,
    amplitude: float = 1.0,
    radius: float = 1.0,
) -> Mesh:
    """Closed-loop disk boundary with a prescribed height function.

    Parameters
    ----------
    z_fun : callable, takes (n_theta,) theta array, returns z values.
    n_radial : radial refinement layers.
    n_theta : points on the boundary circle.
    amplitude : ignored (z_fun controls z); radius sets the boundary circle.
    radius : radius of the boundary circle.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = z_fun(theta)
    boundary_xyz = np.column_stack([x, y, z])
    return _make_disk_mesh(boundary_xyz, n_radial=n_radial)


def saddle_ring(
    amplitude: float = 0.55,
    k: int = 2,
    n_radial: int = 16,
    n_theta: int = 48,
    radius: float = 1.0,
) -> Mesh:
    """Saddle-shaped ring: z = amplitude * sin(k * theta).

    k=2 gives a 2-fold saddle (classic Plateau saddle surface).
    k=3 gives a monkey-saddle (wavy-ring / Enneper-like).
    """
    return disk_boundary(
        z_fun=lambda theta: amplitude * np.sin(k * theta),
        n_radial=n_radial,
        n_theta=n_theta,
        radius=radius,
    )


def two_rings(
    separation: float = 1.0,
    radius: float = 1.0,
    n_z: int = 24,
    n_theta: int = 48,
) -> Mesh:
    """Two coaxial circles (tube topology) — seed for catenoid relaxation.

    Parameters
    ----------
    separation : full z-span between the two rings (total height).
    radius : circle radius.
    n_z : number of interior z-layers.
    n_theta : azimuthal resolution.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    h = separation / 2.0
    ring_a = np.column_stack([radius * cos_t, radius * sin_t, np.full(n_theta, -h)])
    ring_b = np.column_stack([radius * cos_t, radius * sin_t, np.full(n_theta, +h)])
    return _make_tube_mesh(ring_a, ring_b, n_z=n_z)


def polygon_boundary(
    points: np.ndarray,
    n_radial: int = 16,
) -> Mesh:
    """Arbitrary planar or 3-D polygon boundary.

    Parameters
    ----------
    points : (nb, 3) ordered boundary polygon vertices.
    n_radial : radial refinement layers.
    """
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 3
    return _make_disk_mesh(points, n_radial=n_radial)


def from_points(
    boundary_xyz: np.ndarray,
    n_radial: int = 16,
) -> Mesh:
    """Build a seed mesh from an explicit boundary point cloud.

    Parameters
    ----------
    boundary_xyz : (nb, 3) ordered boundary vertices.
    """
    return polygon_boundary(boundary_xyz, n_radial=n_radial)
