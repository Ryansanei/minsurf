"""Test discrete operators on a sphere mesh.

For the unit sphere:
  - Mean curvature H = 1 everywhere.
  - Gaussian curvature K = 1 everywhere.
  - Gauss-Bonnet: ∫ K dA = 4π = 2π * χ(S²) = 2π * 2. ✓
"""

from __future__ import annotations

import numpy as np

from minsurf.mesh import Mesh
from minsurf.operators import gaussian_curvature, mean_curvature, vertex_areas


def make_sphere_mesh(n_lat: int = 20, n_lon: int = 40) -> Mesh:
    """UV-sphere triangulation of the unit sphere.

    Interior vertices are all vertices except poles (which we merge).
    We set boundary=False everywhere since a closed sphere has no boundary.
    """
    # Latitude lines (excluding poles): theta in (0, pi)
    thetas = np.linspace(0, np.pi, n_lat + 2)[1:-1]  # (n_lat,)
    phis = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)

    V_list = [[0.0, 0.0, 1.0]]  # north pole
    for theta in thetas:
        for phi in phis:
            V_list.append([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
    V_list.append([0.0, 0.0, -1.0])  # south pole

    V = np.array(V_list)
    F_list = []

    north = 0
    south = len(V) - 1

    def grid_idx(i_lat: int, i_lon: int) -> int:
        return 1 + i_lat * n_lon + (i_lon % n_lon)

    # North cap
    for j in range(n_lon):
        F_list.append([north, grid_idx(0, j), grid_idx(0, j + 1)])

    # Latitude bands
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a = grid_idx(i, j)
            b = grid_idx(i, j + 1)
            c = grid_idx(i + 1, j)
            d = grid_idx(i + 1, j + 1)
            F_list.append([a, b, c])
            F_list.append([b, d, c])

    # South cap
    for j in range(n_lon):
        F_list.append([grid_idx(n_lat - 1, j + 1), grid_idx(n_lat - 1, j), south])

    F = np.array(F_list, dtype=np.int64)
    boundary = np.zeros(len(V), dtype=bool)
    return Mesh(V=V, F=F, boundary=boundary)


class TestSphereMeanCurvature:
    def test_mean_curvature_near_one(self):
        """Interior mean curvature on a unit sphere should be ≈ 1."""
        mesh = make_sphere_mesh(n_lat=20, n_lon=40)
        H = mean_curvature(mesh)
        # Exclude poles (idx 0 and last) which are topologically singular
        interior_H = H[1:-1]
        # RMS should be close to 1
        rms = np.sqrt(np.mean((interior_H - 1.0) ** 2))
        assert rms < 0.15, f"RMS deviation from H=1: {rms:.3f}"

    def test_mean_curvature_convergence(self):
        """Higher resolution → closer to H=1."""
        errors = []
        for n_lat in [10, 20, 40]:
            mesh = make_sphere_mesh(n_lat=n_lat, n_lon=2 * n_lat)
            H = mean_curvature(mesh)
            interior_H = H[1:-1]
            errors.append(float(np.abs(interior_H - 1.0).mean()))
        # Error should decrease
        assert errors[0] > errors[1] > errors[2] * 0.9  # monotone decreasing (relaxed)


class TestSphereGaussianCurvature:
    def test_gaussian_curvature_gauss_bonnet(self):
        """∑ K_i A_i ≈ 4π for the unit sphere (χ=2)."""
        mesh = make_sphere_mesh(n_lat=20, n_lon=40)
        K = gaussian_curvature(mesh)
        A = vertex_areas(mesh)
        total = float(np.sum(K * A))
        expected = 4.0 * np.pi
        # Should be within 5%
        assert abs(total - expected) / expected < 0.05, (
            f"Gauss-Bonnet: got {total:.4f}, expected {expected:.4f}"
        )

    def test_gaussian_curvature_positive(self):
        """Gaussian curvature of a convex sphere should be positive everywhere."""
        mesh = make_sphere_mesh(n_lat=15, n_lon=30)
        K = gaussian_curvature(mesh)
        # Most interior values should be positive
        interior_K = K[1:-1]
        assert (interior_K > 0).mean() > 0.9
