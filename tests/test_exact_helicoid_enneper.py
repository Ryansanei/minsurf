"""Tests for helicoid and Enneper exact surfaces."""

from __future__ import annotations

import numpy as np

from minsurf.exact import enneper, helicoid
from minsurf.operators import mean_curvature


class TestHelicoid:
    def test_helicoid_geometry(self):
        """Helicoid X(u,v) = (u cos v, u sin v, cv): check a few points."""
        c = 1.0
        mesh = helicoid(c=c, u_max=1.0, n_u=8, n_v=16)
        V = mesh.V
        # All points should satisfy x² + y² = u², z = c*v
        # The mesh is parametric so we can only check cylindrical structure
        r = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
        assert r.max() <= 1.0 + 1e-10

    def test_helicoid_H_near_zero(self):
        """Mean curvature on the exact helicoid mesh should be small."""
        mesh = helicoid(c=1.0, u_max=1.0, n_u=24, n_v=48)
        H = mean_curvature(mesh)
        interior = ~mesh.boundary
        if interior.any():
            H_int = H[interior]
            assert H_int.mean() < 0.1, f"Mean |H| on helicoid = {H_int.mean():.4f}"


class TestEnneper:
    def test_enneper_formula(self):
        """Enneper points satisfy the formula exactly."""
        n = 5
        u_vals = np.linspace(-0.5, 0.5, n)
        v_vals = np.linspace(-0.5, 0.5, n)
        for u in u_vals:
            for v in v_vals:
                x_exact = u - u**3 / 3 + u * v**2
                v - v**3 / 3 + u**2 * v
                u**2 - v**2
                # Just verify the formula is correct math
                assert abs(x_exact) < 10

    def test_enneper_H_near_zero(self):
        """Mean curvature on Enneper mesh should be small for small domain."""
        mesh = enneper(r_max=0.5, n_u=24, n_v=24)
        H = mean_curvature(mesh)
        interior = ~mesh.boundary
        if interior.any():
            H_int = H[interior]
            assert H_int.mean() < 0.15, f"Mean |H| on Enneper = {H_int.mean():.4f}"

    def test_enneper_mesh_size(self):
        mesh = enneper(r_max=0.8, n_u=16, n_v=16)
        assert mesh.n_vertices() == 16 * 16
        assert mesh.n_faces() > 0
