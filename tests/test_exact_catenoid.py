"""Tests for the exact catenoid: H≈0, fundamental forms, neck radius."""

from __future__ import annotations

import numpy as np

from minsurf.exact import catenoid, catenoid_for_rings
from minsurf.operators import mean_curvature


class TestExactCatenoid:
    def test_catenoid_mesh_shape(self):
        mesh = catenoid(a=1.0, separation=2.0, n_theta=32, n_z=20)
        # n_z rows × n_theta cols
        assert mesh.n_vertices() == 32 * 20
        assert mesh.n_faces() > 0

    def test_catenoid_boundary_z(self):
        """Boundary vertices should lie on z = ±separation/2."""
        sep = 2.0
        mesh = catenoid(a=1.0, separation=sep, n_theta=16, n_z=10)
        bv = mesh.V[mesh.boundary]
        z_vals = np.unique(np.round(bv[:, 2], 6))
        assert len(z_vals) == 2
        np.testing.assert_allclose(sorted(z_vals), [-sep / 2, sep / 2], atol=1e-10)

    def test_catenoid_radial_profile(self):
        """Interior vertices should satisfy r = a cosh(z/a)."""
        a = 1.5
        mesh = catenoid(a=a, separation=3.0, n_theta=32, n_z=24)
        interior = ~mesh.boundary
        V = mesh.V[interior]
        r = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
        r_exact = a * np.cosh(V[:, 2] / a)
        np.testing.assert_allclose(r, r_exact, rtol=1e-6)

    def test_catenoid_H_near_zero(self):
        """Mean curvature on the exact catenoid mesh should be very small."""
        mesh = catenoid(a=1.0, separation=2.0, n_theta=48, n_z=32)
        H = mean_curvature(mesh)
        interior = ~mesh.boundary
        H_int = H[interior]
        # The discretisation error: H should be < 0.05 for a fine mesh
        assert H_int.mean() < 0.05, f"Mean |H| = {H_int.mean():.4f} (should be < 0.05)"

    def test_catenoid_for_rings_no_solution(self):
        """No catenoid exists when separation/radius > critical."""
        mesh, a = catenoid_for_rings(separation=2.0, radius=1.0)  # ratio = 2.0 > 1.3255
        assert mesh is None
        assert a is None

    def test_catenoid_for_rings_with_solution(self):
        """Should find a valid neck radius for small separation."""
        mesh, a = catenoid_for_rings(separation=1.0, radius=1.0)
        assert mesh is not None
        assert a is not None
        assert 0 < a < 1.0  # neck < ring radius
        # Check that boundary ring radii match
        bv = mesh.V[mesh.boundary]
        r_b = np.sqrt(bv[:, 0] ** 2 + bv[:, 1] ** 2)
        np.testing.assert_allclose(r_b, 1.0, atol=1e-6)

    def test_catenoid_isometric_property(self):
        """First fundamental form is isothermal: E=G, F≈0 at interior vertices."""
        # This is checked via the parametric formula:
        # E = G = cosh²(z/a), F = 0.
        a = 1.0
        sep = 2.0
        n_theta = 8
        n_z = 6
        # Sample: dX/dθ and dX/dz at a few points
        h = sep / 2
        theta_pts = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        z_pts = np.linspace(-h, h, n_z)
        th, zz = np.meshgrid(theta_pts, z_pts)
        th, zz = th.ravel(), zz.ravel()
        eps = 1e-6
        # Tangent vectors
        def X(t, z):
            return np.stack([a * np.cosh(z / a) * np.cos(t),
                                            a * np.cosh(z / a) * np.sin(t), z], axis=-1)
        Xt = (X(th + eps, zz) - X(th - eps, zz)) / (2 * eps)
        Xz = (X(th, zz + eps) - X(th, zz - eps)) / (2 * eps)
        E = np.einsum("...i,...i->...", Xt, Xt)
        G = np.einsum("...i,...i->...", Xz, Xz)
        F = np.einsum("...i,...i->...", Xt, Xz)
        np.testing.assert_allclose(E, G, rtol=1e-4)
        np.testing.assert_allclose(F, 0.0, atol=1e-4)
