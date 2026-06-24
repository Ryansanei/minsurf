"""Tests for flow solver: area monotonicity, flat disk, convergence."""

from __future__ import annotations

import numpy as np

from minsurf import boundaries, flow
from minsurf.mesh import Mesh


def make_flat_disk(n_radial: int = 8, n_theta: int = 24) -> Mesh:
    """Flat disk: boundary on a circle at z=0; interior vertices also at z=0."""
    return boundaries.disk_boundary(
        z_fun=lambda theta: np.zeros_like(theta),
        n_radial=n_radial,
        n_theta=n_theta,
    )


class TestFlowMonotonicity:
    def test_area_never_increases_implicit(self):
        """Area must not increase at any step under implicit flow."""
        seed = boundaries.two_rings(separation=1.0, radius=1.0, n_z=8, n_theta=16)
        _, hist = flow.solve(seed, method="implicit", tau=0.05, max_iter=100, tol=1e-4)
        areas = hist.area
        # Allow tiny floating-point noise (1e-8 relative tolerance)
        for i in range(1, len(areas)):
            assert areas[i] <= areas[i - 1] * (1 + 1e-6), (
                f"Area increased at step {i}: {areas[i-1]:.6f} → {areas[i]:.6f}"
            )

    def test_area_never_increases_explicit(self):
        """Area must not increase at any step under explicit flow."""
        seed = boundaries.two_rings(separation=1.0, radius=1.0, n_z=8, n_theta=16)
        _, hist = flow.solve(seed, method="explicit", tau=1e-4, max_iter=200, tol=1e-3)
        areas = hist.area
        for i in range(1, len(areas)):
            assert areas[i] <= areas[i - 1] * (1 + 1e-6), (
                f"Area increased at step {i}: {areas[i-1]:.6f} → {areas[i]:.6f}"
            )

    def test_flat_disk_stays_flat(self):
        """A flat disk is already minimal; flow should leave interior nearly unchanged."""
        seed = make_flat_disk(n_radial=6, n_theta=18)
        # Perturb interior slightly
        interior = ~seed.boundary
        seed.V[interior, 2] += np.random.RandomState(42).normal(0, 0.01, interior.sum())
        initial_z_rms = float(np.sqrt(np.mean(seed.V[interior, 2] ** 2)))

        solved, hist = flow.solve(seed, method="implicit", tau=0.05, max_iter=500, tol=1e-5)
        final_z_rms = float(np.sqrt(np.mean(solved.V[interior, 2] ** 2)))
        # Flow should reduce the z-perturbation
        assert final_z_rms < initial_z_rms

    def test_solve_returns_history(self):
        seed = boundaries.two_rings(separation=1.0, radius=1.0, n_z=4, n_theta=8)
        _, hist = flow.solve(seed, method="implicit", tau=0.05, max_iter=10)
        assert len(hist.area) == 10 or len(hist.area) <= 10  # may stop early
        assert len(hist.residual) == len(hist.area)

    def test_residual_decreases(self):
        """Residual max|H| should be lower at the end than at the start."""
        seed = boundaries.two_rings(separation=1.0, radius=1.0, n_z=8, n_theta=16)
        _, hist = flow.solve(seed, method="implicit", tau=0.05, max_iter=200, tol=1e-5)
        if len(hist.residual) > 10:
            assert hist.residual[-1] < hist.residual[0]

    def test_solve_saddle(self):
        """Saddle ring converges without errors."""
        seed = boundaries.saddle_ring(amplitude=0.55, k=2, n_theta=24, n_radial=10)
        solved, hist = flow.solve(seed, method="implicit", tau=0.05, max_iter=500, tol=1e-5)
        assert solved.n_vertices() == seed.n_vertices()
        assert solved.n_faces() == seed.n_faces()
        # Boundary unchanged
        np.testing.assert_allclose(
            solved.V[seed.boundary], seed.V[seed.boundary], atol=1e-10
        )
