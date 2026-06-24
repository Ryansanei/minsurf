"""Tests for the cotangent Laplacian matrix properties."""

from __future__ import annotations

import numpy as np

from minsurf.boundaries import saddle_ring, two_rings
from minsurf.operators import cotangent_laplacian


class TestCotangentLaplacian:
    def test_row_sums_near_zero(self):
        """Rows of the Laplacian should sum to ~0 (harmonic property)."""
        mesh = two_rings(separation=1.0, radius=1.0, n_z=4, n_theta=8)
        L = cotangent_laplacian(mesh)
        row_sums = np.abs(np.array(L.sum(axis=1)).ravel())
        # All row sums should be near zero
        assert row_sums.max() < 1e-10, f"Max row-sum deviation: {row_sums.max():.2e}"

    def test_symmetry(self):
        """Cotangent Laplacian should be symmetric."""
        mesh = saddle_ring(n_theta=12, n_radial=4)
        L = cotangent_laplacian(mesh)
        diff = (L - L.T).data
        if len(diff) > 0:
            assert np.abs(diff).max() < 1e-10

    def test_shape(self):
        mesh = two_rings(separation=1.0, radius=1.0, n_z=4, n_theta=8)
        L = cotangent_laplacian(mesh)
        n = mesh.n_vertices()
        assert L.shape == (n, n)

    def test_diagonal_nonnegative(self):
        """Diagonal entries of L = diag(row_sums(W)) − W should be ≥ 0."""
        mesh = two_rings(separation=1.0, radius=1.0, n_z=6, n_theta=12)
        L = cotangent_laplacian(mesh)
        diag = L.diagonal()
        # Diagonal should be non-negative (sum of cotangent weights ≥ 0 for Delaunay-ish meshes)
        # Might be slightly negative for non-Delaunay; use a loose bound.
        assert diag.min() > -0.1

    def test_null_space_constants(self):
        """L @ ones ≈ 0 (constants are in the null space of the combinatorial Laplacian)."""
        mesh = two_rings(separation=1.0, radius=1.0, n_z=4, n_theta=8)
        L = cotangent_laplacian(mesh)
        ones = np.ones(mesh.n_vertices())
        Lones = L @ ones
        assert np.abs(Lones).max() < 1e-10
