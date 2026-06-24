"""Tests for Mesh dataclass and topological utilities."""

from __future__ import annotations

import numpy as np

from minsurf.boundaries import saddle_ring, two_rings
from minsurf.mesh import Mesh


def _make_tetrahedron() -> Mesh:
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ])
    F = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
    boundary = np.zeros(4, dtype=bool)
    return Mesh(V=V, F=F, boundary=boundary)


def _make_disk_triangle() -> Mesh:
    """Minimal disk: 3 boundary + 1 interior vertex."""
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.3, 0.0],  # interior
    ])
    F = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]])
    boundary = np.array([True, True, True, False])
    return Mesh(V=V, F=F, boundary=boundary)


class TestMeshBasics:
    def test_postinit_casts(self):
        V = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        F = [[0, 1, 2], [0, 1, 3]]
        b = [False] * 4
        m = Mesh(V=V, F=F, boundary=b)
        assert m.V.dtype == np.float64
        assert m.F.dtype == np.int64
        assert m.boundary.dtype == bool

    def test_copy_is_independent(self):
        m = _make_tetrahedron()
        c = m.copy()
        c.V[0, 0] = 999.0
        assert m.V[0, 0] != 999.0

    def test_n_vertices_faces(self):
        m = _make_tetrahedron()
        assert m.n_vertices() == 4
        assert m.n_faces() == 4

    def test_face_areas_positive(self):
        m = _make_tetrahedron()
        areas = m.face_areas()
        assert (areas > 0).all()

    def test_total_area(self):
        m = _make_tetrahedron()
        assert m.total_area() > 0


class TestTopology:
    def test_edges_unique_and_ordered(self):
        m = _make_tetrahedron()
        edges = m.edges()
        # Each row: i < j
        assert (edges[:, 0] < edges[:, 1]).all()
        # Tetrahedron has 6 edges
        assert edges.shape == (6, 2)

    def test_euler_characteristic_sphere(self):
        """A closed tetrahedron homeomorphic to S² has χ=2."""
        m = _make_tetrahedron()
        assert m.euler_characteristic() == 2

    def test_euler_characteristic_disk(self):
        """A disk has χ=1."""
        m = _make_disk_triangle()
        assert m.euler_characteristic() == 1

    def test_euler_characteristic_two_rings(self):
        """A tube (annulus) has χ=0."""
        m = two_rings(separation=1.0, radius=1.0, n_z=4, n_theta=8)
        assert m.euler_characteristic() == 0

    def test_neighbors_symmetric(self):
        m = _make_tetrahedron()
        nbrs = m.neighbors()
        assert len(nbrs) == 4
        for i, ni in enumerate(nbrs):
            for j in ni:
                assert i in nbrs[j]

    def test_boundary_detection_two_rings(self):
        m = two_rings(separation=1.0, radius=1.0, n_z=2, n_theta=8)
        # Boundary = top and bottom rings only (n_theta * 2)
        n_b = m.boundary.sum()
        assert n_b == 2 * 8

    def test_boundary_detection_disk(self):
        m = saddle_ring(n_theta=12, n_radial=4)
        # Boundary = outer ring = n_theta vertices
        assert m.boundary[:12].all()
