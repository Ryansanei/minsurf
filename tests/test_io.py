"""Tests for mesh I/O: OBJ/STL round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from minsurf.boundaries import two_rings
from minsurf.io import export, read_obj, read_stl, write_obj, write_ply, write_stl


@pytest.fixture
def simple_mesh():
    return two_rings(separation=1.0, radius=1.0, n_z=4, n_theta=8)


class TestOBJRoundTrip:
    def test_write_read_vertices(self, simple_mesh, tmp_path):
        path = tmp_path / "test.obj"
        write_obj(simple_mesh, path)
        loaded = read_obj(path)
        np.testing.assert_allclose(simple_mesh.V, loaded.V, atol=1e-6)

    def test_write_read_faces(self, simple_mesh, tmp_path):
        path = tmp_path / "test.obj"
        write_obj(simple_mesh, path)
        loaded = read_obj(path)
        assert loaded.n_faces() == simple_mesh.n_faces()

    def test_obj_file_exists(self, simple_mesh, tmp_path):
        path = tmp_path / "mesh.obj"
        write_obj(simple_mesh, path)
        assert path.exists()
        assert path.stat().st_size > 0


class TestSTLRoundTrip:
    def test_binary_stl_write(self, simple_mesh, tmp_path):
        path = tmp_path / "test.stl"
        write_stl(simple_mesh, path, binary=True)
        assert path.exists()
        # Binary STL size = 80 + 4 + n_faces * 50
        expected_size = 80 + 4 + simple_mesh.n_faces() * 50
        assert path.stat().st_size == expected_size

    def test_ascii_stl_write(self, simple_mesh, tmp_path):
        path = tmp_path / "test_ascii.stl"
        write_stl(simple_mesh, path, binary=False)
        assert path.exists()
        content = path.read_text()
        assert "solid minsurf" in content
        assert "endsolid" in content

    def test_binary_stl_read_vertices(self, simple_mesh, tmp_path):
        path = tmp_path / "test.stl"
        write_stl(simple_mesh, path, binary=True)
        loaded = read_stl(path)
        # STL doesn't merge vertices; loaded.n_faces() should match
        assert loaded.n_faces() == simple_mesh.n_faces()


class TestPLY:
    def test_ply_write(self, simple_mesh, tmp_path):
        path = tmp_path / "test.ply"
        write_ply(simple_mesh, path)
        assert path.exists()
        content = path.read_text()
        assert "element vertex" in content
        assert "element face" in content

    def test_export_dispatch(self, simple_mesh, tmp_path):
        for fmt in ("obj", "stl", "ply"):
            path = tmp_path / f"mesh.{fmt}"
            export(simple_mesh, path, fmt=fmt)
            assert path.exists()

    def test_export_infer_format(self, simple_mesh, tmp_path):
        path = tmp_path / "mesh.obj"
        export(simple_mesh, path)
        assert path.exists()
