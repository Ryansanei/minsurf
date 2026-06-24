"""Tests for the JSON report writer."""

from __future__ import annotations

import pytest

from minsurf.boundaries import two_rings
from minsurf.report import load_report, write_report


@pytest.fixture
def simple_mesh():
    return two_rings(separation=1.0, radius=1.0, n_z=4, n_theta=8)


class TestReport:
    def test_report_written(self, simple_mesh, tmp_path):
        path = tmp_path / "report.json"
        write_report(
            path, preset="test", boundary="two-rings", params={"tau": 0.05},
            mesh=simple_mesh, initial_area=10.0, iterations=42,
            residual_max_H=1e-5, converged=True,
        )
        assert path.exists()

    def test_report_schema_keys(self, simple_mesh, tmp_path):
        path = tmp_path / "report.json"
        rpt = write_report(
            path, preset="test", boundary="two-rings", params={"tau": 0.05},
            mesh=simple_mesh, initial_area=10.0, iterations=42,
            residual_max_H=1e-5, converged=True,
        )
        assert "input" in rpt
        assert "mesh" in rpt
        assert "result" in rpt
        assert "validation" in rpt
        assert "stability" in rpt
        assert "outputs" in rpt

    def test_report_mesh_fields(self, simple_mesh, tmp_path):
        path = tmp_path / "report.json"
        rpt = write_report(
            path, mesh=simple_mesh, initial_area=10.0, iterations=1,
            residual_max_H=0.0, converged=True,
        )
        assert rpt["mesh"]["n_vertices"] == simple_mesh.n_vertices()
        assert rpt["mesh"]["n_faces"] == simple_mesh.n_faces()
        assert "euler_characteristic" in rpt["mesh"]

    def test_report_result_fields(self, simple_mesh, tmp_path):
        path = tmp_path / "report.json"
        rpt = write_report(
            path, mesh=simple_mesh, initial_area=10.0, iterations=100,
            residual_max_H=1e-7, converged=True,
        )
        r = rpt["result"]
        assert "final_area" in r
        assert "initial_area" in r
        assert "area_reduction_pct" in r
        assert r["initial_area"] == 10.0
        assert r["iterations"] == 100
        assert r["converged"] is True

    def test_load_report(self, simple_mesh, tmp_path):
        path = tmp_path / "report.json"
        write_report(
            path, mesh=simple_mesh, initial_area=5.0, iterations=10,
            residual_max_H=0.001, converged=False,
        )
        loaded = load_report(path)
        assert loaded["result"]["iterations"] == 10
        assert loaded["result"]["converged"] is False
