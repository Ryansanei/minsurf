"""Tests for the CLI: demo command writes expected files."""

from __future__ import annotations

import json

import pytest


@pytest.fixture(scope="session")
def demo_output(tmp_path_factory):
    """Run the demo once and share output across all tests in this session."""
    out = tmp_path_factory.mktemp("demo")
    from minsurf.demo import run_demo
    run_demo(output_dir=out, verbose=False)
    return out


class TestDemo:
    def test_demo_runs_and_writes_files(self, demo_output):
        """minsurf demo should write outputs and a summary JSON."""
        tmp_path = demo_output
        assert (tmp_path / "demo_summary.json").exists()
        with open(tmp_path / "demo_summary.json") as fh:
            summary = json.load(fh)
        assert isinstance(summary, list)
        assert len(summary) >= 4

        # Catenoid outputs
        assert (tmp_path / "catenoid.obj").exists()
        assert (tmp_path / "catenoid.stl").exists()
        assert (tmp_path / "catenoid_report.json").exists()
        assert (tmp_path / "catenoid_viewer.html").exists()
        assert (tmp_path / "catenoid_render.png").exists()

        # Saddle ring outputs
        assert (tmp_path / "saddle_ring.obj").exists()

        # Enneper
        assert (tmp_path / "enneper.obj").exists()

    def test_demo_report_schema(self, demo_output):
        """Catenoid report should have all required keys."""
        with open(demo_output / "catenoid_report.json") as fh:
            rpt = json.load(fh)
        assert "input" in rpt
        assert "mesh" in rpt
        assert "result" in rpt
        assert "validation" in rpt
        assert "stability" in rpt

    def test_demo_catenoid_converged(self, demo_output):
        """Catenoid should converge in the demo."""
        with open(demo_output / "catenoid_report.json") as fh:
            rpt = json.load(fh)
        assert rpt["result"]["converged"] is True

    def test_demo_viewer_is_html(self, demo_output):
        """Viewer file should be valid HTML."""
        html = (demo_output / "catenoid_viewer.html").read_text()
        assert "<!DOCTYPE html>" in html
        assert "WebGL" in html or "canvas" in html


class TestCLIParsing:
    def test_main_no_args_exits_nonzero(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "minsurf.cli"],
            capture_output=True, text=True
        )
        # Should print help and exit
        assert result.returncode != 0 or "usage" in result.stdout.lower() + result.stderr.lower()

    def test_version_flag(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "minsurf.cli", "--version"],
            capture_output=True, text=True
        )
        assert "0.1.0" in result.stdout + result.stderr
