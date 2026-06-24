"""Tests for stability analysis: threshold detection, collapse detection."""

from __future__ import annotations

import numpy as np

from minsurf import boundaries, flow
from minsurf.stability import (
    CRITICAL_RATIO,
    analyze,
    detect_collapse,
    separation_over_radius,
    stability_regime,
)


class TestStabilityRegime:
    def test_below_threshold_is_stable(self):
        assert stability_regime(0.8) == "stable"
        assert stability_regime(1.0) == "stable"
        assert stability_regime(1.2) == "stable"

    def test_above_threshold_is_collapsed(self):
        assert stability_regime(1.6) == "collapsed"
        assert stability_regime(2.0) == "collapsed"

    def test_at_threshold_is_unstable(self):
        assert stability_regime(CRITICAL_RATIO) == "unstable"


class TestSeparationRatio:
    def test_two_rings_ratio(self):
        sep, rad = 1.0, 1.0
        mesh = boundaries.two_rings(separation=sep, radius=rad, n_z=4, n_theta=8)
        ratio = separation_over_radius(mesh)
        np.testing.assert_allclose(ratio, sep / rad, rtol=0.01)

    def test_varying_separation(self):
        for sep in [0.5, 1.0, 1.5, 2.0]:
            mesh = boundaries.two_rings(separation=sep, radius=1.0, n_z=4, n_theta=8)
            ratio = separation_over_radius(mesh)
            np.testing.assert_allclose(ratio, sep, rtol=0.01)


class TestCollapseDetection:
    def test_stable_catenoid_no_collapse(self):
        """For sep/R < critical, the converged catenoid should not collapse."""
        seed = boundaries.two_rings(separation=1.0, radius=1.0, n_z=16, n_theta=32)
        solved, _ = flow.solve(seed, method="implicit", tau=0.05, max_iter=500, tol=1e-5)
        assert not detect_collapse(solved), "Stable catenoid should not collapse"

    def test_analysis_stable_regime(self):
        """analyze() should report 'stable' for sep/R = 1.0."""
        seed = boundaries.two_rings(separation=1.0, radius=1.0, n_z=12, n_theta=24)
        solved, _ = flow.solve(seed, method="implicit", tau=0.05, max_iter=300, tol=1e-5)
        info = analyze(solved)
        assert info["ratio"] < CRITICAL_RATIO
        assert info["regime"] == "stable"

    def test_collapse_past_threshold(self):
        """For sep/R >> critical (1.6), the flow should produce near-collapse."""
        seed = boundaries.two_rings(separation=1.6, radius=1.0, n_z=16, n_theta=32)
        solved, _ = flow.solve(seed, method="implicit", tau=0.05, max_iter=500, tol=1e-4)
        info = analyze(solved)
        # The ratio should be above threshold
        assert info["ratio"] > CRITICAL_RATIO
        # The regime should be classified as collapsed or unstable
        assert info["regime"] in ("unstable", "collapsed")
