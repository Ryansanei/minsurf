"""Tests for catenoid convergence under mesh refinement."""

from __future__ import annotations

from minsurf import boundaries, exact, flow, metrics


class TestCatenoidConvergence:
    def test_l2_error_decreases_with_refinement(self):
        """L2 radial error vs exact catenoid should decrease as mesh is refined."""
        sep, rad = 1.0, 1.0
        a_exact = exact._catenoid_neck(sep, rad)
        assert a_exact is not None, "No catenoid solution for these params"

        errors = []
        for n_theta, n_z in [(16, 12), (32, 20), (48, 30)]:
            seed = boundaries.two_rings(separation=sep, radius=rad, n_z=n_z, n_theta=n_theta)
            solved, _ = flow.solve(
                seed, method="implicit", tau=0.05, max_iter=500, tol=1e-7
            )
            err = metrics.l2_error_to_catenoid(solved, a_exact)
            errors.append(err)

        # Finest mesh should be significantly better than coarsest (overall convergence)
        assert errors[2] < errors[0] * 0.5, (
            f"No meaningful convergence: errors={[f'{e:.4e}' for e in errors]}"
        )
        # At least one step must show improvement
        assert errors[1] < errors[0] or errors[2] < errors[1], (
            f"Error never improved: {errors}"
        )

    def test_neck_radius_close_to_exact(self):
        """Computed neck radius should be within 10% of exact a."""
        sep, rad = 1.0, 1.0
        a_exact = exact._catenoid_neck(sep, rad)
        assert a_exact is not None

        seed = boundaries.two_rings(separation=sep, radius=rad, n_z=24, n_theta=48)
        solved, _ = flow.solve(seed, method="implicit", tau=0.05, max_iter=1000, tol=1e-6)
        neck = metrics.neck_radius(solved)
        rel_err = abs(neck - a_exact) / a_exact
        assert rel_err < 0.10, f"Neck radius: exact={a_exact:.4f}, measured={neck:.4f}, rel={rel_err:.3f}"

    def test_convergence_study_output_schema(self):
        """convergence_study returns the expected keys."""
        sep, rad = 1.0, 1.0
        a_exact = exact._catenoid_neck(sep, rad)

        def bfn(**kw):
            return boundaries.two_rings(**kw)

        def efn(mesh):
            return metrics.l2_error_to_catenoid(mesh, a_exact), None

        resolutions = [
            {"separation": sep, "radius": rad, "n_theta": 16, "n_z": 10},
            {"separation": sep, "radius": rad, "n_theta": 24, "n_z": 16},
        ]
        results = metrics.convergence_study(
            bfn, efn, resolutions,
            flow_kwargs={"method": "implicit", "tau": 0.05, "max_iter": 200, "tol": 1e-5},
        )
        assert len(results) == 2
        for r in results:
            assert "error" in r
            assert "n_vertices" in r
            assert "empirical_order" in r
