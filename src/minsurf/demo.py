"""Demo runner: builds and solves a curated set of minimal surfaces.

Run via: minsurf demo [--output <dir>]
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def run_demo(output_dir: str | Path = "examples/output/demo", verbose: bool = True) -> None:
    """Run the curated demo set and write outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from minsurf import boundaries, exact, flow, io, metrics, report, stability, visualize

    results_summary: list[dict] = []

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # -----------------------------------------------------------------------
    # 1. Catenoid (with exact validation)
    # -----------------------------------------------------------------------
    log("\n[1/5] Catenoid — two-ring boundary, implicit flow, exact validation")
    sep, rad = 1.0, 1.0
    seed = boundaries.two_rings(separation=sep, radius=rad, n_z=24, n_theta=48)
    initial_area = seed.total_area()
    t0 = time.time()
    solved, hist = flow.solve(seed, method="implicit", tau=0.05, max_iter=2000, tol=1e-6, verbose=False)
    elapsed = time.time() - t0
    log(f"   Solved in {len(hist.area)} iters ({elapsed:.1f}s). Area: {solved.total_area():.4f}")

    # Exact validation
    exact_mesh, a_exact = exact.catenoid_for_rings(sep, rad, n_theta=48, n_z=32)
    l2_err = metrics.l2_error_to_catenoid(solved, a_exact) if a_exact else float("nan")
    neck = metrics.neck_radius(solved)
    stab = stability.analyze(solved)
    log(f"   Exact neck a={a_exact:.4f}, measured neck={neck:.4f}, L2 error={l2_err:.4e}")

    # Write outputs
    io.write_obj(solved, output_dir / "catenoid.obj")
    io.write_stl(solved, output_dir / "catenoid.stl")
    visualize.render_surface(solved, title="Catenoid (computed)", path=output_dir / "catenoid_render.png")
    visualize.render_catenoid_profile(solved, a_exact, path=output_dir / "catenoid_profile.png")
    visualize.render_history(hist, path=output_dir / "catenoid_history.png")

    validation_dict = {
        "exact": "catenoid",
        "neck_radius_measured": neck,
        "neck_radius_exact": a_exact,
        "l2_error": l2_err,
        "rel_error_pct": 100 * abs(neck - a_exact) / (a_exact + 1e-30) if a_exact else None,
    }
    from minsurf.viewer import write_viewer
    viewer_path = write_viewer(solved, output_dir / "catenoid_viewer.html",
                                title="Catenoid — minsurf", preset="catenoid",
                                exact_mesh=exact_mesh, stability_info=stab)

    report.write_report(
        output_dir / "catenoid_report.json",
        preset="catenoid", boundary="two-rings",
        params={"separation": sep, "radius": rad, "method": "implicit", "tau": 0.05},
        mesh=solved, initial_area=initial_area,
        iterations=len(hist.area),
        residual_max_H=hist.residual[-1] if hist.residual else float("nan"),
        converged=hist.converged,
        validation=validation_dict, stability=stab,
        outputs={
            "obj": str(output_dir / "catenoid.obj"),
            "stl": str(output_dir / "catenoid.stl"),
            "render": str(output_dir / "catenoid_render.png"),
            "viewer": str(viewer_path),
        },
    )
    results_summary.append({"preset": "catenoid", "status": "ok", "l2_error": l2_err})

    # -----------------------------------------------------------------------
    # 2. Saddle ring
    # -----------------------------------------------------------------------
    log("\n[2/5] Saddle ring — z = 0.55 sin(2θ), disk topology")
    seed = boundaries.saddle_ring(amplitude=0.55, k=2, n_theta=48, n_radial=20)
    initial_area = seed.total_area()
    t0 = time.time()
    solved_saddle, hist_saddle = flow.solve(seed, method="implicit", tau=0.05, max_iter=2000, tol=1e-6)
    log(f"   Solved in {len(hist_saddle.area)} iters ({time.time()-t0:.1f}s).")
    io.write_obj(solved_saddle, output_dir / "saddle_ring.obj")
    io.write_stl(solved_saddle, output_dir / "saddle_ring.stl")
    visualize.render_surface(solved_saddle, title="Saddle Ring (computed)", path=output_dir / "saddle_ring_render.png")
    write_viewer(solved_saddle, output_dir / "saddle_ring_viewer.html", title="Saddle Ring — minsurf", preset="saddle-ring")
    results_summary.append({"preset": "saddle-ring", "status": "ok"})

    # -----------------------------------------------------------------------
    # 3. Wavy ring (monkey saddle)
    # -----------------------------------------------------------------------
    log("\n[3/5] Wavy ring — z = 0.55 sin(3θ), monkey-saddle topology")
    seed = boundaries.saddle_ring(amplitude=0.55, k=3, n_theta=48, n_radial=20)
    initial_area = seed.total_area()
    solved_wavy, hist_wavy = flow.solve(seed, method="implicit", tau=0.05, max_iter=2000, tol=1e-6)
    log(f"   Solved in {len(hist_wavy.area)} iters.")
    io.write_obj(solved_wavy, output_dir / "wavy_ring.obj")
    io.write_stl(solved_wavy, output_dir / "wavy_ring.stl")
    visualize.render_surface(solved_wavy, title="Wavy Ring / Monkey Saddle", path=output_dir / "wavy_ring_render.png")
    write_viewer(solved_wavy, output_dir / "wavy_ring_viewer.html", title="Wavy Ring — minsurf", preset="wavy-ring")
    results_summary.append({"preset": "wavy-ring", "status": "ok"})

    # -----------------------------------------------------------------------
    # 4. Enneper (analytic — no flow needed)
    # -----------------------------------------------------------------------
    log("\n[4/5] Enneper surface (analytic)")
    enneper_mesh = exact.enneper(r_max=0.8, n_u=32, n_v=32)
    io.write_obj(enneper_mesh, output_dir / "enneper.obj")
    visualize.render_surface(enneper_mesh, title="Enneper Surface (exact)", path=output_dir / "enneper_render.png")
    write_viewer(enneper_mesh, output_dir / "enneper_viewer.html", title="Enneper — minsurf", preset="enneper")
    results_summary.append({"preset": "enneper", "status": "ok"})

    # -----------------------------------------------------------------------
    # 5. Catenoid ↔ Helicoid associate family
    # -----------------------------------------------------------------------
    log("\n[5/5] Associate family: catenoid ↔ helicoid (6 frames)")
    family_dir = output_dir / "associate_family"
    family_dir.mkdir(exist_ok=True)
    import numpy as np
    for i, t in enumerate(np.linspace(0, np.pi / 2, 6)):
        fam_mesh = exact.associate_family(t=t, a=1.0, separation=2.0, n_theta=32, n_z=20)
        label = f"t={t:.2f}"
        io.write_obj(fam_mesh, family_dir / f"associate_{i:02d}.obj")
        visualize.render_surface(fam_mesh, title=f"Associate family {label}", path=family_dir / f"associate_{i:02d}.png")
    log("   Done.")
    results_summary.append({"preset": "associate-family", "status": "ok", "frames": 6})

    # Summary
    summary_path = output_dir / "demo_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(results_summary, fh, indent=2)

    log(f"\n✓ Demo complete. Outputs in: {output_dir}")
    log(f"  Summary: {summary_path}")
