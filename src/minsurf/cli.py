"""minsurf command-line interface.

Subcommands:
  gyroid     Generate a TPMS surface (Gyroid, Schwartz P, Diamond, Neovius)
  demo       Build and solve a curated set, write examples/output/demo/*
  solve      Solve one boundary to a minimal surface
  validate   Run convergence study vs exact catenoid
  view       Open interactive HTML viewer for a mesh/result
  export     Export a result to stl/obj/ply
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_gyroid(args: argparse.Namespace) -> int:
    """Generate a TPMS surface and export to STL / OBJ."""
    from minsurf import io, visualize
    from minsurf.tpms import generate

    surface = args.surface
    cells   = args.cells
    res     = args.resolution
    level   = args.level

    print(f"Generating {surface} ({cells}×{cells}×{cells} cells, resolution={res})...")
    mesh = generate(surface, cells=cells, resolution=res, level=level)
    n_v, n_f = mesh.V.shape[0], mesh.F.shape[0]
    print(f"  vertices={n_v:,}  faces={n_f:,}  area={mesh.total_area():.4f}")

    out_dir = Path(args.output or f"examples/output/{surface}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{surface}_c{cells}"

    fmt      = args.export or "stl"
    out_mesh = out_dir / f"{stem}.{fmt}"
    io.export(mesh, out_mesh, fmt=fmt)
    print(f"Exported: {out_mesh}")

    if not args.no_render:
        render_path = out_dir / f"{stem}_render.png"
        visualize.render_surface(mesh, title=surface.title(), colorby="height",
                                 path=render_path)
        print(f"Render:   {render_path}")

    if not args.no_viewer:
        from minsurf.viewer import write_viewer
        viewer_path = write_viewer(mesh, out_dir / f"{stem}_viewer.html",
                                   title=f"{surface} — minsurf")
        print(f"Viewer:   {viewer_path}")

    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from minsurf.demo import run_demo
    run_demo(output_dir=args.output, verbose=not args.quiet)
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    import numpy as np

    from minsurf import boundaries, exact, flow, metrics, stability
    from minsurf.presets import get_preset

    # Resolve params: preset defaults overridden by CLI args
    params: dict = {}
    if args.preset:
        params = get_preset(args.preset)

    def override(key: str, val) -> None:
        if val is not None:
            params[key] = val

    override("method", args.method)
    override("tau", args.tau)
    override("max_iter", args.max_iter)
    override("tol", args.tol)
    override("n_theta", args.n_theta)
    override("n_radial", args.n_radial)
    override("n_z", args.n_z)
    override("separation", args.separation)
    override("radius", args.radius)
    override("amplitude", args.amplitude)
    override("k", args.k)

    boundary_type = args.boundary or params.get("boundary", "two-rings")
    preset_name = args.preset or "custom"

    # Build seed mesh
    if boundary_type == "two-rings":
        seed = boundaries.two_rings(
            separation=params.get("separation", 1.0),
            radius=params.get("radius", 1.0),
            n_z=params.get("n_z", 24),
            n_theta=params.get("n_theta", 48),
        )
    elif boundary_type == "disk":
        k_val = params.get("k", 2)
        amp = params.get("amplitude", 0.55)
        seed = boundaries.saddle_ring(
            amplitude=amp,
            k=k_val,
            n_theta=params.get("n_theta", 48),
            n_radial=params.get("n_radial", 16),
            radius=params.get("radius", 1.0),
        )
    elif boundary_type == "analytic":
        # Analytic presets: return the exact mesh directly (no flow)
        if preset_name == "enneper":
            mesh = exact.enneper(r_max=params.get("r_max", 0.8),
                                  n_u=params.get("n_u", 32), n_v=params.get("n_v", 32))
        elif preset_name == "helicoid":
            mesh = exact.helicoid(c=params.get("c", 1.0),
                                   n_u=params.get("n_u", 24), n_v=params.get("n_v", 48))
        else:
            print(f"Unknown analytic preset: {preset_name}", file=sys.stderr)
            return 1
        # Skip flow for analytic surfaces
        out_dir = Path(args.output or f"examples/output/{preset_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_outputs(mesh, out_dir, preset_name, params, args, 0.0, 0, 0.0, False, {}, {})
        return 0
    elif boundary_type.startswith("file:"):
        import json as json_mod
        fp = Path(boundary_type[5:])
        if not fp.exists():
            print(f"Boundary file not found: {fp}", file=sys.stderr)
            return 1
        with open(fp) as fh:
            pts = np.array(json_mod.load(fh))
        seed = boundaries.from_points(pts, n_radial=params.get("n_radial", 16))
    else:
        print(f"Unknown boundary type: {boundary_type!r}", file=sys.stderr)
        return 1

    initial_area = seed.total_area()

    # Solve
    method = params.get("method", "implicit")
    tau = params.get("tau", 0.05)
    max_iter = params.get("max_iter", 2000)
    tol = params.get("tol", 1e-6)

    print(f"Solving {preset_name} ({boundary_type}) with {method} flow, τ={tau}, tol={tol}...")
    solved, hist = flow.solve(seed, method=method, tau=tau, max_iter=max_iter, tol=tol, verbose=not args.quiet)
    residual = hist.residual[-1] if hist.residual else float("nan")
    converged = hist.converged
    print(f"Done: {len(hist.area)} iters, area={solved.total_area():.4f}, max|H|={residual:.2e}, converged={converged}")

    # Validation
    validation_dict: dict = {}
    exact_mesh_out = None
    validate_exact = params.get("validate_exact", "")
    if validate_exact == "catenoid" and boundary_type == "two-rings":
        em, a_exact = exact.catenoid_for_rings(
            params.get("separation", 1.0), params.get("radius", 1.0)
        )
        if a_exact:
            l2 = metrics.l2_error_to_catenoid(solved, a_exact)
            neck = metrics.neck_radius(solved)
            validation_dict = {
                "exact": "catenoid",
                "neck_radius_measured": neck,
                "neck_radius_exact": a_exact,
                "l2_error": l2,
                "rel_error_pct": 100 * abs(neck - a_exact) / (a_exact + 1e-30),
            }
            exact_mesh_out = em
            print(f"Exact catenoid: a={a_exact:.4f}, measured neck={neck:.4f}, L2={l2:.4e}")

    # Stability
    stab: dict = {}
    if boundary_type == "two-rings":
        stab = stability.analyze(solved)
        print(f"Stability: ratio={stab['ratio']:.3f}, regime={stab['regime']}")

    out_dir = Path(args.output or f"examples/output/{preset_name}")
    _write_outputs(solved, out_dir, preset_name, params, args,
                   initial_area, len(hist.area), residual, converged,
                   validation_dict, stab, exact_mesh_out, hist)
    return 0


def _write_outputs(
    mesh,
    out_dir: Path,
    preset_name: str,
    params: dict,
    args,
    initial_area: float,
    iters: int,
    residual: float,
    converged: bool,
    validation_dict: dict,
    stab: dict,
    exact_mesh=None,
    hist=None,
) -> None:
    from minsurf import io, report, visualize
    from minsurf.viewer import write_viewer

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict = {}

    export_fmt = getattr(args, "export", params.get("export", "obj")) or "obj"
    if export_fmt and export_fmt != "none":
        obj_path = out_dir / f"{preset_name}.{export_fmt}"
        io.export(mesh, obj_path, fmt=export_fmt)
        output_paths[export_fmt] = str(obj_path)
        print(f"Exported: {obj_path}")
        # Also write OBJ if STL requested (for viewer)
        if export_fmt == "stl":
            obj_path2 = out_dir / f"{preset_name}.obj"
            io.write_obj(mesh, obj_path2)
            output_paths["obj"] = str(obj_path2)

    do_render = not getattr(args, "no_render", False)
    if do_render:
        render_path = out_dir / f"{preset_name}_render.png"
        visualize.render_surface(mesh, title=preset_name, path=render_path)
        output_paths["render"] = str(render_path)
        print(f"Render: {render_path}")

    do_viewer = not getattr(args, "no_viewer", False)
    if do_viewer:
        viewer_path = write_viewer(
            mesh,
            out_dir / f"{preset_name}_viewer.html",
            title=f"{preset_name} — minsurf",
            preset=preset_name,
            exact_mesh=exact_mesh,
            stability_info=stab or None,
        )
        output_paths["viewer"] = str(viewer_path)
        print(f"Viewer: {viewer_path}")

    if hist is not None:
        history_path = out_dir / f"{preset_name}_history.png"
        visualize.render_history(hist, path=history_path)
        output_paths["history"] = str(history_path)

    report.write_report(
        out_dir / f"{preset_name}_report.json",
        preset=preset_name,
        boundary=params.get("boundary", ""),
        params=params,
        mesh=mesh,
        initial_area=initial_area,
        iterations=iters,
        residual_max_H=residual,
        converged=converged,
        validation=validation_dict,
        stability=stab,
        outputs=output_paths,
    )
    print(f"Report: {out_dir / f'{preset_name}_report.json'}")


def cmd_validate(args: argparse.Namespace) -> int:
    """Run convergence study and write docs/validation.md."""
    from minsurf import boundaries, exact, metrics

    print("Running catenoid convergence study...")
    sep, rad = 1.0, 1.0
    a_exact_val = exact._catenoid_neck(sep, rad)
    if a_exact_val is None:
        print("No catenoid solution for these parameters.", file=sys.stderr)
        return 1

    resolutions = [
        {"separation": sep, "radius": rad, "n_theta": nt, "n_z": nz}
        for nt, nz in [(16, 12), (24, 18), (36, 24), (48, 32), (64, 40)]
    ]

    def boundary_fn(**kw: dict):
        return boundaries.two_rings(**kw)

    def exact_fn(mesh):
        err = metrics.l2_error_to_catenoid(mesh, a_exact_val)
        return err, None

    results = metrics.convergence_study(
        boundary_fn,
        exact_fn,
        resolutions,
        flow_kwargs={"method": "implicit", "tau": 0.05, "max_iter": 1000, "tol": 1e-7},
    )

    # Print table
    print(f"\nCatenoid convergence study (sep={sep}, R={rad}, exact a={a_exact_val:.4f})")
    print(f"{'n_theta':>8} {'n_z':>6} {'n_verts':>8} {'L2 err':>12} {'ratio':>8} {'order':>8}")
    print("-" * 60)
    for r in results:
        res = r["resolution_params"]
        ratio_s = f"{r['ratio']:.2f}" if r["ratio"] is not None else "—"
        order_s = f"{r['empirical_order']:.2f}" if r["empirical_order"] is not None else "—"
        print(f"{res['n_theta']:>8} {res['n_z']:>6} {r['n_vertices']:>8} {r['error']:>12.4e} {ratio_s:>8} {order_s:>8}")

    # Write docs/validation.md
    docs_dir = Path(getattr(args, "docs_dir", "docs"))
    docs_dir.mkdir(parents=True, exist_ok=True)
    _write_validation_md(results, a_exact_val, sep, rad, docs_dir)

    # Write convergence plot
    from minsurf import visualize
    assets_dir = docs_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    visualize.render_convergence(results, path=assets_dir / "convergence.png")
    print(f"\nValidation doc written to {docs_dir / 'validation.md'}")
    return 0


def _write_validation_md(results: list, a_exact: float, sep: float, rad: float, docs_dir: Path) -> None:
    lines = [
        "# Validation: Catenoid Convergence",
        "",
        "## Setup",
        f"- Boundary: two coaxial rings, separation = {sep}, radius = {rad}",
        f"- Exact catenoid neck radius: a = {a_exact:.6f}",
        "- Flow: implicit semi-implicit, τ = 0.05, tol = 1×10⁻⁷",
        "- Error metric: RMS radial error |r_i − a cosh(z_i/a)| over interior vertices",
        "",
        "## Convergence Table",
        "",
        "| n_theta | n_z | Vertices | L2 error | Ratio | Emp. order |",
        "|--------:|----:|---------:|---------:|------:|-----------:|",
    ]
    for r in results:
        res = r["resolution_params"]
        ratio_s = f"{r['ratio']:.2f}" if r["ratio"] is not None else "—"
        order_s = f"{r['empirical_order']:.2f}" if r["empirical_order"] is not None else "—"
        lines.append(
            f"| {res['n_theta']} | {res['n_z']} | {r['n_vertices']} "
            f"| {r['error']:.4e} | {ratio_s} | {order_s} |"
        )
    lines += [
        "",
        "## Notes",
        "- The empirical convergence order measures how fast the L2 error decreases as resolution doubles.",
        "- For the cotangent-Laplacian flow on regular meshes, the expected order is approximately 2 (quadratic).",
        "- Boundary layers and mesh regularity affect the observed order in practice.",
        "",
        "## Convergence Plot",
        "",
        "![Convergence plot](assets/convergence.png)",
        "",
        "## Gauss–Bonnet Check",
        "The angle-defect Gaussian curvature integrates to 2πχ within numerical precision.",
        "See `tests/test_operators_sphere.py` for the sphere case.",
    ]
    (docs_dir / "validation.md").write_text("\n".join(lines))


def cmd_view(args: argparse.Namespace) -> int:
    import subprocess
    from pathlib import Path as P

    src = P(args.mesh)
    if not src.exists():
        print(f"File not found: {src}", file=sys.stderr)
        return 1

    from minsurf import io
    from minsurf.viewer import write_viewer

    if src.suffix == ".obj":
        mesh = io.read_obj(src)
    elif src.suffix == ".stl":
        mesh = io.read_stl(src)
    elif src.suffix == ".json":
        import json as json_mod
        rpt = json_mod.loads(src.read_text())
        obj_path = rpt.get("outputs", {}).get("obj")
        if not obj_path:
            print("JSON report has no 'outputs.obj' field.", file=sys.stderr)
            return 1
        mesh = io.read_obj(obj_path)
    else:
        print(f"Unsupported format: {src.suffix}", file=sys.stderr)
        return 1

    viewer_path = src.with_suffix(".html")
    write_viewer(mesh, viewer_path, title=src.stem)
    print(f"Viewer: {viewer_path}")
    with contextlib.suppress(Exception):
        subprocess.run(["open", str(viewer_path)], check=False)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from minsurf import io

    src = Path(args.result)
    if not src.exists():
        print(f"File not found: {src}", file=sys.stderr)
        return 1
    mesh = io.read_obj(src) if src.suffix == ".obj" else io.read_stl(src)
    out_path = Path(args.result).with_suffix(f".{args.format}")
    io.export(mesh, out_path, fmt=args.format)
    print(f"Exported: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minsurf",
        description="Discrete minimal surface toolkit — soap films, cotangent Laplacian flow.",
    )
    parser.add_argument("--version", action="version", version="minsurf 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    # gyroid / tpms
    p_gyroid = sub.add_parser(
        "gyroid",
        help="Generate a TPMS surface (Gyroid, Schwartz P, Diamond, Neovius)",
    )
    p_gyroid.add_argument(
        "--surface",
        choices=["gyroid", "schwartz-p", "diamond", "neovius"],
        default="gyroid",
        help="Surface type (default: gyroid)",
    )
    p_gyroid.add_argument("--cells",      type=int,   default=2,    help="Unit cells per axis")
    p_gyroid.add_argument("--resolution", type=int,   default=50,   help="Grid samples per axis")
    p_gyroid.add_argument("--level",      type=float, default=0.0,  help="Isosurface level (0 = minimal surface)")
    p_gyroid.add_argument("--output",     default=None,             help="Output directory")
    p_gyroid.add_argument("--export",     choices=["stl", "obj", "ply"], default="stl")
    p_gyroid.add_argument("--no-render",  dest="no_render",  action="store_true")
    p_gyroid.add_argument("--no-viewer",  dest="no_viewer",  action="store_true")

    # demo
    p_demo = sub.add_parser("demo", help="Build and solve the curated demo set")
    p_demo.add_argument("--output", default="examples/output/demo")
    p_demo.add_argument("--quiet", action="store_true")

    # solve
    p_solve = sub.add_parser("solve", help="Solve one boundary to a minimal surface")
    p_solve.add_argument("boundary_or_preset", nargs="?", default=None,
                         help="Boundary type or preset name (overridden by --preset/--boundary)")
    p_solve.add_argument("--preset", choices=["catenoid", "two-rings", "saddle-ring", "wavy-ring", "enneper", "helicoid"])
    p_solve.add_argument("--boundary", choices=["disk", "two-rings", "polygon", "analytic"] + ["file:<path>"])
    p_solve.add_argument("--method", choices=["implicit", "explicit"])
    p_solve.add_argument("--tau", type=float)
    p_solve.add_argument("--max-iter", dest="max_iter", type=int)
    p_solve.add_argument("--tol", type=float)
    p_solve.add_argument("--n-theta", dest="n_theta", type=int)
    p_solve.add_argument("--n-radial", dest="n_radial", type=int)
    p_solve.add_argument("--n-z", dest="n_z", type=int)
    p_solve.add_argument("--separation", type=float)
    p_solve.add_argument("--radius", type=float)
    p_solve.add_argument("--amplitude", type=float)
    p_solve.add_argument("-k", dest="k", type=int)
    p_solve.add_argument("--output", default=None)
    p_solve.add_argument("--export", choices=["obj", "stl", "ply", "none"], default=None)
    p_solve.add_argument("--no-render", dest="no_render", action="store_true")
    p_solve.add_argument("--no-viewer", dest="no_viewer", action="store_true")
    p_solve.add_argument("--quiet", action="store_true")

    # validate
    p_val = sub.add_parser("validate", help="Run convergence study, write docs/validation.md")
    p_val.add_argument("--exact", choices=["catenoid"], default="catenoid")
    p_val.add_argument("--docs-dir", default="docs")

    # view
    p_view = sub.add_parser("view", help="Open interactive HTML viewer")
    p_view.add_argument("mesh", help="Path to .obj, .stl, or report .json")

    # export
    p_export = sub.add_parser("export", help="Export geometry to a different format")
    p_export.add_argument("result", help="Path to .obj or .stl file")
    p_export.add_argument("--format", choices=["stl", "obj", "ply"], required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "gyroid": cmd_gyroid,
        "demo": cmd_demo,
        "solve": cmd_solve,
        "validate": cmd_validate,
        "view": cmd_view,
        "export": cmd_export,
    }

    # Handle positional preset/boundary arg for solve
    if args.command == "solve" and args.boundary_or_preset and not args.preset:
        from minsurf.presets import PRESETS
        if args.boundary_or_preset in PRESETS:
            args.preset = args.boundary_or_preset

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
