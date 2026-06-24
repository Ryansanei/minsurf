# Use Cases

## 1. Tensile/Membrane Form-Finding for Architecture

Tensile structures (fabric roofs, cable nets, pneumatic membranes) are naturally governed by minimal-surface geometry when the membrane is under uniform tension with no applied load other than the boundary constraints. In this regime, equilibrium requires $H = 0$ everywhere — exactly the problem `minsurf` solves.

### Workflow

1. Specify the boundary as a closed cable or ring geometry (3-D polygon).
2. Build a seed mesh with `boundaries.from_points(boundary_xyz)`.
3. Solve: `flow.solve(seed)`.
4. Export the relaxed surface to OBJ/STL for further structural analysis.

```python
import json, numpy as np
from minsurf import boundaries, flow, io, metrics

# Load boundary from a surveyed cable position (JSON array of [x,y,z] points)
with open("cable_boundary.json") as f:
    pts = np.array(json.load(f))

seed = boundaries.from_points(pts, n_radial=24)
initial_area = seed.total_area()
membrane, hist = flow.solve(seed, method="implicit", tau=0.05, tol=1e-6)

print(f"Membrane area: {membrane.total_area():.3f} m²  (initial: {initial_area:.3f})")
print(f"max |H|: {metrics.max_mean_curvature(membrane):.2e}")

io.write_obj(membrane, "membrane_form.obj")
io.write_stl(membrane, "membrane_form.stl")
```

**Limitations:** `minsurf` solves the *pure* Plateau problem (constant tension, no gravity, no pressure differential). For engineering use with loads, gravity, or material nonlinearity, export the geometry and use a dedicated FEM membrane solver (e.g., Form-Finding Research).

---

## 2. 3-D Printable Minimal-Surface Geometry

Minimal surfaces have beautiful, printable geometry: smooth saddle shapes, thin waists, and intrinsic aesthetic appeal. The STL export from `minsurf` can be opened directly in a slicer (PrusaSlicer, Bambu Studio, Chitubox).

### Printable examples

| Preset | Shape | Print notes |
|--------|-------|------------|
| `catenoid` | Hourglass / throat | Needs support for the overhang near the waist |
| `saddle-ring` | Anti-clastic saddle | Prints flat; no support needed |
| `wavy-ring` | Three-lobe saddle | Best in resin for fine detail |
| `enneper` | Self-intersecting funnel | Print the outer shell only; clip at $r=0.7$ |

### Exporting for printing

```bash
# Solve and export STL
minsurf solve --preset catenoid --export stl --output print/

# Or in Python:
from minsurf import boundaries, flow, io
seed = boundaries.two_rings(separation=1.0, radius=1.0, n_theta=64, n_z=48)
mesh, _ = flow.solve(seed)
io.write_stl(mesh, "catenoid.stl", binary=True)   # binary STL for slicer
```

The resulting STL is a single open surface (not a solid). To make it printable, thicken it using the slicer's "shell thickness" option, or use a surface-to-solid tool (Blender solidify modifier, Meshmixer offset).

**Triply Periodic Minimal Surfaces (TPMS):** Gyroid, Schwarz-P, and Diamond surfaces are widely used for lightweight infill and tissue scaffolds. `minsurf` does not currently compute TPMS (periodic boundary conditions require a different topology), but this is on the roadmap.

---

## 3. Teaching Differential Geometry

`minsurf` was designed to make the bridge between smooth theory and discrete computation explicit. Every smooth concept has a direct discrete analogue implemented transparently in NumPy:

| Smooth concept | Discrete implementation | File |
|----------------|------------------------|------|
| Cotan Laplacian | `cotangent_laplacian()` | `operators.py` |
| Voronoi area | `vertex_areas()` | `operators.py` |
| Mean curvature $H$ | `mean_curvature()` | `operators.py` |
| Gaussian curvature $K$ | `gaussian_curvature()` | `operators.py` |
| Mean-curvature flow | `solve()` | `flow.py` |
| Weierstrass–Enneper rep. | `weierstrass_enneper()` | `exact.py` |
| Associate family | `associate_family()` | `exact.py` |
| Gauss–Bonnet theorem | `gauss_bonnet_residual()` | `metrics.py` |

### Suggested exercises

1. **Gauss–Bonnet sanity check** on a sphere mesh: verify $\sum K_i A_i = 4\pi$.
2. **Observe the catenoid collapse**: solve `two-rings` at increasing `--separation` across 1.3255.
3. **Animate the associate family**: morph from catenoid to helicoid; observe the isometry (same edge lengths in parameter space).
4. **Convergence study**: run `minsurf validate` and verify the empirical order matches theory ($\approx 2$ for the cotangent Laplacian on regular meshes).
5. **Custom boundary**: design a 3-D boundary loop in Python and find the spanning minimal surface.

### Jupyter notebook

`notebooks/minimal_surfaces_demo.ipynb` contains annotated cells reproducing:
- The catenoid relaxation with area/residual plots
- The associate-family animation (6 frames)
- The Enneper surface
- The stability threshold sweep
