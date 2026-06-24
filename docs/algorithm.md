# Algorithm and Pipeline

## Overview

```
boundary curve(s)
      ↓
  seed mesh          boundaries.py — disk or tube topology, boundary pinned
      ↓
mean-curvature flow  flow.py — implicit (M + τL)X = MX, boundary rows pinned
      ↓
minimal surface      operators.py — verify Lx ≈ 0, max|H| < tol
      ↓
validate / export    metrics.py, io.py, report.py, viewer.py
```

---

## Step 1: Boundary specification

`boundaries.py` generates a seed mesh from a boundary curve:

- **`two_rings`**: two coaxial circles → tube topology (χ=0). Used for catenoid.
- **`disk_boundary` / `saddle_ring`**: one closed curve → disk topology (χ=1).
- **`polygon_boundary` / `from_points`**: arbitrary 3-D boundary polygon.

The seed mesh triangulates the interior (structured quads split into triangles) with interior vertices placed on linear interpolation between boundary and centroid. Boundary vertices are flagged `boundary=True` and never updated.

---

## Step 2: Discrete operators

`operators.py` implements DDG operators from scratch (NumPy/SciPy):

### Cotangent Laplacian

For each triangular face $(i, j, k)$ with angle $\alpha$ at $k$ (opposite edge $i$–$j$):
$$w_{ij} \mathrel{+}= \tfrac{1}{2}\cot\alpha_{k}, \quad w_{ji} \mathrel{+}= \tfrac{1}{2}\cot\alpha_{k}$$
(and similarly for the other vertex). The matrix `L = diag(row_sums) - W` is constructed from the symmetric weight matrix `W`.

Safe cotangent: `cot θ = dot(u,v) / max(|u×v|, ε)` to avoid division by near-zero for degenerate triangles.

### Vertex (Voronoi) areas

For each face:
- If any angle ≥ π/2: use triangle-area fractions (1/2 or 1/4 per vertex, Meyer et al. §3.3).
- Otherwise: mixed Voronoi area formula:
$$A_i \mathrel{+}= \tfrac{1}{8}\bigl(\cot\theta_k \,|e_{ij}|^2 + \cot\theta_j \,|e_{ik}|^2\bigr)$$

### Mean curvature vector

$$\mathbf{H}_i = \frac{(\mathbf{L}\mathbf{x})_i}{2A_i}, \qquad |\mathbf{H}_i| = 0 \text{ on a minimal surface.}$$

### Gaussian curvature (angle defect)

$$K_i = \frac{2\pi - \sum_t \theta_{i,t}}{A_i} \quad (\text{interior}), \qquad K_i = \frac{\pi - \sum_t \theta_{i,t}}{A_i} \quad (\text{boundary}).$$

---

## Step 3: Flow to minimal surface

`flow.py` implements both solvers.

### Explicit step

$$\mathbf{x}_i^{n+1} = \mathbf{x}_i^n - \tau \frac{(\mathbf{L}\mathbf{x}^n)_i}{2A_i}, \quad i \notin B.$$

Stability: $\tau < h^2/4 \approx (2\pi/N_\theta)^2 / 4$. Use for diagnostics; prefer implicit for production.

### Implicit step (default)

Assemble and solve the linear system:
$$(\mathbf{M} + \tau \mathbf{L})\,\mathbf{x}^{n+1} = \mathbf{M}\,\mathbf{x}^n,$$
with boundary rows replaced by identity constraints. Solved per coordinate axis (x, y, z) with conjugate gradient (`scipy.sparse.linalg.cg`). The system is SPD for all $\tau > 0$, so CG always converges.

### Convergence and stopping

The solver records `area` and `max|H|` (maximum mean curvature over interior vertices) per iteration. Stopping criteria:
1. `max|H| < tol` (residual tolerance met).
2. Area stalled: relative change `< 1e-10` for 50 consecutive steps (discrete equilibrium reached — this is the typical termination for medium-resolution meshes, where the mesh discretisation sets a floor on `max|H|` ≈ O(h²)).
3. `max_iter` reached.

Area monotonicity is guaranteed by the implicit scheme's variational structure.

---

## Step 4: Validation and export

- `metrics.py`: area, `max|H|`, neck radius, L2 error vs exact catenoid, Gauss–Bonnet check.
- `exact.py`: exact meshes and the Weierstrass–Enneper representation.
- `stability.py`: separation/radius ratio, collapse detection, Jacobi eigenvalue.
- `io.py`: OBJ, STL (binary/ASCII), PLY export (hand-written, no external dependency).
- `report.py`: JSON report with all parameters and results.
- `viewer.py`: self-contained WebGL HTML viewer generated from the solved mesh.

---

## Complexity

| Operation | Cost |
|-----------|------|
| Laplacian assembly | O(F) = O(N) for regular meshes |
| Implicit step (CG) | O(N · k) per solve, k = CG iterations (~10–30 for well-conditioned systems) |
| Convergence (iterations) | Typically 50–200 for standard presets |
| Total | O(N · max_iter) |

For a mesh with N = 2000 vertices, a solve takes ~1 s on a laptop.

---

## Implementation notes

- All operators are implemented in NumPy/SciPy. No libigl or other external geometry libraries are used for the core math — the point is transparency.
- The cotangent Laplacian is assembled in `lil_matrix` format and converted to CSR for the solver. The CG solver in `scipy.sparse.linalg` is used with a loose tolerance (`atol=1e-10`) since the outer flow iteration is the controlling tolerance.
- The boundary pinning in the implicit step sets boundary rows to identity (`lhs[b,b]=1, rhs[b]=V[b]`), which is simple and robust. More sophisticated approaches (Schur complement) are not needed at this scale.
