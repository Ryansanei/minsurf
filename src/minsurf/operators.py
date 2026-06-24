"""Discrete differential geometry operators.

All operators follow the notation of Meyer et al. (2003)
"Discrete Differential-Geometry Operators for Triangulated 2-Manifolds."

The cotangent Laplacian weight for edge (i,j) is:
    w_ij = (cot α_ij + cot β_ij) / 2
where α_ij, β_ij are the angles opposite the edge in the two incident triangles.

The discrete mean curvature vector at vertex i is:
    (L x)_i = (1 / (2 A_i)) Σ_{j ∈ N(i)} (cot α_ij + cot β_ij)(x_j − x_i)

where A_i is the Voronoi area with an obtuse-triangle safeguard (clamped to
one quarter of the incident triangle area when an angle is ≥ π/2).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from minsurf.mesh import Mesh

_EPS = 1e-14  # safe minimum for cotangent denominators


def _safe_cot(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Cotangent of the angle at vertex a in triangle (a, b, c).

    cot θ = cos θ / sin θ = (u·v) / |u×v|  where u = b−a, v = c−a.
    Clipped to [−1e8, 1e8] to handle near-degenerate triangles.
    """
    u = b - a
    v = c - a
    dot = np.einsum("...i,...i->...", u, v)
    cross_mag = np.linalg.norm(np.cross(u, v), axis=-1)
    return dot / np.maximum(cross_mag, _EPS)


def cotangent_laplacian(mesh: Mesh) -> csr_matrix:
    """Symmetric cotangent Laplacian matrix L, shape (n, n).

    L_ij = −(cot α_ij + cot β_ij) / 2   for adjacent i,j
    L_ii = −Σ_{j} L_ij                   (row-sum zero property)

    The returned matrix acts on vertex coordinates as (L @ x).
    To obtain the mean-curvature vector use:
        H_vec = (L @ x) / (2 * vertex_areas)[:, None]

    References
    ----------
    Pinkall & Polthier (1993); Desbrun et al. (1999).
    """
    n = mesh.n_vertices()
    W = lil_matrix((n, n), dtype=np.float64)

    V = mesh.V
    for face in mesh.F:
        i, j, k = int(face[0]), int(face[1]), int(face[2])
        # Angle at each vertex
        cot_k = _safe_cot(V[k], V[i], V[j])  # cot at k, opposite edge (i,j)
        cot_i = _safe_cot(V[i], V[j], V[k])  # cot at i, opposite edge (j,k)
        cot_j = _safe_cot(V[j], V[k], V[i])  # cot at j, opposite edge (i,k)

        # Accumulate half-cotangent weights (factor 1/2 included here)
        for (p, q, w) in [(i, j, cot_k), (j, k, cot_i), (i, k, cot_j)]:
            W[p, q] += 0.5 * w
            W[q, p] += 0.5 * w

    W = W.tocsr()
    # Build L = diag(row-sums) - W  (so L has zero row sums ≈ harmonic)
    row_sums = np.array(W.sum(axis=1)).ravel()
    from scipy.sparse import diags

    L = diags(row_sums) - W
    return L.tocsr()


def vertex_areas(mesh: Mesh) -> np.ndarray:
    """Voronoi vertex areas A_i, shape (n,), with obtuse-triangle safeguard.

    For an obtuse triangle (any angle ≥ π/2), the Voronoi region can extend
    outside the triangle.  We follow Meyer et al. (2003) §3.3 and use:
      - if the angle at i is obtuse: A_i gets half the triangle area.
      - if the angle at j or k is obtuse: A_i gets a quarter of the triangle area.
      - otherwise: A_i gets the standard mixed Voronoi area.
    """
    n = mesh.n_vertices()
    A = np.zeros(n, dtype=np.float64)
    V = mesh.V

    for face in mesh.F:
        i, j, k = int(face[0]), int(face[1]), int(face[2])
        pi, pj, pk = V[i], V[j], V[k]

        # Triangle area
        cross = np.cross(pj - pi, pk - pi)
        area = 0.5 * np.linalg.norm(cross)

        # Angles (in radians) using dot product
        def ang(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            u = b - a
            v = c - a
            cos_a = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + _EPS)
            return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))

        theta_i = ang(pi, pj, pk)
        theta_j = ang(pj, pk, pi)
        theta_k = ang(pk, pi, pj)

        if theta_i >= np.pi / 2:
            A[i] += area / 2.0
            A[j] += area / 4.0
            A[k] += area / 4.0
        elif theta_j >= np.pi / 2:
            A[i] += area / 4.0
            A[j] += area / 2.0
            A[k] += area / 4.0
        elif theta_k >= np.pi / 2:
            A[i] += area / 4.0
            A[j] += area / 4.0
            A[k] += area / 2.0
        else:
            # Voronoi formula: (cot θ_k |e_ij|² + cot θ_j |e_ik|²) / 8
            cot_k = _safe_cot(pk, pi, pj)
            cot_j = _safe_cot(pj, pk, pi)
            cot_i = _safe_cot(pi, pj, pk)
            eij2 = np.dot(pj - pi, pj - pi)
            eik2 = np.dot(pk - pi, pk - pi)
            ejk2 = np.dot(pk - pj, pk - pj)
            A[i] += (cot_k * eij2 + cot_j * eik2) / 8.0
            A[j] += (cot_k * eij2 + cot_i * ejk2) / 8.0
            A[k] += (cot_j * eik2 + cot_i * ejk2) / 8.0

    # Guard against degenerate zero-area vertices
    A = np.maximum(A, _EPS)
    return A


def mean_curvature_vector(mesh: Mesh) -> np.ndarray:
    """Discrete mean-curvature vector at each vertex, shape (n, 3).

    H_vec_i = (L @ V)_i / (2 A_i)

    For interior vertices, H_vec_i = H_i * n_i where H_i is the mean
    curvature (half the sum of principal curvatures) and n_i is the unit
    normal.  Boundary vertex values are set to zero (pinned).
    """
    L = cotangent_laplacian(mesh)
    A = vertex_areas(mesh)
    Lx = L @ mesh.V  # (n, 3)
    H_vec = Lx / (2.0 * A[:, None])
    H_vec[mesh.boundary] = 0.0
    return H_vec


def mean_curvature(mesh: Mesh) -> np.ndarray:
    """Scalar discrete mean curvature magnitude |H_i| at each vertex, shape (n,).

    This is the norm of the mean-curvature vector; it vanishes at boundary
    vertices and should → 0 everywhere on a converged minimal surface.
    """
    return np.linalg.norm(mean_curvature_vector(mesh), axis=1)


def gaussian_curvature(mesh: Mesh) -> np.ndarray:
    """Discrete Gaussian curvature by angle defect, shape (n,).

    K_i = (2π − Σ_t θ_{i,t}) / A_i   (interior)
         = (π − Σ_t θ_{i,t}) / A_i   (boundary; half-angle-defect convention)

    where θ_{i,t} is the interior angle of triangle t at vertex i.
    """
    n = mesh.n_vertices()
    angle_sum = np.zeros(n, dtype=np.float64)
    V = mesh.V

    for face in mesh.F:
        i, j, k = int(face[0]), int(face[1]), int(face[2])
        pi, pj, pk = V[i], V[j], V[k]

        def ang(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            u = b - a
            v = c - a
            cos_a = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + _EPS)
            return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))

        angle_sum[i] += ang(pi, pj, pk)
        angle_sum[j] += ang(pj, pk, pi)
        angle_sum[k] += ang(pk, pi, pj)

    A = vertex_areas(mesh)
    defect = np.where(
        mesh.boundary,
        np.pi - angle_sum,
        2.0 * np.pi - angle_sum,
    )
    return defect / A
