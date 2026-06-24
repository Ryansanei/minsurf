"""Catenoid stability analysis.

A catenoid spanning two equal coaxial rings of radius R at z = ±h exists
only while the separation-to-radius ratio satisfies:

    2h / R ≤ 2 * arcosh(x_c) / x_c ≈ 1.3255 * 2

where x_c ≈ 1.5088 is the unique positive root of tanh(x) = 1/x.
The critical ratio is commonly quoted as:

    (2h) / R ≈ 1.3255

Beyond this threshold the only area-minimising surface is the **Goldschmidt**
configuration: two flat disks (one per ring) connected by a line segment of
zero area.

References
----------
Isenberg (1978) "The Science of Soap Films and Soap Bubbles", §7.3.
Oprea (2007) "Differential Geometry and Its Applications", §6.4.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from minsurf.mesh import Mesh
from minsurf.operators import cotangent_laplacian, vertex_areas

# Critical ratio 2h/R beyond which the catenoid ceases to exist.
# Numerically: 2 arcosh(x_c)/x_c where x_c solves tanh(x)=1/x.
CRITICAL_RATIO = 1.3255  # separation / radius


def separation_over_radius(mesh: Mesh) -> float:
    """Estimate the separation/radius ratio from boundary vertices.

    Assumes a tube-topology mesh where boundary vertices form two rings.
    The z-spread gives the separation, and the mean radial distance gives R.
    """
    boundary_V = mesh.V[mesh.boundary]
    if boundary_V.shape[0] < 4:
        return float("nan")
    z = boundary_V[:, 2]
    r = np.sqrt(boundary_V[:, 0] ** 2 + boundary_V[:, 1] ** 2)
    sep = float(z.max() - z.min())
    rad = float(r.mean())
    if rad < 1e-14:
        return float("inf")
    return sep / rad


def stability_regime(ratio: float) -> str:
    """Classify a separation/radius ratio into 'stable', 'unstable', or 'collapsed'."""
    if ratio < CRITICAL_RATIO * 0.99:
        return "stable"
    if ratio < CRITICAL_RATIO * 1.01:
        return "unstable"
    return "collapsed"


def detect_collapse(mesh: Mesh, threshold: float = 0.05) -> bool:
    """Return True if the neck has pinched off (neck radius < threshold * R).

    Parameters
    ----------
    mesh : solved mesh.
    threshold : fraction of boundary radius below which collapse is declared.
    """
    boundary_V = mesh.V[mesh.boundary]
    if boundary_V.shape[0] < 2:
        return False
    r_boundary = float(np.sqrt(boundary_V[:, 0] ** 2 + boundary_V[:, 1] ** 2).mean())
    interior = ~mesh.boundary
    if not interior.any():
        return False
    r_interior = np.sqrt(mesh.V[interior, 0] ** 2 + mesh.V[interior, 1] ** 2)
    neck = float(r_interior.min())
    return neck < threshold * r_boundary


def jacobi_operator_eigenvalue(mesh: Mesh) -> float:
    """Smallest eigenvalue of the Jacobi (area Hessian) operator on the free vertices.

    The Jacobi operator is J = −Δ − |A|² where Δ is the Laplace–Beltrami
    operator and |A|² = k₁² + k₂² is the squared norm of the second fundamental
    form.  For a minimal surface, |A|² = −2K (Gaussian curvature).

    A negative smallest eigenvalue indicates the surface is an unstable saddle
    of the area functional (Morse index ≥ 1).

    Approximation used here: J_discrete ≈ L_interior − diag(2K_i) where K_i
    is the discrete Gaussian curvature.  This is an approximation valid for
    well-resolved meshes.

    Returns
    -------
    Smallest eigenvalue of the restricted Jacobi operator on interior vertices.
    """
    from minsurf.operators import gaussian_curvature

    K = gaussian_curvature(mesh)
    interior = np.where(~mesh.boundary)[0]
    if len(interior) < 2:
        return float("nan")

    L = cotangent_laplacian(mesh)
    A_v = vertex_areas(mesh)

    # Restrict to interior vertices
    L_int = L[np.ix_(interior, interior)]
    # Jacobi ≈ L_int / (2 A_i) + diag(|A|² / (2A_i)) where |A|² = -2K for minimal surface
    # Discretize as: mass-normalised L − diag(2K_i)
    # J_ij = (L_int)_ij / A_i  for off-diag  (mass normalised)
    A_int = A_v[interior]
    # Scale rows by 1/A_i  (M^{-1} L form)
    from scipy.sparse import diags

    M_inv = diags(1.0 / A_int)
    Jacobi = M_inv @ L_int - diags(2.0 * K[interior])

    # Smallest algebraic eigenvalue (most negative → unstable direction)
    try:
        vals = eigsh(Jacobi, k=1, which="SA", return_eigenvectors=False, tol=1e-6)
        return float(vals[0])
    except Exception:
        return float("nan")


def analyze(mesh: Mesh) -> dict:
    """Full stability analysis of a solved mesh.

    Returns
    -------
    dict with keys: ratio, threshold, regime, collapsed, min_hessian_eigenvalue.
    """
    ratio = separation_over_radius(mesh)
    regime = stability_regime(ratio)
    collapsed = detect_collapse(mesh)
    lam = jacobi_operator_eigenvalue(mesh)
    return {
        "ratio": ratio,
        "threshold": CRITICAL_RATIO,
        "regime": regime,
        "collapsed": collapsed,
        "min_hessian_eigenvalue": lam,
    }
