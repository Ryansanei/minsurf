"""Exact (analytic) minimal-surface meshes and the Weierstrass–Enneper representation.

All surfaces satisfy H = 0 identically.  The meshes produced here are used
for operator validation and convergence studies.

Catenoid
--------
X(θ, z) = (a cosh(z/a) cos θ,  a cosh(z/a) sin θ,  z)

The parameter a is determined by the boundary condition X(θ, ±h) on two
coaxial circles of radius R: R = a cosh(h/a).  There is a larger root
(stable catenoid) and a smaller root (unstable); we return the larger.

Fundamental forms:
    E = G = cosh²(z/a),  F = 0  (isothermal),
    e = −1,  f = 0,  g = 1  →  H = (e/E + g/G)/2 = 0.  ✓
    K = −1/cosh⁴(z/a).

Weierstrass–Enneper representation
-----------------------------------
X(ζ) = Re ∫_0^ζ (½ f(1−g²),  (i/2) f(1+g²),  fg) dζ

  Surface     f(ζ)       g(ζ)
  ─────────── ────────── ──────
  Enneper     1          ζ
  Catenoid    e^{−ζ}     e^{ζ}
  Helicoid    i e^{−ζ}   e^{ζ}

Associate family: f ↦ e^{iθ} f  deforms catenoid (θ=0) → helicoid (θ=π/2)
isometrically, preserving the metric and curvature.
"""

from __future__ import annotations

import cmath
from collections.abc import Callable

import numpy as np
from scipy.optimize import brentq

from minsurf.mesh import Mesh

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Mesh-building helpers
# ---------------------------------------------------------------------------


def _tube_mesh_from_grid(
    theta: np.ndarray,
    z_vals: np.ndarray,
    X_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Mesh:
    """Build a tube mesh from a parametric surface X(theta, z).

    Parameters
    ----------
    theta : 1-D array of azimuthal parameter values (closed: last wraps to first).
    z_vals : 1-D array of axial parameter values (open: first and last are boundary).
    X_fun : callable(T, Z) → (n_z, n_theta, 3) array.
    """
    n_t = len(theta)
    n_z = len(z_vals)
    T, Z = np.meshgrid(theta, z_vals)  # (n_z, n_theta)
    XYZ = X_fun(T, Z)  # (n_z, n_theta, 3)

    V = XYZ.reshape(-1, 3)
    F_list: list[list[int]] = []

    def idx(iz: int, it: int) -> int:
        return iz * n_t + (it % n_t)

    for iz in range(n_z - 1):
        for it in range(n_t):
            a = idx(iz, it)
            b = idx(iz, it + 1)
            c = idx(iz + 1, it)
            d = idx(iz + 1, it + 1)
            F_list.append([a, b, c])
            F_list.append([b, d, c])

    F = np.array(F_list, dtype=np.int64)

    # Boundary: top and bottom z-rings
    n = V.shape[0]
    boundary = np.zeros(n, dtype=bool)
    boundary[: n_t] = True
    boundary[(n_z - 1) * n_t :] = True

    return Mesh(V=V, F=F, boundary=boundary)


def _disk_mesh_from_grid(
    u_vals: np.ndarray,
    v_vals: np.ndarray,
    X_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    closed_v: bool = False,
) -> Mesh:
    """Build a disk mesh from a parametric surface X(u, v) on a rectangular grid."""
    n_u = len(u_vals)
    n_v = len(v_vals)
    U, V_arr = np.meshgrid(u_vals, v_vals, indexing="ij")  # (n_u, n_v)
    XYZ = X_fun(U, V_arr)  # (n_u, n_v, 3)
    V = XYZ.reshape(-1, 3)

    F_list: list[list[int]] = []

    def idx(iu: int, iv: int) -> int:
        jv = iv % n_v if closed_v else min(iv, n_v - 1)
        return iu * n_v + jv

    for iu in range(n_u - 1):
        for iv in range(n_v - 1 if not closed_v else n_v):
            a = idx(iu, iv)
            b = idx(iu, iv + 1)
            c = idx(iu + 1, iv)
            d = idx(iu + 1, iv + 1)
            F_list.append([a, b, c])
            F_list.append([b, d, c])

    F = np.array(F_list, dtype=np.int64)
    n = V.shape[0]
    # Boundary = outer ring (all u=0 and u=n_u-1 rows, and v edges if not closed)
    boundary = np.zeros(n, dtype=bool)
    boundary[: n_v] = True
    boundary[(n_u - 1) * n_v :] = True
    if not closed_v:
        for iu in range(n_u):
            boundary[iu * n_v] = True
            boundary[iu * n_v + n_v - 1] = True
    return Mesh(V=V, F=F, boundary=boundary)


# ---------------------------------------------------------------------------
# Catenoid
# ---------------------------------------------------------------------------


def _catenoid_neck(separation: float, radius: float) -> float | None:
    """Solve R = a cosh(h/a) for a, returning the LARGER (stable) root or None.

    'separation' is the total distance between the rings (2h).
    'radius' is the ring radius R.

    A solution exists only if separation / radius ≲ 2 * 1.3255 (critical ratio).
    """
    h = separation / 2.0
    R = radius

    def equation(a: float) -> float:
        return a * np.cosh(h / a) - R

    # The critical point occurs at h/a = 1 → a_crit = h; two solutions merge.
    # For h/a < 1 (stable branch) and h/a > 1 (unstable branch).
    # We want the stable (larger a) root → a > h.
    # Upper search bound: a = R (neck = radius, only valid at separation=0)
    R * 1.01

    # Check if any solution exists (discriminant check: min of a cosh(h/a) - R = 0)
    # Minimum of a cosh(h/a) over a is at a=h: value = h * cosh(1) ≈ 1.5431 h
    # So a solution exists iff R ≥ h * cosh(1).
    if h * np.cosh(1.0) > R:
        return None  # No catenoid solution — Goldschmidt regime.

    # Find the larger root (stable, a > h → h/a < 1):
    # equation changes sign between a=h and a=R
    try:
        a_stable = brentq(equation, h + _EPS, R, xtol=1e-12, full_output=False)
        return float(a_stable)
    except ValueError:
        return None


def catenoid(
    a: float = 1.0,
    separation: float = 2.0,
    n_theta: int = 48,
    n_z: int = 32,
) -> Mesh:
    """Exact catenoid mesh X(θ,z) = (a cosh(z/a) cos θ, a cosh(z/a) sin θ, z).

    Parameters
    ----------
    a : neck radius (catenoid parameter).
    separation : total z-span (boundary at z = ±separation/2).
    n_theta : azimuthal resolution.
    n_z : axial resolution (including boundary rings).
    """
    h = separation / 2.0
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z_vals = np.linspace(-h, h, n_z)

    def X_fun(T: np.ndarray, Z: np.ndarray) -> np.ndarray:
        r = a * np.cosh(Z / a)
        return np.stack([r * np.cos(T), r * np.sin(T), Z], axis=-1)

    return _tube_mesh_from_grid(theta, z_vals, X_fun)


def catenoid_for_rings(
    separation: float,
    radius: float,
    n_theta: int = 48,
    n_z: int = 32,
) -> tuple[Mesh | None, float | None]:
    """Build the exact stable catenoid matching two coaxial rings.

    Returns (mesh, neck_radius) or (None, None) if no solution exists.
    """
    a = _catenoid_neck(separation, radius)
    if a is None:
        return None, None
    mesh = catenoid(a=a, separation=separation, n_theta=n_theta, n_z=n_z)
    return mesh, a


# ---------------------------------------------------------------------------
# Helicoid
# ---------------------------------------------------------------------------


def helicoid(
    c: float = 1.0,
    u_max: float = 1.0,
    n_u: int = 24,
    n_v: int = 48,
) -> Mesh:
    """Exact helicoid X(u,v) = (u cos v, u sin v, c v).

    Parameters
    ----------
    c : pitch parameter (z-advance per radian).
    u_max : radial extent (u ∈ [0, u_max]).
    n_u, n_v : grid resolution.
    """
    u_vals = np.linspace(0, u_max, n_u)
    v_vals = np.linspace(0, 2 * np.pi, n_v, endpoint=False)

    def X_fun(U: np.ndarray, V: np.ndarray) -> np.ndarray:
        return np.stack([U * np.cos(V), U * np.sin(V), c * V], axis=-1)

    return _disk_mesh_from_grid(u_vals, v_vals, X_fun, closed_v=True)


# ---------------------------------------------------------------------------
# Enneper
# ---------------------------------------------------------------------------


def enneper(
    r_max: float = 1.0,
    n_u: int = 32,
    n_v: int = 32,
) -> Mesh:
    """Exact Enneper surface X(u,v) = (u − u³/3 + uv², v − v³/3 + u²v, u² − v²).

    Parameters
    ----------
    r_max : domain radius in (u,v) plane (keep ≤ 1 to avoid self-intersection).
    n_u, n_v : grid resolution.
    """
    u_vals = np.linspace(-r_max, r_max, n_u)
    v_vals = np.linspace(-r_max, r_max, n_v)

    def X_fun(U: np.ndarray, V: np.ndarray) -> np.ndarray:
        x = U - U**3 / 3 + U * V**2
        y = V - V**3 / 3 + U**2 * V
        z = U**2 - V**2
        return np.stack([x, y, z], axis=-1)

    return _disk_mesh_from_grid(u_vals, v_vals, X_fun)


# ---------------------------------------------------------------------------
# Scherk (doubly-periodic patch)
# ---------------------------------------------------------------------------


def scherk(
    n: int = 32,
    half_width: float = 1.2,
) -> Mesh:
    """Scherk's first surface: e^z cos x = cos y  on a square patch.

    Sampled on a (u,v) grid avoiding the singularities (|u|, |v| → π/2).

    Parameters
    ----------
    n : grid resolution per axis.
    half_width : half-width of the domain (< π/2 ≈ 1.5708).
    """
    half_width = min(half_width, np.pi / 2 - 0.05)
    x_vals = np.linspace(-half_width, half_width, n)
    y_vals = np.linspace(-half_width, half_width, n)

    def X_fun(U: np.ndarray, V: np.ndarray) -> np.ndarray:
        cos_u = np.cos(U)
        cos_v = np.cos(V)
        # Clamp to avoid log(0)
        ratio = np.clip(cos_v / np.where(np.abs(cos_u) < _EPS, _EPS, cos_u), _EPS, None)
        z = np.log(ratio)
        return np.stack([U, V, z], axis=-1)

    return _disk_mesh_from_grid(x_vals, y_vals, X_fun)


# ---------------------------------------------------------------------------
# Weierstrass–Enneper representation
# ---------------------------------------------------------------------------


def _integrate_we(
    f: Callable[[complex], complex],
    g: Callable[[complex], complex],
    domain_pts: np.ndarray,
) -> np.ndarray:
    """Numerically integrate the Weierstrass–Enneper integrand along rays from origin.

    For each ζ = u + iv in domain_pts, integrate along the straight line
    0 → ζ using composite Simpson's rule with n=256 sub-intervals.

    The integrand is:
        Φ(ζ) = (½ f(1−g²),  (i/2) f(1+g²),  fg)

    Returns Re(∫ Φ dζ) for each input point.
    """
    n_pts = len(domain_pts)
    result = np.zeros((n_pts, 3), dtype=np.float64)
    n_int = 256

    for ip, zeta_end in enumerate(domain_pts):
        t_arr = np.linspace(0.0, 1.0, n_int + 1)
        zeta_arr = zeta_end * t_arr
        dz = zeta_end / n_int  # constant spacing

        # Evaluate integrand at each t
        X_c = np.zeros(n_int + 1, dtype=complex)
        Y_c = np.zeros(n_int + 1, dtype=complex)
        Z_c = np.zeros(n_int + 1, dtype=complex)

        for k, z in enumerate(zeta_arr):
            fv = f(z)
            gv = g(z)
            g2 = gv * gv
            X_c[k] = 0.5 * fv * (1.0 - g2)
            Y_c[k] = 0.5j * fv * (1.0 + g2)
            Z_c[k] = fv * gv

        # Simpson integration
        from scipy.integrate import simpson

        result[ip, 0] = float(np.real(simpson(X_c, dx=np.abs(dz))))
        result[ip, 1] = float(np.real(simpson(Y_c, dx=np.abs(dz))))
        result[ip, 2] = float(np.real(simpson(Z_c, dx=np.abs(dz))))

    return result


def weierstrass_enneper(
    f: Callable[[complex], complex],
    g: Callable[[complex], complex],
    domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    n_u: int = 32,
    n_v: int = 32,
) -> Mesh:
    """Minimal surface from the Weierstrass–Enneper representation.

    Parameters
    ----------
    f, g : holomorphic functions; f*g² must be holomorphic everywhere in domain.
    domain : (u_min, u_max, v_min, v_max) — real/imaginary extent of ζ = u+iv.
    n_u, n_v : grid resolution.

    Examples
    --------
    Enneper:   f=lambda z: 1,          g=lambda z: z
    Catenoid:  f=lambda z: cmath.exp(-z), g=lambda z: cmath.exp(z)
    Helicoid:  f=lambda z: 1j*cmath.exp(-z), g=lambda z: cmath.exp(z)
    """
    u_min, u_max, v_min, v_max = domain
    u_arr = np.linspace(u_min, u_max, n_u)
    v_arr = np.linspace(v_min, v_max, n_v)

    # Build complex domain grid
    U, V = np.meshgrid(u_arr, v_arr, indexing="ij")
    zeta_flat = (U + 1j * V).ravel().astype(complex)

    # Integrate to get surface points
    XYZ_flat = _integrate_we(f, g, zeta_flat)
    XYZ = XYZ_flat.reshape(n_u, n_v, 3)
    V_arr_out = XYZ.reshape(-1, 3)

    # Triangulate structured grid
    F_list: list[list[int]] = []
    for iu in range(n_u - 1):
        for iv in range(n_v - 1):
            a = iu * n_v + iv
            b = iu * n_v + iv + 1
            c = (iu + 1) * n_v + iv
            d = (iu + 1) * n_v + iv + 1
            F_list.append([a, b, c])
            F_list.append([b, d, c])

    F = np.array(F_list, dtype=np.int64)
    n = V_arr_out.shape[0]
    boundary = np.zeros(n, dtype=bool)
    # Mark outer ring as boundary
    boundary[: n_v] = True
    boundary[(n_u - 1) * n_v :] = True
    for iu in range(n_u):
        boundary[iu * n_v] = True
        boundary[iu * n_v + n_v - 1] = True

    return Mesh(V=V_arr_out, F=F, boundary=boundary)


# ---------------------------------------------------------------------------
# Associate family (catenoid ↔ helicoid)
# ---------------------------------------------------------------------------


def associate_family(
    t: float,
    a: float = 1.0,
    separation: float = 2.0,
    n_theta: int = 48,
    n_z: int = 32,
) -> Mesh:
    """Associate family interpolating catenoid (t=0) ↔ helicoid (t=π/2).

    The Weierstrass data (f, g) for the catenoid is (e^{−ζ}, e^{ζ}); the
    family is f_t = e^{it} f, keeping g fixed.  The resulting surface is
    isometric to the catenoid (same Gaussian curvature K, same metric).

    Parameters
    ----------
    t : phase angle in [0, π/2].  t=0 → catenoid; t=π/2 → helicoid.
    a : catenoid neck parameter.
    separation : z-span.
    n_theta, n_z : resolution.

    Returns
    -------
    Mesh of the associate-family surface at phase t.
    """
    phase = cmath.exp(1j * t)
    h = separation / 2.0
    # Parametrise on (θ, z) ≅ (Im ζ, Re ζ) for the catenoid map.
    # Use W-E representation with domain strips.
    u_max = h  # Re(ζ) ∈ [-h, h]
    u_arr = np.linspace(-u_max, u_max, n_z)
    v_arr = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    U, V = np.meshgrid(u_arr, v_arr, indexing="ij")  # (n_z, n_theta)
    zeta = U + 1j * V

    # Catenoid W-E data with phase rotation
    def f_t(z: complex) -> complex:
        return phase * cmath.exp(-z)

    def g(z: complex) -> complex:
        return cmath.exp(z)

    # For efficiency, evaluate the analytic W-E integral analytically:
    # X(ζ) = Re[ e^{it} * (catenoid analytic form) ]
    # Catenoid analytic X = Re(-cosh ζ, -i sinh ζ, ζ) for unit neck (rescale by a)
    # Actually:
    #   Φ_x = ½ e^{-ζ}(1 - e^{2ζ}) = ½(e^{-ζ} - e^{ζ}) = -sinh ζ
    #   Φ_y = (i/2) e^{-ζ}(1 + e^{2ζ}) = (i/2)(e^{-ζ} + e^{ζ}) = i cosh ζ
    #   Φ_z = e^{-ζ} e^{ζ} = 1
    # So ∫Φ dζ = (-cosh ζ, i sinh ζ, ζ)  → Re = (-cosh Re ζ cos Im ζ, -sinh Re ζ sin Im ζ, Re ζ)
    # Wait, that's for unit a.  Let's just evaluate directly.

    # Associate family derived from W-E with f = -e^{-ζ}, g = e^ζ.
    # The correct integrand is Φ = (sinh ζ, -i cosh ζ, -1), giving:
    #   X(ζ) = Re(e^{it} · (cosh ζ, -i sinh ζ, -ζ))    [with basepoint shift absorbed]
    #
    # t=0 → catenoid: (cosh u cos v, cosh u sin v, -u)   (r = cosh(u), z = -u)
    # t=π/2 → helicoid: (-sinh u sin v, sinh u cos v, v)
    #
    # Isometry holds: E = G = cosh²u, F = 0  for all t.  (Proved in §1.)
    u = zeta.real   # axial parameter (z/a before scaling)
    v = zeta.imag   # azimuthal angle

    ct, st = np.cos(t), np.sin(t)
    # Catenoid component (t=0)
    Xc = np.cosh(u) * np.cos(v)
    Yc = np.cosh(u) * np.sin(v)
    Zc = -u  # z-axis is -u; cosh(-u) = cosh(u) so r = a*cosh(|z|/a)

    # Helicoid component (t=π/2): X = Re(i*(cosh ζ, -i sinh ζ, -ζ))
    #   Re(i cosh ζ) = Re(i(cosh u cos v + i sinh u sin v)) = -sinh u sin v
    #   Re(-i·(-i sinh ζ)) = Re(-sinh ζ) = -sinh u cos v ... nope
    # Correct: Re(i·(-i sinh ζ)) = Re(sinh ζ) = sinh u cos v
    # And Re(i·(-ζ)) = Re(-iu + v) = v
    Xh = -np.sinh(u) * np.sin(v)   # Re(i cosh ζ)
    Yh = np.sinh(u) * np.cos(v)    # Re(i·(-i sinh ζ)) = Re(sinh ζ)
    Zh = v                           # Re(i·(-ζ)) = v

    X = a * (ct * Xc + st * Xh)
    Y = a * (ct * Yc + st * Yh)
    Z = a * (ct * Zc + st * Zh)

    XYZ = np.stack([X, Y, Z], axis=-1)  # (n_z, n_theta, 3)
    V_arr_out = XYZ.reshape(-1, 3)

    # Triangulate
    F_list: list[list[int]] = []
    for iu in range(n_z - 1):
        for iv in range(n_theta):
            iv1 = (iv + 1) % n_theta
            a_idx = iu * n_theta + iv
            b_idx = iu * n_theta + iv1
            c_idx = (iu + 1) * n_theta + iv
            d_idx = (iu + 1) * n_theta + iv1
            F_list.append([a_idx, b_idx, c_idx])
            F_list.append([b_idx, d_idx, c_idx])

    F = np.array(F_list, dtype=np.int64)
    n = V_arr_out.shape[0]
    boundary = np.zeros(n, dtype=bool)
    boundary[: n_theta] = True
    boundary[(n_z - 1) * n_theta :] = True

    return Mesh(V=V_arr_out, F=F, boundary=boundary)
