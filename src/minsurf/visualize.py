"""Visualization utilities using matplotlib — dark theme.

Produces:
- 3D surface render coloured by |H| or z-height
- Catenoid profile vs exact overlay
- Convergence plots (log-log error vs h)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from minsurf.flow import History
from minsurf.mesh import Mesh
from minsurf.operators import mean_curvature

matplotlib.use("Agg")

# --------------------------------------------------------------------------- #
#  Theme constants
# --------------------------------------------------------------------------- #

DARK_BG = "#070d14"
PANEL_BG = "#0d1925"
GRID_COL = "#1a2b3a"
TEXT_COL = "#dbe6ee"
MUT_COL  = "#93a4b2"
ACCENT   = "#2dd4bf"

# Teal → bright teal → amber (matches the Three.js interactive demos)
TEAL_AMBER = LinearSegmentedColormap.from_list(
    "minsurf",
    [(0.04, 0.42, 0.38), (0.18, 0.83, 0.75), (0.96, 0.65, 0.14)],
    N=256,
)

_DARK_RC: dict = {
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL_BG,
    "text.color":       TEXT_COL,
    "axes.labelcolor":  MUT_COL,
    "xtick.color":      MUT_COL,
    "ytick.color":      MUT_COL,
    "axes.edgecolor":   GRID_COL,
    "grid.color":       GRID_COL,
    "grid.alpha":       0.25,
    "legend.facecolor": PANEL_BG,
    "legend.edgecolor": GRID_COL,
    "savefig.facecolor": DARK_BG,
}


def _strip_3d(ax: plt.Axes) -> None:
    """Remove all chrome from a 3-D axes; apply dark fill."""
    ax.set_facecolor(DARK_BG)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("none")
    ax.grid(False)
    ax.set_axis_off()


# --------------------------------------------------------------------------- #
#  3-D surface render
# --------------------------------------------------------------------------- #


def render_surface(
    mesh: Mesh,
    title: str = "",
    colorby: str = "mean_curvature",
    path: str | Path | None = None,
    show: bool = False,
    elev: float = 28.0,
    azim: float = -55.0,
) -> matplotlib.figure.Figure:
    """3-D surface render coloured by |H| or z-height, dark theme.

    Parameters
    ----------
    mesh    : mesh to render.
    title   : plot title (white text; omit for a clean hero render).
    colorby : ``'mean_curvature'``, ``'height'``, or ``'none'``.
    path    : save figure to this path if given.
    show    : call plt.show() if True.
    elev, azim : camera angles.
    """
    V, F = mesh.V, mesh.F

    if colorby == "mean_curvature":
        H      = mean_curvature(mesh)
        H_face = H[F].mean(axis=1)
        H_max  = float(H_face.max()) if H_face.max() > 0 else 1.0
        norm_vals  = H_face / H_max
        cmap   = TEAL_AMBER
        vmin, vmax, cbar_label = 0.0, H_max, "|H|  (mean curvature)"
    elif colorby == "height":
        z_face = V[F].mean(axis=1)[:, 2]
        z_min, z_max = float(z_face.min()), float(z_face.max())
        span   = z_max - z_min or 1.0
        norm_vals  = (z_face - z_min) / span
        cmap   = TEAL_AMBER
        vmin, vmax, cbar_label = z_min, z_max, "z"
    else:
        norm_vals  = np.full(F.shape[0], 0.6)
        cmap   = TEAL_AMBER
        vmin, vmax, cbar_label = 0.0, 1.0, ""

    # Soft directional shading from per-face normals
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n_len = np.linalg.norm(n, axis=1, keepdims=True)
    n /= np.where(n_len > 0, n_len, 1.0)

    key  = np.array([1.2, 2.0, 1.6])
    key  /= np.linalg.norm(key)
    fill = np.array([-0.8, -0.4, -1.0])
    fill /= np.linalg.norm(fill)
    shade = (
        np.clip(n @ key,  0, 1) * 0.60
        + np.clip(-n @ key, 0, 1) * 0.15   # backface
        + np.clip(n @ fill, 0, 1) * 0.18
        + 0.15                               # ambient
    )
    shade = np.clip(shade, 0.0, 1.0)

    rgba = cmap(norm_vals).copy()
    rgba[:, :3] *= shade[:, None]
    rgba[:, 3]   = 0.92

    fig = plt.figure(figsize=(10, 8), facecolor=DARK_BG)
    ax  = fig.add_subplot(111, projection="3d", facecolor=DARK_BG)
    _strip_3d(ax)

    polys = [[V[F[i, 0]], V[F[i, 1]], V[F[i, 2]]] for i in range(F.shape[0])]
    ax.add_collection3d(Poly3DCollection(polys, facecolors=rgba, edgecolors="none"))

    margin = 0.04
    for dim in range(3):
        lo, hi = V[:, dim].min(), V[:, dim].max()
        span = hi - lo or 1.0
        [ax.set_xlim, ax.set_ylim, ax.set_zlim][dim](lo - margin * span, hi + margin * span)

    ax.view_init(elev=elev, azim=azim)

    if title:
        ax.set_title(title, color=TEXT_COL, pad=8, fontsize=13, fontweight="bold")

    if colorby != "none":
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.04, shrink=0.65)
        cbar.set_label(cbar_label, color=MUT_COL, fontsize=9)
        cbar.ax.tick_params(colors=MUT_COL, labelsize=8)
        cbar.outline.set_edgecolor("none")

    fig.tight_layout(pad=0.5)
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------- #
#  Catenoid profile overlay
# --------------------------------------------------------------------------- #


def render_catenoid_profile(
    mesh: Mesh,
    a: float,
    path: str | Path | None = None,
    show: bool = False,
) -> matplotlib.figure.Figure:
    """Radial profile r vs z for a tube mesh, overlaid with exact catenoid."""
    with plt.rc_context(_DARK_RC):
        fig, ax = plt.subplots(figsize=(8, 5))
        V     = mesh.V
        z_all = V[:, 2]
        r_all = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
        idx   = np.argsort(z_all)
        ax.plot(z_all[idx], r_all[idx], ".", ms=2.5, alpha=0.45,
                color=ACCENT, label="mesh vertices")
        z_exact = np.linspace(z_all.min(), z_all.max(), 400)
        r_exact = a * np.cosh(z_exact / a)
        ax.plot(z_exact, r_exact, "-", lw=2, color="#f5a623",
                label=f"exact catenoid  a = {a:.4f}")
        ax.set_xlabel("z")
        ax.set_ylabel("r  =  √(x² + y²)")
        ax.set_title("Catenoid profile — computed vs exact", color=TEXT_COL)
        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
        if show:
            plt.show()
    return fig


# --------------------------------------------------------------------------- #
#  Convergence plot
# --------------------------------------------------------------------------- #


def render_convergence(
    results: list[dict],
    x_key: str = "n_vertices",
    x_label: str = "Number of vertices",
    path: str | Path | None = None,
    show: bool = False,
) -> matplotlib.figure.Figure:
    """Log-log convergence plot: L2 error vs resolution."""
    xs = [r[x_key] for r in results]
    ys = [r["error"] for r in results]

    with plt.rc_context(_DARK_RC):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(xs, ys, "o-", lw=1.8, ms=7, color=ACCENT, label="L2 error")

        if len(xs) >= 2:
            log_x  = np.log(xs)
            log_y  = np.log(ys)
            coeffs = np.polyfit(log_x, log_y, 1)
            order  = coeffs[0]
            x_fit  = np.array([min(xs), max(xs)])
            y_fit  = np.exp(np.polyval(coeffs, np.log(x_fit)))
            ax.loglog(x_fit, y_fit, "--", lw=1.5, color="#f5a623",
                      label=f"slope ≈ {order:.2f}")

        ax.set_xlabel(x_label)
        ax.set_ylabel("L2 radial error")
        ax.set_title("Convergence — catenoid L2 error vs resolution", color=TEXT_COL)
        ax.legend()
        ax.grid(True, which="both", alpha=0.2)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
        if show:
            plt.show()
    return fig


# --------------------------------------------------------------------------- #
#  Flow history
# --------------------------------------------------------------------------- #


def render_history(
    history: History,
    path: str | Path | None = None,
    show: bool = False,
) -> matplotlib.figure.Figure:
    """Area and max|H| vs iteration, dark theme."""
    with plt.rc_context(_DARK_RC):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        iters = list(range(len(history.area)))

        ax1.plot(iters, history.area, "-", lw=1.8, color=ACCENT)
        ax1.set_ylabel("Total area")
        ax1.set_title("Flow convergence history", color=TEXT_COL)
        ax1.grid(True, alpha=0.2)

        ax2.semilogy(iters, history.residual, "-", lw=1.8, color="#f5a623")
        ax2.set_ylabel("max |H|")
        ax2.set_xlabel("Iteration")
        ax2.grid(True, alpha=0.2)

        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
        if show:
            plt.show()
    return fig
