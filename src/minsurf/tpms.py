"""Triply Periodic Minimal Surfaces (TPMS).

Generates Gyroid, Schwartz-P, Diamond, and Neovius surfaces via marching
cubes.  All four satisfy H = 0 everywhere — they are genuine minimal surfaces.

Requires scikit-image for marching cubes:
    pip install scikit-image

Quick start
-----------
>>> from minsurf.tpms import gyroid
>>> mesh = gyroid(cells=2, resolution=50)
>>> mesh.total_area()

CLI
---
    minsurf gyroid --cells 2 --resolution 50 --output gyroid.stl
"""

from __future__ import annotations

import numpy as np

from minsurf.mesh import Mesh

SURFACES: tuple[str, ...] = ("gyroid", "schwartz-p", "diamond", "neovius")


def _scalar_field(name: str, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate the TPMS implicit function F(x,y,z) = 0 defines the surface."""
    if name == "gyroid":
        return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    if name == "schwartz-p":
        return np.cos(x) + np.cos(y) + np.cos(z)
    if name == "diamond":
        return (
            np.sin(x) * np.sin(y) * np.sin(z)
            + np.sin(x) * np.cos(y) * np.cos(z)
            + np.cos(x) * np.sin(y) * np.cos(z)
            + np.cos(x) * np.cos(y) * np.sin(z)
        )
    if name == "neovius":
        return (
            3.0 * (np.cos(x) + np.cos(y) + np.cos(z))
            + 4.0 * np.cos(x) * np.cos(y) * np.cos(z)
        )
    raise ValueError(f"Unknown TPMS surface: {name!r}.  Choose from: {SURFACES}")


def generate(
    name: str = "gyroid",
    cells: int = 2,
    resolution: int = 50,
    scale: float = 1.0,
    level: float = 0.0,
) -> Mesh:
    """Generate a TPMS mesh via marching cubes.

    Parameters
    ----------
    name       : surface name — ``'gyroid'``, ``'schwartz-p'``, ``'diamond'``,
                 or ``'neovius'``.
    cells      : number of unit cells in each direction.
    resolution : grid samples per direction (higher = finer mesh).
    scale      : uniform scale applied to the output vertices.
    level      : isosurface level (0 gives the minimal surface; non-zero
                 gives offset surfaces).

    Returns
    -------
    Mesh with no boundary vertices (periodic interior surface).

    Raises
    ------
    ImportError
        If scikit-image is not installed.
    """
    try:
        from skimage.measure import marching_cubes  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "TPMS generation requires scikit-image.\n"
            "Install it with:  pip install scikit-image"
        ) from exc

    name = name.lower()
    if name not in SURFACES:
        raise ValueError(f"Unknown TPMS: {name!r}.  Choose from: {SURFACES}")

    # Grid covers `cells` unit cells; one unit cell = 2π in parameter space
    period = 2.0 * np.pi
    N = resolution
    t = np.linspace(0.0, period * cells, N + 1)
    x, y, z = np.meshgrid(t, t, t, indexing="ij")

    vol     = _scalar_field(name, x, y, z)
    spacing = (period * cells / N,) * 3

    verts, faces, _normals, _ = marching_cubes(vol, level=level, spacing=spacing)

    # Normalise so one unit cell spans [0, scale] in each axis
    verts = verts * (scale / (period * cells / period))

    V        = verts.astype(np.float64)
    F        = faces.astype(np.int64)
    boundary = np.zeros(len(V), dtype=bool)   # periodic — no boundary

    return Mesh(V=V, F=F, boundary=boundary)


# --------------------------------------------------------------------------- #
#  Convenience wrappers
# --------------------------------------------------------------------------- #


def gyroid(cells: int = 2, resolution: int = 50, scale: float = 1.0) -> Mesh:
    """Gyroid TPMS  —  sin x cos y + sin y cos z + sin z cos x = 0."""
    return generate("gyroid", cells=cells, resolution=resolution, scale=scale)


def schwartz_p(cells: int = 2, resolution: int = 50, scale: float = 1.0) -> Mesh:
    """Schwartz P surface  —  cos x + cos y + cos z = 0."""
    return generate("schwartz-p", cells=cells, resolution=resolution, scale=scale)


def diamond(cells: int = 2, resolution: int = 50, scale: float = 1.0) -> Mesh:
    """Schwartz D (Diamond) TPMS."""
    return generate("diamond", cells=cells, resolution=resolution, scale=scale)


def neovius(cells: int = 2, resolution: int = 50, scale: float = 1.0) -> Mesh:
    """Neovius surface  —  3(cos x + cos y + cos z) + 4 cos x cos y cos z = 0."""
    return generate("neovius", cells=cells, resolution=resolution, scale=scale)


__all__ = ["SURFACES", "generate", "gyroid", "schwartz_p", "diamond", "neovius"]
