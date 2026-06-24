"""Mesh dataclass and topological utilities.

A triangle mesh is stored as vertices V ∈ R^{n×3} and faces F ∈ Z^{m×3}.
Boundary vertices are flagged in a boolean array and held fixed during flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Mesh:
    """Triangle mesh with pinned boundary.

    Parameters
    ----------
    V : (n, 3) float64 array of vertex positions.
    F : (m, 3) int array of face indices (0-based).
    boundary : (n,) bool array, True = boundary vertex (pinned during flow).
    """

    V: np.ndarray
    F: np.ndarray
    boundary: np.ndarray

    # Cached neighbor lists; built lazily.
    _neighbors: list[np.ndarray] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.V = np.asarray(self.V, dtype=np.float64)
        self.F = np.asarray(self.F, dtype=np.int64)
        self.boundary = np.asarray(self.boundary, dtype=bool)
        assert self.V.ndim == 2 and self.V.shape[1] == 3
        assert self.F.ndim == 2 and self.F.shape[1] == 3
        assert self.boundary.shape == (self.V.shape[0],)

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def neighbors(self) -> list[np.ndarray]:
        """One-ring vertex neighbors for each vertex (CSR-backed).

        Returns a list of length n; entry i is a sorted int array of all
        vertices sharing an edge with vertex i.
        """
        if self._neighbors is not None:
            return self._neighbors
        n = self.V.shape[0]
        rows: list[int] = []
        cols: list[int] = []
        for tri in self.F:
            for k in range(3):
                i, j = int(tri[k]), int(tri[(k + 1) % 3])
                rows.extend([i, j])
                cols.extend([j, i])
        mat = csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(n, n),
        )
        self._neighbors = [mat.indices[mat.indptr[i] : mat.indptr[i + 1]] for i in range(n)]
        return self._neighbors

    def edges(self) -> np.ndarray:
        """Unique undirected edges, shape (e, 2), each row (i<j)."""
        edge_set: set[tuple[int, int]] = set()
        for tri in self.F:
            for k in range(3):
                i, j = int(tri[k]), int(tri[(k + 1) % 3])
                edge_set.add((min(i, j), max(i, j)))
        return np.array(sorted(edge_set), dtype=np.int64)

    def euler_characteristic(self) -> int:
        """χ = V − E + F for this mesh."""
        n_v = self.V.shape[0]
        n_f = self.F.shape[0]
        n_e = self.edges().shape[0]
        return n_v - n_e + n_f

    def copy(self) -> Mesh:
        """Deep copy (V and F are duplicated; boundary is shared)."""
        return Mesh(
            V=self.V.copy(),
            F=self.F.copy(),
            boundary=self.boundary.copy(),
        )

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    def face_normals(self) -> np.ndarray:
        """Un-normalised face normal vectors, shape (m, 3)."""
        v0 = self.V[self.F[:, 0]]
        v1 = self.V[self.F[:, 1]]
        v2 = self.V[self.F[:, 2]]
        return np.cross(v1 - v0, v2 - v0)

    def face_areas(self) -> np.ndarray:
        """Per-face triangle area, shape (m,)."""
        return 0.5 * np.linalg.norm(self.face_normals(), axis=1)

    def total_area(self) -> float:
        """Sum of triangle areas."""
        return float(self.face_areas().sum())

    def n_vertices(self) -> int:
        return int(self.V.shape[0])

    def n_faces(self) -> int:
        return int(self.F.shape[0])
