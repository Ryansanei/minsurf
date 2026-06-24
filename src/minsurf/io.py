"""Mesh I/O: OBJ, STL (ASCII), PLY read and write.

Implemented by hand to keep dependencies minimal and remain transparent.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from minsurf.mesh import Mesh

# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------


def write_obj(mesh: Mesh, path: str | Path) -> None:
    """Write a Wavefront OBJ file from a Mesh.

    Parameters
    ----------
    mesh : Mesh to export.
    path : output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# minsurf OBJ export\n")
        for v in mesh.V:
            fh.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for f in mesh.F:
            # OBJ is 1-indexed
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def read_obj(path: str | Path) -> Mesh:
    """Read a Wavefront OBJ file.

    Returns a Mesh with boundary=False for all vertices (boundary unknown).
    """
    path = Path(path)
    V: list[list[float]] = []
    F: list[list[int]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                V.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # Support f v, f v/t, f v/t/n
                idxs = [int(p.split("/")[0]) - 1 for p in parts[1:4]]
                F.append(idxs)
    V_arr = np.array(V, dtype=np.float64)
    F_arr = np.array(F, dtype=np.int64)
    boundary = np.zeros(len(V), dtype=bool)
    return Mesh(V=V_arr, F=F_arr, boundary=boundary)


# ---------------------------------------------------------------------------
# STL (ASCII)
# ---------------------------------------------------------------------------


def write_stl(mesh: Mesh, path: str | Path, binary: bool = True) -> None:
    """Write an STL file (binary by default, ASCII if binary=False).

    Binary STL is far more compact and the standard for slicers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normals = mesh.face_normals()
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normals = normals / norms

    if binary:
        with open(path, "wb") as fh:
            fh.write(b"\x00" * 80)  # 80-byte header
            fh.write(struct.pack("<I", mesh.n_faces()))
            for i, face in enumerate(mesh.F):
                n = normals[i].astype(np.float32)
                fh.write(struct.pack("<fff", *n))
                for vi in face:
                    v = mesh.V[vi].astype(np.float32)
                    fh.write(struct.pack("<fff", *v))
                fh.write(struct.pack("<H", 0))  # attribute byte count
    else:
        with open(path, "w") as fh:
            fh.write("solid minsurf\n")
            for i, face in enumerate(mesh.F):
                n = normals[i]
                fh.write(f"  facet normal {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n")
                fh.write("    outer loop\n")
                for vi in face:
                    v = mesh.V[vi]
                    fh.write(f"      vertex {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
                fh.write("    endloop\n")
                fh.write("  endfacet\n")
            fh.write("endsolid minsurf\n")


def read_stl(path: str | Path) -> Mesh:
    """Read a binary or ASCII STL file.

    Note: STL has no vertex sharing, so the returned mesh has duplicate
    vertices; call read_stl_welded for a welded mesh.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        fh.read(80)
        count_bytes = fh.read(4)
        if len(count_bytes) < 4:
            # ASCII fallback
            return _read_stl_ascii(path)
        n_faces = struct.unpack("<I", count_bytes)[0]
        # Sanity check: binary STL size = 80 + 4 + n_faces*50
        expected = 80 + 4 + n_faces * 50
        actual = path.stat().st_size
        if abs(actual - expected) > 4:
            return _read_stl_ascii(path)
        V = []
        F = []
        for i in range(n_faces):
            fh.read(12)  # skip normal
            for _k in range(3):
                xyz = struct.unpack("<fff", fh.read(12))
                V.append(list(xyz))
            F.append([3 * i, 3 * i + 1, 3 * i + 2])
            fh.read(2)  # attribute
    V_arr = np.array(V, dtype=np.float64)
    F_arr = np.array(F, dtype=np.int64)
    boundary = np.zeros(len(V_arr), dtype=bool)
    return Mesh(V=V_arr, F=F_arr, boundary=boundary)


def _read_stl_ascii(path: Path) -> Mesh:
    V: list[list[float]] = []
    F: list[list[int]] = []
    with open(path) as fh:
        vi = 0
        face_verts: list[int] = []
        for line in fh:
            line = line.strip()
            if line.startswith("vertex"):
                parts = line.split()
                V.append([float(parts[1]), float(parts[2]), float(parts[3])])
                face_verts.append(vi)
                vi += 1
                if len(face_verts) == 3:
                    F.append(face_verts)
                    face_verts = []
    V_arr = np.array(V, dtype=np.float64)
    F_arr = np.array(F, dtype=np.int64)
    boundary = np.zeros(len(V_arr), dtype=bool)
    return Mesh(V=V_arr, F=F_arr, boundary=boundary)


# ---------------------------------------------------------------------------
# PLY (ASCII)
# ---------------------------------------------------------------------------


def write_ply(mesh: Mesh, path: str | Path) -> None:
    """Write an ASCII PLY file.

    Parameters
    ----------
    mesh : Mesh to export.
    path : output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_v = mesh.n_vertices()
    n_f = mesh.n_faces()
    with open(path, "w") as fh:
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write("comment minsurf PLY export\n")
        fh.write(f"element vertex {n_v}\n")
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write(f"element face {n_f}\n")
        fh.write("property list uchar int vertex_indices\n")
        fh.write("end_header\n")
        for v in mesh.V:
            fh.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for f in mesh.F:
            fh.write(f"3 {f[0]} {f[1]} {f[2]}\n")


def export(mesh: Mesh, path: str | Path, fmt: str | None = None) -> None:
    """Export mesh to file, inferring format from extension if fmt is None.

    Supported formats: obj, stl, ply.
    """
    path = Path(path)
    if fmt is None:
        fmt = path.suffix.lstrip(".").lower()
    fmt = fmt.lower()
    if fmt == "obj":
        write_obj(mesh, path)
    elif fmt == "stl":
        write_stl(mesh, path)
    elif fmt == "ply":
        write_ply(mesh, path)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use obj, stl, or ply.")
