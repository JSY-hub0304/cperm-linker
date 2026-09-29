"""Cyclic-symmetry geometry utilities for coaxial antigen/scaffold docking.

Implements the geometric half of the linker-design step in Haas & Jasti et al.,
PNAS 2025 (10.1073/pnas.2409566122): place a C3 antigen above a C3 scaffold on
their shared threefold axis, then sample the translational (r) and rotational
(omega) degrees of freedom along that axis.
"""

from __future__ import annotations

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
from Bio.PDB.Structure import Structure


# --------------------------------------------------------------------------
# structure loading
# --------------------------------------------------------------------------

CHAIN_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def load(path: str, model_id: int = 0) -> Structure:
    """Load one model.

    RCSB biological-assembly files (.pdb1) store each symmetry copy as a
    separate MODEL sharing one chain ID. When that is what we are given, merge
    the models into a single model with distinct chain IDs, so a C3 trimer
    looks the same whether it came from the asymmetric unit or an assembly.
    """
    parser = (MMCIFParser(QUIET=True) if str(path).lower().endswith((".cif", ".mmcif"))
              else PDBParser(QUIET=True))
    struct = parser.get_structure("s", path)
    models = list(struct)
    if len(models) == 1:
        return struct[model_id]

    merged = models[0].copy()
    used = {c.id for c in merged}
    for extra in models[1:]:
        for chain in extra:
            ch = chain.copy()
            if ch.id in used:
                ch.id = next(x for x in CHAIN_POOL if x not in used)
            used.add(ch.id)
            ch.detach_parent()
            merged.add(ch)
    return merged


def ca_table(chain) -> dict[int, np.ndarray]:
    """resseq -> CA coordinate, standard amino acids only (no HETATM/water)."""
    out = {}
    for res in chain:
        if res.id[0] != " ":
            continue
        if "CA" in res:
            out[res.id[1]] = res["CA"].get_coord().astype(float)
    return out


def matched_ca(model, chain_ids: list[str]) -> list[np.ndarray]:
    """CA arrays for each chain, restricted to residue numbers common to all."""
    tables = [ca_table(model[c]) for c in chain_ids]
    common = sorted(set.intersection(*(set(t) for t in tables)))
    if len(common) < 10:
        raise ValueError(
            f"only {len(common)} residue numbers shared across chains {chain_ids}; "
            "chains may not be identical copies"
        )
    return [np.array([t[r] for r in common]) for t in tables]


# --------------------------------------------------------------------------
# symmetry axis
# --------------------------------------------------------------------------

def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t with Q ~= P @ R.T + t."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc


def cyclic_axis(model, chain_ids: list[str], ref_point: np.ndarray | None = None) -> dict:
    """Symmetry axis of a homo-oligomer from the chain-to-chain rotation.

    Returns axis (unit vector), center (point on the axis), and the rotation
    angle in degrees, which should be close to 360/n for a clean Cn.

    `ref_point` fixes two things that are otherwise arbitrary: where along the
    axis `center` sits, and which of the two axis directions is positive. The
    axis is oriented to point from `ref_point` toward the oligomer. For a
    trimer cut out of a nanoparticle, pass the centre of the whole assembly and
    the axis then points out of the particle. Defaults to the coordinate
    origin, which is only the particle centre if the file happens to be
    centred there - deposited structures often are not.
    """
    cas = matched_ca(model, chain_ids)
    R, t = kabsch(cas[0], cas[1])

    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos))

    # axis = real eigenvector of R with eigenvalue 1
    vals, vecs = np.linalg.eig(R)
    axis = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    axis /= np.linalg.norm(axis)

    # fixed point: (I - R) c = t, singular along the axis -> least squares
    center = np.linalg.lstsq(np.eye(3) - R, t, rcond=None)[0]

    ref = np.zeros(3) if ref_point is None else np.asarray(ref_point, dtype=float)
    # slide `center` along the axis to the point closest to the reference
    center = center + axis * float(np.dot(ref - center, axis))

    # orient the axis so it points from the reference toward the oligomer
    centroids = np.array([c.mean(0) for c in cas])
    if np.dot(axis, centroids.mean(0) - ref) < 0:
        axis = -axis

    rmsd_expected = 360.0 / len(chain_ids)
    return {
        "axis": axis,
        "center": center,
        "angle_deg": angle,
        "expected_angle_deg": rmsd_expected,
        "angle_error_deg": abs(angle - rmsd_expected),
    }


def rotation_to_z(axis: np.ndarray) -> np.ndarray:
    """Rotation matrix taking `axis` onto +z."""
    a = axis / np.linalg.norm(axis)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, z)
    s, c = np.linalg.norm(v), float(np.dot(a, z))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s**2)


def rz(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def apply(model, R: np.ndarray, t: np.ndarray) -> None:
    """In-place rigid transform of every atom (Bio.PDB wants row-vector convention)."""
    model.transform(R.T, t)


def canonicalize(model, chain_ids: list[str], flip: bool = False,
                 ref_point: np.ndarray | None = None) -> dict:
    """Move an oligomer so its symmetry axis is +z and its center is the origin.

    With `ref_point` set to the centre of a parent assembly, +z comes out as
    the direction pointing out of that assembly.
    """
    info = cyclic_axis(model, chain_ids, ref_point=ref_point)
    R = rotation_to_z(info["axis"])
    apply(model, R, -R @ info["center"])
    if flip:
        apply(model, np.diag([1.0, -1.0, -1.0]), np.zeros(3))
    return info


# --------------------------------------------------------------------------
# coordinates / clash
# --------------------------------------------------------------------------

def heavy_coords(model, chain_ids: list[str] | None = None) -> np.ndarray:
    sel = chain_ids or [c.id for c in model]
    return np.array(
        [
            a.get_coord()
            for c in sel
            for res in model[c]
            if res.id[0] == " "
            for a in res
            if a.element != "H"
        ],
        dtype=float,
    )


def z_extent(coords: np.ndarray) -> tuple[float, float]:
    return float(coords[:, 2].min()), float(coords[:, 2].max())


try:
    from scipy.spatial import cKDTree as _KDTree
except ImportError:  # pragma: no cover - fallback keeps the module usable
    _KDTree = None


def min_pair_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Exact closest heavy-atom distance between two point sets."""
    if len(A) == 0 or len(B) == 0:
        return float("inf")
    if _KDTree is not None:
        return float(_KDTree(A).query(B, k=1)[0].min())
    best = np.inf
    for i in range(0, len(B), 2048):
        d = np.linalg.norm(A[None, :, :] - B[i:i + 2048, None, :], axis=2)
        best = min(best, float(d.min()))
    return best


def contact_count(A: np.ndarray, B: np.ndarray, cutoff: float = 8.0) -> int:
    """Heavy-atom pairs within `cutoff` — a cheap proxy for interface size."""
    if len(A) == 0 or len(B) == 0:
        return 0
    if _KDTree is not None:
        return int(sum(len(x) for x in _KDTree(A).query_ball_point(B, r=cutoff)))
    n = 0
    for i in range(0, len(B), 2048):
        d = np.linalg.norm(A[None, :, :] - B[i:i + 2048, None, :], axis=2)
        n += int((d <= cutoff).sum())
    return n


# --------------------------------------------------------------------------
# terminus helpers
# --------------------------------------------------------------------------

def terminus(model, chain_id: str, which: str) -> tuple[int, np.ndarray]:
    """(resseq, CA coord) of the N- or C-terminal modelled residue of a chain."""
    t = ca_table(model[chain_id])
    if not t:
        raise ValueError(f"chain {chain_id} has no CA atoms")
    r = min(t) if which.upper() == "N" else max(t)
    return r, t[r]


class HeavySelect(Select):
    def accept_residue(self, res):
        return res.id[0] == " "

    def accept_atom(self, atom):
        return atom.element != "H" and atom.get_altloc() in (" ", "A")


def write_pdb(model, path: str) -> None:
    io = PDBIO()
    io.set_structure(model)
    io.save(str(path), select=HeavySelect())
