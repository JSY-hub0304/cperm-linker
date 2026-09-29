"""Step 7 (prep) - pull one C3 building block out of a larger assembly.

The deposited nanoparticle structures from Haas & Jasti et al. (9CLZ = I3-A6,
9CM1 = I3-D12, 9CM0 = I3-A7) are full 60-subunit icosahedra. Linker design
operates on the trimeric building block, so this finds a mutually-nearest
triple of chains, verifies it really is a C3, and writes it out.

Example
-------
py -3 linker_pipeline/extract_trimer.py ^
    --input linker_pipeline/inputs/9CM0.cif ^
    --out linker_pipeline/inputs/I3-A7_trimer.pdb
"""

from __future__ import annotations

import argparse
import copy
from itertools import combinations

import numpy as np

import geom


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed-chain", help="anchor the trimer on this chain (default: first)")
    p.add_argument("--tol", type=float, default=5.0, help="allowed deviation from 120 deg")
    p.add_argument("--rename", default="ABC", help="chain IDs to write")
    p.add_argument("--canonicalize", action="store_true",
                   help="also put the threefold axis on +z and the axis point at the origin")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    print(f"loading {a.input}")
    model = geom.load(a.input)

    chains = [c.id for c in model if geom.ca_table(c)]
    print(f"  {len(chains)} polymer chains")
    if len(chains) < 3:
        raise SystemExit("need at least 3 chains")

    centroid = {}
    for c in chains:
        t = geom.ca_table(model[c])
        centroid[c] = np.mean(list(t.values()), axis=0)

    seed = a.seed_chain or chains[0]
    if seed not in centroid:
        raise SystemExit(f"chain {seed} not found; available: {chains[:20]}")

    # candidate partners: nearest neighbours of the seed by centroid
    near = sorted((c for c in chains if c != seed),
                  key=lambda c: np.linalg.norm(centroid[c] - centroid[seed]))[:8]

    best = None
    for pair in combinations(near, 2):
        trio = [seed, *pair]
        try:
            info = geom.cyclic_axis(model, trio)
        except ValueError:
            continue
        # a real C3 also has three mutually equal centroid separations
        d = [np.linalg.norm(centroid[x] - centroid[y]) for x, y in combinations(trio, 2)]
        spread = (max(d) - min(d)) / max(d)
        score = info["angle_error_deg"] + 10 * spread
        if info["angle_error_deg"] <= a.tol and (best is None or score < best[0]):
            best = (score, trio, info, float(np.mean(d)))

    if best is None:
        raise SystemExit(
            f"no C3 triple found within {a.tol} deg of 120 around chain {seed}. "
            "Try --seed-chain with a different chain, or raise --tol."
        )

    _, trio, info, mean_sep = best
    print(f"  C3 trimer: chains {trio}")
    print(f"    rotation      {info['angle_deg']:.2f} deg (target 120)")
    print(f"    centroid sep  {mean_sep:.1f} A")

    out_model = copy.deepcopy(model)
    for c in [x.id for x in list(out_model)]:
        if c not in trio:
            out_model.detach_child(c)

    # rename in C3 cycle order so chain A -> B -> C follows the rotation
    tmp = {}
    for old, new in zip(trio, a.rename):
        ch = out_model[old]
        out_model.detach_child(old)
        ch.id = f"_{new}"
        ch.detach_parent()
        tmp[new] = ch
    for new in a.rename:
        ch = tmp[new]
        ch.id = new
        out_model.add(ch)

    if a.canonicalize:
        # the centre of the whole assembly defines which way is "out of the
        # particle"; the coordinate origin generally does not
        assembly_center = np.mean([centroid[c] for c in chains], axis=0)
        geom.canonicalize(out_model, list(a.rename), ref_point=assembly_center)
        print(f"    assembly centre {np.round(assembly_center, 1)}")
        print("    axis aligned to +z (= pointing out of the assembly)")

    res = geom.ca_table(out_model[a.rename[0]])
    print(f"    chain {a.rename[0]}: {len(res)} residues, {min(res)}-{max(res)}")
    geom.write_pdb(out_model, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
