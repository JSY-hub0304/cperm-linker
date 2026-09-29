"""Step 7b - turn docked poses into RFdiffusion jobs.

Two linker problems from Haas & Jasti et al., PNAS 2025:

  fusion    antigen C-term -> de novo linker -> scaffold N-term.
            The paper diffused 5-80 residues per pose and kept a 60-mer.

  cperm     circular permutation: cut the subunit at a chosen bond, then join
            the two original termini with a short de novo linker (<=4 aa in
            the paper), letting 2 flanking residues on each side move.

Emits a shell script of run_inference.py calls plus a jobs.csv index.

The contig strings assume RFdiffusion's `contigmap.contigs` syntax. Symmetry
flags differ between RFdiffusion releases - check `--symmetry` output against
your install before launching a large batch.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geom


def contig_segment(present: set[int], lo: int, hi: int, chain: str) -> tuple[str, list[tuple[int, int]]]:
    """Contig text for residues lo..hi, diffusing across unmodelled gaps.

    A deposited structure often has disordered loops with no coordinates
    (I3-A7 is missing 123-129). Naming such a range as one fixed segment tells
    RFdiffusion to hold residues that are not there. Instead, each run of
    present residues stays fixed and each gap becomes a diffused span of the
    same length, so the loop gets rebuilt.
    """
    parts: list[str] = []
    gaps: list[tuple[int, int]] = []
    run_start: int | None = None
    r = lo
    while r <= hi:
        if r in present:
            if run_start is None:
                run_start = r
            r += 1
        else:
            if run_start is not None:
                parts.append(f"{chain}{run_start}-{r - 1}")
                run_start = None
            g0 = r
            while r <= hi and r not in present:
                r += 1
            n = r - g0
            parts.append(f"{n}-{n}")
            gaps.append((g0, r - 1))
    if run_start is not None:
        parts.append(f"{chain}{run_start}-{hi}")
    return "/".join(parts), gaps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    f = sub.add_parser("fusion", help="antigen -> linker -> scaffold")
    f.add_argument("--manifest", required=True, help="manifest.csv from make_docks.py")
    f.add_argument("--dock-dir", required=True)
    f.add_argument("--antigen-chain", default="X")
    f.add_argument("--antigen-range", required=True, help="e.g. 1-251")
    f.add_argument("--scaffold-chain", default="A")
    f.add_argument("--scaffold-range", required=True, help="e.g. 2-192")
    f.add_argument("--len-min", type=int, default=5, help="paper: 5")
    f.add_argument("--len-max", type=int, default=80, help="paper: 80")
    f.add_argument("--slack", type=int, default=10,
                   help="widen the geometry-derived length window by +/- this many residues")
    f.add_argument("--num-designs", type=int, default=20)

    c = sub.add_parser("cperm", help="circular permutation linker")
    c.add_argument("--pdb", required=True)
    c.add_argument("--chain", default="A")
    c.add_argument("--first-res", type=int, required=True, help="first modelled residue, e.g. 2")
    c.add_argument("--last-res", type=int, required=True, help="last modelled residue, e.g. 183")
    c.add_argument("--cut-after", type=int, required=True,
                   help="new C-term of the permuted chain; paper cut between 133 and 134 -> 133")
    c.add_argument("--len-min", type=int, default=1)
    c.add_argument("--len-max", type=int, default=4, help="paper: up to 4 aa")
    c.add_argument("--flank", type=int, default=2,
                   help="residues on each side allowed to move during diffusion (paper: 2)")
    c.add_argument("--num-designs", type=int, default=50)

    p.add_argument("--symmetry", default="C3", help="pass 'none' to diffuse a single protomer")
    p.add_argument("--out", required=True)
    return p.parse_args()


def sym_flags(symmetry: str) -> str:
    if symmetry.lower() in ("none", "c1", ""):
        return ""
    return f" inference.symmetry={symmetry}"


def fusion_jobs(a) -> list[dict]:
    rows = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8")) if r["status"] == "ok"]
    if not rows:
        raise SystemExit("no accepted poses in the manifest")

    jobs = []
    for r in rows:
        lo = max(a.len_min, int(r["extended_res"]) - a.slack)
        hi = min(a.len_max, int(r["helix_res"]) + a.slack)
        if hi < lo:
            lo, hi = a.len_min, a.len_max
        contig = (f"[{a.antigen_chain}{a.antigen_range}/{lo}-{hi}/"
                  f"{a.scaffold_chain}{a.scaffold_range}]")
        jobs.append(dict(
            name=r["name"], r=r["r"], omega=r["omega"], span=r["span"],
            len_lo=lo, len_hi=hi, contig=contig,
            input_pdb=str(Path(a.dock_dir) / f"{r['name']}.pdb"),
        ))
    return jobs


def cperm_jobs(a) -> list[dict]:
    """New chain runs cut+1..last, linker, first..cut.

    The flanking residues are dropped from the fixed segments so RFdiffusion is
    free to rebuild them, which is what "the two adjacent amino acids on each
    side were allowed to move" means in practice.
    """
    seg1_lo, seg1_hi = a.cut_after + 1, a.last_res - a.flank
    seg2_lo, seg2_hi = a.first_res + a.flank, a.cut_after
    if seg1_hi <= seg1_lo or seg2_hi <= seg2_lo:
        raise SystemExit("cut position leaves an empty segment; check --cut-after/--flank")

    present = set(geom.ca_table(geom.load(a.pdb)[a.chain]))
    for r in (seg1_lo, seg1_hi, seg2_lo, seg2_hi):
        if r not in present:
            raise SystemExit(
                f"residue {r} has no coordinates in chain {a.chain}; "
                "a segment boundary must be a modelled residue"
            )

    s1, gaps1 = contig_segment(present, seg1_lo, seg1_hi, a.chain)
    s2, gaps2 = contig_segment(present, seg2_lo, seg2_hi, a.chain)
    for lo, hi in gaps1 + gaps2:
        print(f"note: residues {lo}-{hi} are unmodelled and will be rebuilt by diffusion")

    jobs = []
    for n in range(a.len_min, a.len_max + 1):
        total = n + 2 * a.flank  # diffused linker plus the freed flanks
        contig = f"[{s1}/{total}-{total}/{s2}]"
        jobs.append(dict(
            name=f"cperm_link{n:02d}", linker_len=n, freed_flanks=2 * a.flank,
            diffused_total=total,
            rebuilt_gaps=";".join(f"{lo}-{hi}" for lo, hi in gaps1 + gaps2) or "-",
            contig=contig, input_pdb=a.pdb,
        ))
    return jobs


def main() -> None:
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = fusion_jobs(a) if a.mode == "fusion" else cperm_jobs(a)
    flags = sym_flags(a.symmetry)

    lines = [
        "#!/usr/bin/env bash",
        "# generated by make_contigs.py - run from the RFdiffusion repo root",
        "set -euo pipefail",
        'RFD="${RFD:-.}"',
        'OUT="${OUT:-./rfd_out}"',
        'mkdir -p "$OUT"',
        "",
    ]
    for j in jobs:
        lines += [
            f"# {j['name']}",
            f"python \"$RFD/scripts/run_inference.py\" \\",
            f"    inference.input_pdb={j['input_pdb']} \\",
            f"    inference.output_prefix=\"$OUT/{j['name']}\" \\",
            f"    'contigmap.contigs={j['contig']}' \\",
            f"    inference.num_designs={a.num_designs}{flags} \\",
            "    denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0",
            "",
        ]

    script = out / f"run_rfdiffusion_{a.mode}.sh"
    script.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    index = out / "jobs.csv"
    with open(index, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
        w.writeheader()
        w.writerows(jobs)

    print(f"{len(jobs)} RFdiffusion jobs -> {script}")
    print(f"index -> {index}")
    print(f"total backbones at --num-designs={a.num_designs}: {len(jobs) * a.num_designs}")
    print("\nfirst job:")
    print(f"  input  {jobs[0]['input_pdb']}")
    print(f"  contig {jobs[0]['contig']}")
    if flags:
        print(f"  symmetry {a.symmetry}")


if __name__ == "__main__":
    main()
