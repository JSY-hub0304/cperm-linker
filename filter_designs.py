"""Step 7d - filter ProteinMPNN/AF2 outputs the way the paper does.

Reads ColabFold score JSONs (*_scores_rank_*.json) and applies the thresholds
from Haas & Jasti et al., PNAS 2025:

  circular-permutation linkers   pLDDT > 92 over the diffused region, pAE < 5, ipTM > 0.9
  trimer-interface redesign      pLDDT > 90, pAE < 8, ipTM > 0.9
  antigen-fusion linkers         ranked chiefly on AF2 accuracy for the whole trimer

pLDDT is averaged over the linker window only, because a long well-folded
scaffold will otherwise carry a bad linker over any global threshold.

Two ways to say where the linker is:

  --windows CSV   per-design windows, e.g. outputs/linker_windows.csv written by
                  the notebook (columns: tag, length, linker_start, linker_end).
                  Each JSON is matched to a row by the design tag in its filename.
  --linker-start/--linker-end   one window for every design.

When the prediction is a homo-oligomer, pass --copies N. The window is then
scored in all N chains, since ColabFold returns one flat array for the whole
complex (a 179-residue trimer gives 537 values).

Presets:  --preset cperm | interface | fusion
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

PRESETS = {
    "cperm": dict(plddt=92.0, pae=5.0, iptm=0.90),
    "interface": dict(plddt=90.0, pae=8.0, iptm=0.90),
    "fusion": dict(plddt=80.0, pae=10.0, iptm=0.80),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="directory of ColabFold outputs")
    p.add_argument("--preset", choices=sorted(PRESETS), default="fusion")
    p.add_argument("--plddt", type=float, help="override preset")
    p.add_argument("--pae", type=float, help="override preset")
    p.add_argument("--iptm", type=float, help="override preset")

    p.add_argument("--windows", help="CSV of per-design linker windows")
    p.add_argument("--linker-start", type=int, help="1-based first linker residue")
    p.add_argument("--linker-end", type=int, help="1-based last linker residue")
    p.add_argument("--copies", type=int, default=1,
                   help="chains in the predicted complex (3 for a homotrimer)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def design_tag(path: Path) -> str:
    return re.sub(r"_scores_rank_.*$", "", path.stem)


def load_windows(path: str) -> dict[str, dict]:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["tag"]] = dict(
                length=int(r["length"]),
                lo=int(r["linker_start"]),
                hi=int(r["linker_end"]),
            )
    return out


def window_indices(lo: int, hi: int, length: int, copies: int, total: int) -> np.ndarray:
    """0-based indices of the window in every chain of the flattened array."""
    idx = [c * length + i - 1 for c in range(copies) for i in range(lo, hi + 1)]
    return np.array([i for i in idx if 0 <= i < total], dtype=int)


def summarize(d: dict, lo: int | None, hi: int | None,
              length: int | None, copies: int) -> dict:
    plddt = np.asarray(d["plddt"], dtype=float)
    n = len(plddt)
    L = length or n

    if lo and hi:
        idx = window_indices(lo, hi, L, copies, n)
        region = f"{lo}-{hi}" + (f" x{copies}" if copies > 1 else "")
    else:
        idx = np.arange(n)
        region = f"1-{n}"

    win = plddt[idx]
    row = dict(
        n_res=n,
        chains=copies,
        region=region,
        n_scored=len(idx),
        plddt_region=round(float(win.mean()), 2) if len(win) else float("nan"),
        plddt_min=round(float(win.min()), 2) if len(win) else float("nan"),
        plddt_global=round(float(plddt.mean()), 2),
        iptm=round(float(d.get("iptm", float("nan"))), 4),
        ptm=round(float(d.get("ptm", float("nan"))), 4),
    )

    if "pae" in d:
        pae = np.asarray(d["pae"], dtype=float)
        row["pae_mean"] = round(float(pae.mean()), 2)
        row["pae_max"] = round(float(pae.max()), 2)
        if len(idx) and len(idx) < n:
            rest = np.setdiff1d(np.arange(n), idx)
            # how confidently the linker is placed relative to everything else
            cross = pae[np.ix_(idx, rest)]
            row["pae_linker_vs_rest"] = round(float(cross.mean()), 2)
    else:
        row["pae_mean"] = row["pae_max"] = float("nan")
    return row


def main() -> None:
    a = parse_args()
    thr = dict(PRESETS[a.preset])
    for k in ("plddt", "pae", "iptm"):
        if getattr(a, k) is not None:
            thr[k] = getattr(a, k)

    windows = load_windows(a.windows) if a.windows else {}

    files = sorted(Path(a.results).rglob("*_scores_rank_*.json"))
    if not files:
        raise SystemExit(f"no *_scores_rank_*.json under {a.results}")

    rows, unmatched = [], []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "plddt" not in d:
            continue

        tag = design_tag(f)
        if windows:
            w = windows.get(tag)
            if w is None:
                unmatched.append(tag)
                continue
            lo, hi, length = w["lo"], w["hi"], w["length"]
        else:
            lo, hi, length = a.linker_start, a.linker_end, None

        row = dict(design=tag, file=f.name)
        row.update(summarize(d, lo, hi, length, a.copies))

        pae_metric = row.get("pae_linker_vs_rest")
        if pae_metric is None or pae_metric != pae_metric:
            pae_metric = row["pae_mean"]

        checks = {
            "plddt": row["plddt_region"] >= thr["plddt"],
            "pae": pae_metric <= thr["pae"],
            "iptm": (row["iptm"] != row["iptm"]) or row["iptm"] >= thr["iptm"],
        }
        row["pae_used"] = pae_metric
        row["fails"] = ",".join(k for k, ok in checks.items() if not ok) or "-"
        row["pass"] = all(checks.values())
        rows.append(row)

    if unmatched:
        print(f"warning: {len(unmatched)} designs had no row in --windows, skipped "
              f"(e.g. {unmatched[:3]})")
    if not rows:
        raise SystemExit("found score files but none could be scored")

    rows.sort(key=lambda r: (not r["pass"], -r["plddt_region"], r["pae_used"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_pass = sum(r["pass"] for r in rows)
    print(f"preset {a.preset}: pLDDT>={thr['plddt']}  pAE<={thr['pae']}  ipTM>={thr['iptm']}")
    print(f"{n_pass}/{len(rows)} designs pass -> {out}\n")

    print(f"{'design':<24}{'pLDDT_lnk':>10}{'pLDDT_min':>10}{'pAE_lnk':>9}"
          f"{'ipTM':>8}{'pLDDT_all':>10}  fails")
    for r in rows[:25]:
        print(f"{r['design']:<24}{r['plddt_region']:>10.2f}{r['plddt_min']:>10.2f}"
              f"{r['pae_used']:>9.2f}{r['iptm']:>8.3f}{r['plddt_global']:>10.2f}  {r['fails']}")

    if not n_pass:
        from collections import Counter
        print("\nnothing passed; most common failures:",
              dict(Counter(r["fails"] for r in rows)))


if __name__ == "__main__":
    main()
