#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
import ROOT

def sum_weights_after_cut(path, tree, weight_branch, cut_expr):
    f = ROOT.TFile.Open(path, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open {path}")
    t = f.Get(tree)
    if not t:
        f.ls()
        raise RuntimeError(f"Tree '{tree}' not found in {path}")

    hname = "htmp_scan"
    h = ROOT.gROOT.FindObject(hname)
    if not h:
        h = ROOT.TH1F(hname, hname, 1, 0, 1)
    h.Reset()

    draw_expr = f"0.5>>{hname}"
    wexpr = f"({weight_branch})*({cut_expr})"
    t.Draw(draw_expr, wexpr, "goff")

    s = float(h.GetSumOfWeights())
    f.Close()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tree", default="events")
    ap.add_argument("--sr-cut", required=True)
    ap.add_argument("--weight", default="weight")
    ap.add_argument("--bdt-branch", default="bdt")
    ap.add_argument("--cut-min", type=float, default=-1.0)
    ap.add_argument("--cut-max", type=float, default=1.0)
    ap.add_argument("--cut-step", type=float, default=0.01)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    ROOT.gROOT.SetBatch(True)

    in_dir = args.workdir
    tag = args.tag

    sig = f"{in_dir}/{tag}_signal_withBDT.root"

    if not os.path.exists(sig):
        raise FileNotFoundError(f"Missing applied signal file: {sig}")

    bkg_paths = []
    for fn in os.listdir(in_dir):
        if fn.startswith(f"{tag}_") and fn.endswith("_withBDT.root") and "signal" not in fn:
            bkg_paths.append(f"{in_dir}/{fn}")

    if not bkg_paths:
        raise RuntimeError("No background *_withBDT.root found")

    cuts = np.arange(args.cut_min, args.cut_max + 1e-12, args.cut_step)

    rows = []
    for c in cuts:
        sel = f"({args.sr_cut}) && ({args.bdt_branch} > {c})"

        S = sum_weights_after_cut(sig, args.tree, args.weight, sel)
        B = sum(sum_weights_after_cut(bp, args.tree, args.weight, sel) for bp in bkg_paths)

        Z = 0.0
        if (S + B) > 0:
            Z = S / np.sqrt(S + B)

        rows.append({"cut": float(c), "S": float(S), "B": float(B), "Z": float(Z)})

    df = pd.DataFrame(rows)
    best = df.iloc[df["Z"].idxmax()]

    df.to_csv(args.out_csv, index=False)

    print("[OK] Saved scan:", args.out_csv)
    print(f"[BEST] cut={best['cut']:.3f}  S={best['S']:.3f}  B={best['B']:.3f}  Z={best['Z']:.3f}")

if __name__ == "__main__":
    main()
