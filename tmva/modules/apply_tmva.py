#!/usr/bin/env python3
import argparse
import os
from array import array
import ROOT

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def find_weights_xml(weights_root: str, method: str) -> str:
    base = os.path.dirname(os.path.abspath(weights_root))
    name = f"TMVAClassification_{method}.weights.xml"
    candidates = [
        os.path.join(base, "dataset", "weights", name),
        os.path.join(os.getcwd(), "dataset", "weights", name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    msg = ["Missing weights XML. Tried:"]
    msg += [f"  - {c}" for c in candidates]
    msg += ["", f"Tip: find . -type f -name '{name}'"]
    raise FileNotFoundError("\n".join(msg))

def in_path_for(tag, sample, in_dir, syst):
    # Nominal keeps old naming (backward compatible)
    if syst == "Nominal":
        return os.path.join(in_dir, f"{tag}_{sample}_cutcount.root")
    return os.path.join(in_dir, f"{tag}_{sample}_cutcount_{syst}.root")

def out_path_for(tag, sample, out_dir, syst):
    if syst == "Nominal":
        return os.path.join(out_dir, f"{tag}_{sample}_withBDT.root")
    return os.path.join(out_dir, f"{tag}_{sample}_withBDT_{syst}.root")

def main():
    ap = argparse.ArgumentParser(description="Apply TMVA weights to samples and write *_withBDT*.root")
    ap.add_argument("--in-dir", required=True, help="Input directory containing *_cutcount*.root")
    ap.add_argument("--out-dir", required=True, help="Output directory for *_withBDT*.root")
    ap.add_argument("--tag", required=True, help="e.g. monoHiggs")
    ap.add_argument("--tree", default="events")
    ap.add_argument("--method", required=True, choices=["BDT", "BDTG"])
    ap.add_argument("--weights-root", required=True, help="TMVA ROOT file: TMVA_${METHOD}.root")
    ap.add_argument("--weights-xml", default=None, help="(Optional) path to TMVA weights XML")
    ap.add_argument("--bdt-branch", default=None, help="Output branch name (default bdt_${METHOD})")
    ap.add_argument("--batch", action="store_true", help="Batch mode")

    # NEW:
    ap.add_argument(
        "--syst",
        default="Nominal",
        choices=["Nominal", "JESUp", "JESDown", "JERUp", "JERDown"],
        help="Which systematic file suffix to read/write (default: Nominal)",
    )

    args = ap.parse_args()

    if args.batch:
        ROOT.gROOT.SetBatch(True)

    ROOT.TMVA.Tools.Instance()

    in_dir = os.path.expanduser(args.in_dir)
    out_dir = os.path.expanduser(args.out_dir)
    ensure_dir(out_dir)

    weights_root = os.path.expanduser(args.weights_root)
    if not os.path.exists(weights_root):
        raise FileNotFoundError(f"Missing TMVA root file: {weights_root}")

    if args.weights_xml:
        xml = os.path.expanduser(args.weights_xml)
        if not os.path.exists(xml):
            raise FileNotFoundError(f"--weights-xml provided but not found: {xml}")
    else:
        xml = find_weights_xml(weights_root, args.method)

    print(f"[INFO] Using weights XML: {xml}")
    print(f"[INFO] Systematic: {args.syst}")

    out_branch = args.bdt_branch or f"bdt_{args.method}"

    # Variables must match training exactly
    var_names = [
        "MET","HT","mbb","ptbb",
        "dr_bb","dphi_bb","dphi_bb_met",
        "bjet1_pt","dphi_bjet1_met",
        "recoil","balance","ptbb_minus_met"
    ]
    spectator_names = ["Njets", "Nbjets", "Nlep"]

    # Build TMVA reader
    reader = ROOT.TMVA.Reader("!Color:!Silent")

    vars_dict = {}
    for v in var_names:
        vars_dict[v] = array('f', [0.0])
        reader.AddVariable(v, vars_dict[v])

    specs_dict = {}
    for s in spectator_names:
        specs_dict[s] = array('f', [0.0])
        reader.AddSpectator(s, specs_dict[s])

    # Book method (name must match training BookMethod name)
    reader.BookMVA(f"{args.method} method", xml)

    # Samples list
    env_samples = os.environ.get("APPLY_SAMPLES", "").strip()
    if env_samples:
        samples = env_samples.split()
    else:
        samples = [
            "signal",
            "tt_semilep","tt_dilep","tt_had",
            "st_tW","st_tch_top","st_tch_tbar",
            "wjets","zvvjets"
        ]

    for s in samples:
        in_path  = in_path_for(args.tag, s, in_dir, args.syst)
        out_path = out_path_for(args.tag, s, out_dir, args.syst)

        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Missing input: {in_path}")

        fin = ROOT.TFile.Open(in_path)
        if not fin or fin.IsZombie():
            raise RuntimeError(f"Cannot open: {in_path}")

        tin = fin.Get(args.tree)
        if not tin:
            raise RuntimeError(f"Tree '{args.tree}' not found in {in_path}")

        fout = ROOT.TFile.Open(out_path, "RECREATE")
        tout = tin.CloneTree(0)

        bdt_val = array('f', [0.0])
        tout.Branch(out_branch, bdt_val, f"{out_branch}/F")

        n = tin.GetEntries()
        print(f"Processing {os.path.basename(in_path)} ({n} events)")

        for i in range(n):
            tin.GetEntry(i)

            for v in var_names:
                vars_dict[v][0] = float(getattr(tin, v))
            for sp in spectator_names:
                specs_dict[sp][0] = float(getattr(tin, sp))

            bdt_val[0] = float(reader.EvaluateMVA(f"{args.method} method"))
            tout.Fill()

        fout.Write()
        fout.Close()
        fin.Close()
        print(f"  -> wrote {out_path}")

    print(f"[OK] wrote *_withBDT*.root to: {out_dir}")

if __name__ == "__main__":
    main()
