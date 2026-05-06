import argparse, json, os
import pandas as pd
import ROOT

def sum_w(path, tree, wbranch, selection):
    f = ROOT.TFile.Open(path, "READ")
    t = f.Get(tree)
    if not t:
        raise RuntimeError(f"Tree '{tree}' not found in {path}")
    hname = "htmp2"
    ROOT.gROOT.ProcessLine(f"TH1F* {hname} = new TH1F(\"{hname}\",\"{hname}\",1,0,1);")
    h = ROOT.gROOT.FindObject(hname)
    h.Reset()
    t.Draw("0.5>>htmp2", f"({wbranch})*({selection})", "goff")
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
    ap.add_argument("--scan-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    ROOT.gROOT.SetBatch(True)

    df = pd.read_csv(args.scan_csv)
    best = df.iloc[df["Z"].idxmax()]
    cut = float(best["cut"])

    sel = f"({args.sr_cut}) && ({args.bdt_branch} > {cut})"

    sig_path = f"{args.workdir}/{args.tag}_signal_withBDT.root"
    if not os.path.exists(sig_path):
        raise FileNotFoundError(sig_path)

    out = {"best_cut": cut, "selection": sel, "yields": {}}

    out["yields"]["signal"] = sum_w(sig_path, args.tree, args.weight, sel)

    B = 0.0
    for fn in sorted(os.listdir(args.workdir)):
        if not (fn.startswith(f"{args.tag}_") and fn.endswith("_withBDT.root")):
            continue
        if "signal" in fn:
            continue
        name = fn.replace(f"{args.tag}_", "").replace("_withBDT.root", "")
        yp = sum_w(f"{args.workdir}/{fn}", args.tree, args.weight, sel)
        out["yields"][name] = yp
        B += yp

    S = out["yields"]["signal"]
    out["S"] = S
    out["B"] = B
    out["Z_SoverSplusB"] = float(S / ((S + B) ** 0.5)) if (S + B) > 0 else 0.0

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("[OK] Wrote:", args.out_json)
    print("[SR] cut =", cut, "S =", out["S"], "B =", out["B"], "Z =", out["Z_SoverSplusB"])

if __name__ == "__main__":
    main()
