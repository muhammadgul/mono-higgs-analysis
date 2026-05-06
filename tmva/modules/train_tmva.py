#!/usr/bin/env python3
import argparse
import os
import ROOT

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Train TMVA classifier and write TMVA_${METHOD}.root")
    ap.add_argument("--in-dir", required=True, help="Folder containing *_cutcount.root files")
    ap.add_argument("--tag", required=True, help="e.g. monoHiggs")
    ap.add_argument("--tree", default="events", help="TTree name")
    ap.add_argument("--method", required=True, choices=["BDT", "BDTG"], help="TMVA method name")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--out", default=None, help="Output ROOT file (default OUT_DIR/TMVA_${METHOD}.root)")
    ap.add_argument("--batch", action="store_true", help="Batch mode (no GUI)")
    ap.add_argument("--gui", action="store_true", help="Open TMVAGui at the end (needs display)")
    ap.add_argument("--sr-cut", default="(isSR==1 && Nlep==0)", help="Training selection cut")
    args = ap.parse_args()

    if args.batch:
        ROOT.gROOT.SetBatch(True)

    ROOT.TMVA.Tools.Instance()

    in_dir = os.path.expanduser(args.in_dir)
    out_dir = os.path.expanduser(args.out_dir)
    ensure_dir(out_dir)

    out_file = args.out
    if out_file is None:
        out_file = f"TMVA_{args.method}.root"
    else:
        out_file = os.path.basename(out_file)  # keep filename only

    # IMPORTANT: TMVA writes dataset/weights relative to current working dir
    # So we cd into out_dir to keep everything in one folder.
    old_cwd = os.getcwd()
    os.chdir(out_dir)

    # Input files
    def fpath(name):
        return os.path.join(in_dir, f"{args.tag}_{name}_cutcount.root")

    sig_path = fpath("signal")
        # Backgrounds from config.sh (env) if provided
    env_bkgs = os.environ.get("BKG_SAMPLES", "").strip()
    if env_bkgs:
        bkg_names = env_bkgs.split()
    else:
        bkg_names = [
            "tt_semilep", "tt_dilep", "tt_had",
            "st_tW", "st_tch_top", "st_tch_tbar",
            "wjets", "zvvjets"
        ]
    bkg_paths = [fpath(n) for n in bkg_names]

    for p in [sig_path] + bkg_paths:
        if not os.path.exists(p):
            os.chdir(old_cwd)
            raise FileNotFoundError(f"Missing input file: {p}")

    # Open files and trees
    fSig = ROOT.TFile.Open(sig_path)
    if not fSig or fSig.IsZombie():
        os.chdir(old_cwd)
        raise RuntimeError(f"Cannot open {sig_path}")
    tSig = fSig.Get(args.tree)
    if not tSig:
        os.chdir(old_cwd)
        raise RuntimeError(f"Tree '{args.tree}' not found in {sig_path}")

    bkg_files = []
    bkg_trees = []
    for p in bkg_paths:
        f = ROOT.TFile.Open(p)
        if not f or f.IsZombie():
            os.chdir(old_cwd)
            raise RuntimeError(f"Cannot open {p}")
        t = f.Get(args.tree)
        if not t:
            os.chdir(old_cwd)
            raise RuntimeError(f"Tree '{args.tree}' not found in {p}")
        bkg_files.append(f)
        bkg_trees.append(t)

    # Output ROOT file (now created in out_dir because we chdir'ed)
    outROOT = ROOT.TFile.Open(out_file, "RECREATE")
    if not outROOT or outROOT.IsZombie():
        os.chdir(old_cwd)
        raise RuntimeError(f"Cannot create output: {os.path.join(out_dir, out_file)}")

    factory = ROOT.TMVA.Factory(
        "TMVAClassification",
        outROOT,
        "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification"
    )

    dataloader = ROOT.TMVA.DataLoader("dataset")

    # Variables
    dataloader.AddVariable("MET", 'F')
    dataloader.AddVariable("HT", 'F')
    dataloader.AddVariable("mbb", 'F')
    dataloader.AddVariable("ptbb", 'F')
    dataloader.AddVariable("dr_bb", 'F')
    dataloader.AddVariable("dphi_bb", 'F')
    dataloader.AddVariable("dphi_bb_met", 'F')
    dataloader.AddVariable("bjet1_pt", 'F')
    dataloader.AddVariable("dphi_bjet1_met", 'F')
    dataloader.AddVariable("recoil", 'F')
    dataloader.AddVariable("balance", 'F')
    dataloader.AddVariable("ptbb_minus_met", 'F')

    # Spectators
    dataloader.AddSpectator("Njets", 'I')
    dataloader.AddSpectator("Nbjets", 'I')
    dataloader.AddSpectator("Nlep", 'I')
    dataloader.AddSpectator("weight", 'F')
    dataloader.AddSpectator("isSR", 'I')
    dataloader.AddSpectator("isCR_top", 'I')

    # Trees
    dataloader.AddSignalTree(tSig, 1.0)
    for t in bkg_trees:
        dataloader.AddBackgroundTree(t, 1.0)

    # Weights
    dataloader.SetSignalWeightExpression("weight")
    dataloader.SetBackgroundWeightExpression("weight")

    # Cuts
    mycut = ROOT.TCut(args.sr_cut)
    dataloader.PrepareTrainingAndTestTree(
        mycut, mycut,
        "SplitMode=Random:NormMode=NumEvents:!V"
    )

    # Book method
    if args.method == "BDTG":
        factory.BookMethod(
            dataloader, ROOT.TMVA.Types.kBDT, "BDTG",
            "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:"
            "UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=2:"
            "NegWeightTreatment=Pray"
        )
    else:
        factory.BookMethod(
            dataloader, ROOT.TMVA.Types.kBDT, "BDT",
            "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:"
            "AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:"
            "SeparationType=GiniIndex:nCuts=20:"
            "NegWeightTreatment=InverseBoostNegWeights"
        )

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    outROOT.Close()

    # Return to old cwd
    os.chdir(old_cwd)

    print(f"[OK] Wrote: {os.path.join(out_dir, out_file)}")
    print(f"[OK] Weights in: {os.path.join(out_dir, 'dataset/weights/')}")

    if args.gui and (not ROOT.gROOT.IsBatch()):
        try:
            ROOT.TMVA.TMVAGui(os.path.join(out_dir, out_file))
        except Exception:
            ROOT.gROOT.ProcessLine(f'.x TMVAGui.C("{os.path.join(out_dir, out_file)}")')

if __name__ == "__main__":
    main()
