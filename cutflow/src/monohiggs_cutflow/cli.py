#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from typing import Optional, Tuple, List

import awkward as ak
import numpy as np
import ROOT
import uproot

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)

try:
    from .definitions import define_cutflow_hists, define_histograms, define_tree
    from .analysis import CRTopConfig, ObjectCuts, SRConfig, SystConfig, select_event
    from .physics import passes_minmax
except ImportError:
    from definitions import define_cutflow_hists, define_histograms, define_tree
    from analysis import CRTopConfig, ObjectCuts, SRConfig, SystConfig, select_event
    from physics import passes_minmax


# ============================================================
# Small helpers
# ============================================================
def to_scalar(x, default: float = 0.0) -> float:
    """
    Convert scalar-like awkward / numpy objects to a Python float safely.
    """
    arr = ak.to_numpy(x)
    if np.isscalar(arr):
        return float(arr)
    arr = np.asarray(arr)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


# ============================================================
# I/O helper merged from io.py
# ============================================================
def iter_events(
    input_path: str,
    step_size: int = 50000,
    max_events: int = -1,
    extra_branches=None,
):
    """
    Iterate over Delphes events in chunks.

    Required output keys:
      start, ele_pt, mu_pt, met, met_phi, jet_pt, jet_eta, jet_phi, jet_mass, jet_btag

    Optional:
      any extra scalar branches requested in extra_branches
    """
    if extra_branches is None:
        extra_branches = []

    with uproot.open(input_path) as f:
        tree = f["Delphes"]

        branches = [
            "Electron.PT",
            "Muon.PT",
            "MissingET.MET",
            "MissingET.Phi",
            "Jet.PT",
            "Jet.Eta",
            "Jet.Phi",
            "Jet.Mass",
            "Jet.BTag",
        ]

        all_keys = set(tree.keys(recursive=True, full_paths=True))
        for b in extra_branches:
            if b in all_keys and b not in branches:
                branches.append(b)

        entry_stop = None
        if max_events is not None and max_events > 0:
            entry_stop = int(max_events)

        start = 0
        for arr in tree.iterate(
            expressions=branches,
            step_size=step_size,
            entry_stop=entry_stop,
            library="ak",
        ):
            out = {
                "start": start,
                "ele_pt": arr["Electron.PT"],
                "mu_pt": arr["Muon.PT"],
                "met": arr["MissingET.MET"],
                "met_phi": arr["MissingET.Phi"],
                "jet_pt": arr["Jet.PT"],
                "jet_eta": arr["Jet.Eta"],
                "jet_phi": arr["Jet.Phi"],
                "jet_mass": arr["Jet.Mass"],
                "jet_btag": arr["Jet.BTag"],
            }

            for b in extra_branches:
                if b in arr.fields:
                    out[b] = arr[b]

            n_chunk = len(out["met"])
            yield out
            start += n_chunk


# ============================================================
# Weight helpers
# ============================================================
def _first_existing_branch(
    tree: uproot.TTree, candidates: List[str]
) -> Optional[str]:
    keys = set(tree.keys(recursive=True, full_paths=True))

    for c in candidates:
        if c in keys:
            return c

    lower_map = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


def debug_weight_branches(tree: uproot.TTree) -> None:
    keys = tree.keys(recursive=True, full_paths=True)
    print("[DEBUG] Branches containing 'Weight':")
    found = False
    for k in keys:
        if "weight" in k.lower():
            print("   ", k)
            found = True
    if not found:
        print("    <none>")


def find_event_weight_branch(tree: uproot.TTree) -> Optional[str]:
    """
    Prefer real scalar/leaf weight branches over container branches.
    Use full recursive uproot paths.
    """
    candidates = [
        "Weight/Weight.Weight",
        "Event/Event.Weight",
        "GenEvent/GenEvent.Weight",
        "Generator/Generator.Weight",
        "LHEFWeight/LHEFWeight.Weight",
    ]
    return _first_existing_branch(tree, candidates)


def _branch_is_scalar_numeric(
    tree: uproot.TTree, branch_name: str, step_size: int = 1000
) -> bool:
    """
    Return True only if the branch can be read as a scalar numeric array.
    """
    try:
        for arr in tree.iterate(
            expressions=[branch_name],
            step_size=step_size,
            entry_stop=1,
            library="ak",
        ):
            data = arr[branch_name]
            np_data = ak.to_numpy(data)
            np_data = np.asarray(np_data)
            return np.issubdtype(np_data.dtype, np.number)
    except Exception:
        return False
    return False


def compute_weight_normalization(
    input_path: str,
    max_events: int,
    step_size: int,
    xsec_pb: Optional[float],
    lumi_pb: Optional[float],
) -> Tuple[Optional[str], float, int]:
    """
    Returns:
      (weight_branch_name, global_scale, nentries_used)

    Final per-event weight:
      evt_weight = gen_weight * global_scale

    If no generator-weight branch exists:
      evt_weight = (xsec*lumi/nentries_used)
    """
    with uproot.open(input_path) as f:
        tree = f["Delphes"]

        nentries = int(tree.num_entries)
        if max_events is not None and max_events > 0:
            nentries = min(nentries, int(max_events))

        if nentries <= 0:
            return None, 1.0, 0

        # Hard-priority order for Delphes-like files
        all_keys = set(tree.keys(recursive=True, full_paths=True))
        if "Weight/Weight.Weight" in all_keys:
            wbranch = "Weight/Weight.Weight"
        elif "Event/Event.Weight" in all_keys:
            wbranch = "Event/Event.Weight"
        else:
            wbranch = find_event_weight_branch(tree)

        if wbranch is not None and not _branch_is_scalar_numeric(
            tree, wbranch, step_size=step_size
        ):
            print(
                f"[WARN] Ignoring non-scalar or non-numeric weight branch: {wbranch}"
            )
            wbranch = None

        if xsec_pb is None or lumi_pb is None:
            return wbranch, 1.0, nentries

        target_yield = float(xsec_pb) * float(lumi_pb)

        if wbranch is None:
            scale = target_yield / float(nentries)
            return None, scale, nentries

        sumw = 0.0
        seen = 0
        for arr in tree.iterate(
            expressions=[wbranch],
            entry_stop=nentries,
            step_size=step_size,
            library="ak",
        ):
            w = arr[wbranch]
            w_np = ak.to_numpy(w)
            sumw += float(np.sum(w_np))
            seen += len(w_np)

        if seen == 0 or abs(sumw) < 1e-15:
            scale = target_yield / float(nentries)
            return None, scale, nentries

        scale = target_yield / sumw
        return wbranch, scale, nentries


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Mono-Higgs (H->bb)+MET cutflow + diagnostics using uproot (no Delphes libs)."
    )
    ap.add_argument(
        "input",
        nargs="?",
        help="Input Delphes ROOT file (TTree=Delphes). Not needed for --report.",
    )
    ap.add_argument(
        "-o", "--output", default="monoHiggs_cutcount.root", help="Output ROOT file"
    )

    ap.add_argument("--xsec", type=float, default=None, help="Cross section in pb (optional)")
    ap.add_argument("--lumi", type=float, default=None, help="Integrated luminosity in /pb (optional)")

    ap.add_argument(
        "--syst",
        default="Nominal",
        choices=["Nominal", "JESUp", "JESDown", "JERUp", "JERDown"],
    )
    ap.add_argument("--jesFrac", type=float, default=0.02)
    ap.add_argument("--jerSigma", type=float, default=0.10)
    ap.add_argument("--jerUncFrac", type=float, default=0.10)
    ap.add_argument("--jerSeed", type=int, default=12345)

    ap.add_argument("--jetPtCut", type=float, default=20.0)
    ap.add_argument("--jetEtaCut", type=float, default=4.0)
    ap.add_argument("--bjetPtCut", type=float, default=20.0)
    ap.add_argument("--bjetEtaCut", type=float, default=2.8)
    ap.add_argument("--targetMbb", type=float, default=125.0)

    ap.add_argument("--minLeptons", type=int, default=None)
    ap.add_argument("--maxLeptons", type=int, default=0)
    ap.add_argument("--noLeptonVeto", action="store_true")

    ap.add_argument("--crMinLeptons", type=int, default=1)
    ap.add_argument("--crMaxLeptons", type=int, default=1)
    ap.add_argument("--crMinMET", type=float, default=80.0)
    ap.add_argument("--crMinJets", type=int, default=4)
    ap.add_argument("--crMaxJets", type=int, default=None)
    ap.add_argument("--crMinBJets", type=int, default=2)
    ap.add_argument("--crMaxBJets", type=int, default=None)
    ap.add_argument("--crMinHT", type=float, default=300.0)
    ap.add_argument("--crUseMbbWindow", action="store_true")
    ap.add_argument("--crInvertMbbWindow", action="store_true")
    ap.add_argument("--crInvertDPhiBBMET", action="store_true")

    ap.add_argument("--maxEvents", type=int, default=-1)
    ap.add_argument("--progress", type=int, default=100000)
    ap.add_argument("--stepSize", type=int, default=50000)

    ap.add_argument("--minMET", type=float, default=None)
    ap.add_argument("--minJets", type=int, default=None)
    ap.add_argument("--maxJets", type=int, default=None)
    ap.add_argument("--minBJets", type=int, default=None)
    ap.add_argument("--maxBJets", type=int, default=None)

    ap.add_argument("--minBalance", type=float, default=None)
    ap.add_argument("--maxBalance", type=float, default=None)

    ap.add_argument("--minDPhiBB", type=float, default=None)
    ap.add_argument("--maxDPhiBB", type=float, default=None)

    ap.add_argument("--minBjet1Pt", type=float, default=None)

    ap.add_argument("--minDPhiBBMET", type=float, default=None)
    ap.add_argument("--maxDPhiBBMET", type=float, default=None)

    ap.add_argument("--minDPhiB1MET", type=float, default=None)
    ap.add_argument("--maxDPhiB1MET", type=float, default=None)

    ap.add_argument("--minDRBB", type=float, default=None)
    ap.add_argument("--maxDRBB", type=float, default=None)

    ap.add_argument("--minMbb", type=float, default=None)
    ap.add_argument("--maxMbb", type=float, default=None)

    ap.add_argument("--minBBPt", type=float, default=None)

    ap.add_argument("--minPtBBMinusMET", type=float, default=None)
    ap.add_argument("--maxPtBBMinusMET", type=float, default=None)

    ap.add_argument("--minRecoil", type=float, default=None)

    ap.add_argument("--minHT", type=float, default=None)
    ap.add_argument("--maxHT", type=float, default=None)

    ap.add_argument(
        "--report",
        action="store_true",
        help="Print available histograms/branches and exit (no input needed).",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.report:
        hist = define_histograms()
        tree, br = define_tree()
        print("[REPORT] Histograms (including *_SR mirrors):")
        for k in sorted(hist.keys()):
            print("  -", k)
        print("\n[REPORT] Tree branches:")
        for k in sorted(br.keys()):
            print("  -", k)
        return

    if not args.input:
        raise SystemExit("You must provide an input Delphes ROOT file, or use --report")

    syst = SystConfig(
        args.syst,
        args.jesFrac,
        args.jerSigma,
        args.jerUncFrac,
        args.jerSeed,
    )
    obj = ObjectCuts(
        args.jetPtCut,
        args.jetEtaCut,
        args.bjetPtCut,
        args.bjetEtaCut,
        args.targetMbb,
    )
    sr = SRConfig(
        no_lepton_veto=args.noLeptonVeto,
        min_leptons=args.minLeptons,
        max_leptons=args.maxLeptons,
        min_met=args.minMET,
        min_jets=args.minJets,
        max_jets=args.maxJets,
        min_bjets=args.minBJets,
        max_bjets=args.maxBJets,
        min_balance=args.minBalance,
        max_balance=args.maxBalance,
        min_dphi_bb=args.minDPhiBB,
        max_dphi_bb=args.maxDPhiBB,
        min_bjet1_pt=args.minBjet1Pt,
        min_dphi_bb_met=args.minDPhiBBMET,
        max_dphi_bb_met=args.maxDPhiBBMET,
        min_dphi_b1_met=args.minDPhiB1MET,
        max_dphi_b1_met=args.maxDPhiB1MET,
        min_dr_bb=args.minDRBB,
        max_dr_bb=args.maxDRBB,
        min_mbb=args.minMbb,
        max_mbb=args.maxMbb,
        min_bb_pt=args.minBBPt,
        min_ptbb_minus_met=args.minPtBBMinusMET,
        max_ptbb_minus_met=args.maxPtBBMinusMET,
        min_recoil=args.minRecoil,
        min_ht=args.minHT,
        max_ht=args.maxHT,
    )
    cr = CRTopConfig(
        args.crMinLeptons,
        args.crMaxLeptons,
        args.crMinMET,
        args.crMinJets,
        args.crMaxJets,
        args.crMinBJets,
        args.crMaxBJets,
        args.crMinHT,
        args.crUseMbbWindow,
        args.crInvertMbbWindow,
        args.crInvertDPhiBBMET,
    )

    weight_branch, global_scale, nentries = compute_weight_normalization(
        input_path=args.input,
        max_events=args.maxEvents,
        step_size=args.stepSize,
        xsec_pb=args.xsec,
        lumi_pb=args.lumi,
    )

    print(f"[INFO] Input entries used for normalization: {nentries}")
    print(f"[INFO] Weight branch: {weight_branch if weight_branch else '<none>'}")
    print(f"[INFO] Global scale: {global_scale:.12g}")

    fout = ROOT.TFile(args.output, "RECREATE")
    hist = define_histograms()
    tree, br = define_tree()

    hNlep = hist["Nlep"]
    hMET = hist["MET"]
    hNjets = hist["Njets"]
    hNbjets = hist["Nbjets"]
    hHT = hist["HT"]

    h_mbb = hist["mbb"]
    h_ptbb = hist["ptbb"]
    h_drbb = hist["dr_bb"]
    h_dphi_bb = hist["dphi_bb"]
    h_dphi_bb_met = hist["dphi_bb_met"]
    h_bjet1_pt = hist["bjet1_pt"]
    h_dphi_b1_met = hist["dphi_bjet1_met"]
    h_recoil = hist["recoil"]
    h_balance = hist["balance"]
    h_ptbb_minus_met = hist["ptbb_minus_met"]

    h_ptbb_vs_MET = hist.get("ptbb_vs_MET")
    h_recoil_vs_HT = hist.get("recoil_vs_HT")

    hNlep_SR = hist["Nlep_SR"]
    hMET_SR = hist["MET_SR"]
    hNjets_SR = hist["Njets_SR"]
    hNbjets_SR = hist["Nbjets_SR"]
    hHT_SR = hist["HT_SR"]

    h_mbb_SR = hist["mbb_SR"]
    h_ptbb_SR = hist["ptbb_SR"]
    h_drbb_SR = hist["dr_bb_SR"]
    h_dphi_bb_SR = hist["dphi_bb_SR"]
    h_dphi_bb_met_SR = hist["dphi_bb_met_SR"]
    h_bjet1_pt_SR = hist["bjet1_pt_SR"]
    h_dphi_b1_met_SR = hist["dphi_bjet1_met_SR"]
    h_recoil_SR = hist["recoil_SR"]
    h_balance_SR = hist["balance_SR"]
    h_ptbb_minus_met_SR = hist["ptbb_minus_met_SR"]

    if args.noLeptonVeto:
        lep_label = "Leptons: none"
    else:
        lo = args.minLeptons if args.minLeptons is not None else 0
        hi = args.maxLeptons if args.maxLeptons is not None else "inf"
        lep_label = f"Nlep[{lo},{hi}]"

    steps = [
        "All",
        lep_label,
        "Objects",
        "MinMET",
        "Njets",
        "Nbjets",
        "Balance",
        "dphi_bb",
        "bjet1_pt",
        "dphi_bb_met",
        "dphi_b1_met",
        "dr_bb",
        "mbb",
        "ptbb",
        "|ptbb-MET|",
        "recoil",
        "HT",
        "Selected",
    ]

    hcf_raw, hcf_wgt = define_cutflow_hists(steps)

    # NLO-safe variance histogram
    hcf_wgt2 = hcf_wgt.Clone("cutflow_wgt2")
    hcf_wgt2.Reset()
    hcf_wgt2.SetTitle("Cutflow weighted squared (variance)")
    step_to_x = {name: (i + 1) - 0.5 for i, name in enumerate(steps)}

    cf_raw = {lab: 0 for lab in steps}
    cf_wgt = {lab: 0.0 for lab in steps}

    def acc(step: str, evt_weight: float) -> None:
        cf_raw[step] += 1
        cf_wgt[step] += evt_weight
        x = step_to_x[step]
        hcf_raw.Fill(x, 1.0)
        hcf_wgt.Fill(x, evt_weight)
        hcf_wgt2.Fill(x, evt_weight * evt_weight)

    rnd = ROOT.TRandom3(int(args.jerSeed))

    if args.noLeptonVeto:
        minL = 0
        maxL = 10**9
        do_lep_check = False
    else:
        minL = 0 if args.minLeptons is None else int(args.minLeptons)
        maxL = 10**9 if args.maxLeptons is None else int(args.maxLeptons)
        do_lep_check = True

    n_tree = 0
    n_sr = 0
    n_cr = 0
    progress = int(args.progress)

    extra_branches = []
    if weight_branch is not None:
        extra_branches.append(weight_branch)

    for chunk in iter_events(
        args.input,
        step_size=int(args.stepSize),
        max_events=int(args.maxEvents),
        extra_branches=extra_branches,
    ):
        start = int(chunk["start"])

        ele_counts = ak.num(chunk["ele_pt"], axis=1)
        mu_counts = ak.num(chunk["mu_pt"], axis=1)

        met_arr = chunk["met"]
        metphi_arr = chunk["met_phi"]

        jet_pt_arr = chunk["jet_pt"]
        jet_eta_arr = chunk["jet_eta"]
        jet_phi_arr = chunk["jet_phi"]
        jet_mass_arr = chunk["jet_mass"]
        jet_btag_arr = chunk["jet_btag"]

        genw_arr = None
        if weight_branch is not None and weight_branch in chunk:
            genw_arr = chunk[weight_branch]

        n_in_chunk = len(met_arr)
        for i in range(n_in_chunk):
            ievt = start + i

            if progress > 0 and (ievt + 1) % progress == 0:
                print(f"Processed {ievt + 1}/{nentries}")

            genw = 1.0
            if genw_arr is not None:
                genw = to_scalar(genw_arr[i], default=1.0)
            evt_weight = genw * global_scale

            acc("All", evt_weight)

            n_ele = int(ele_counts[i])
            n_mu = int(mu_counts[i])
            nlep = n_ele + n_mu
            hNlep.Fill(nlep, evt_weight)

            pass_lep_sr = True
            if do_lep_check:
                pass_lep_sr = (nlep >= minL) and (nlep <= maxL)

            met_nom = to_scalar(met_arr[i], default=0.0)
            metphi_nom = to_scalar(metphi_arr[i], default=0.0)

            jet_pt = ak.to_numpy(jet_pt_arr[i]).astype(np.float32, copy=False).tolist()
            jet_eta = ak.to_numpy(jet_eta_arr[i]).astype(np.float32, copy=False).tolist()
            jet_phi = ak.to_numpy(jet_phi_arr[i]).astype(np.float32, copy=False).tolist()
            jet_mass = ak.to_numpy(jet_mass_arr[i]).astype(np.float32, copy=False).tolist()
            jet_btag = ak.to_numpy(jet_btag_arr[i]).astype(np.int32, copy=False).tolist()

            out = select_event(
                ievt,
                (jet_pt, jet_eta, jet_phi, jet_mass, jet_btag),
                (n_ele, n_mu),
                (met_nom, metphi_nom),
                obj,
                sr,
                cr,
                syst,
                rnd,
            )

            hNjets.Fill(out["Njets"], evt_weight)
            hNbjets.Fill(out["Nbjets"], evt_weight)
            hHT.Fill(out["HT"], evt_weight)
            hMET.Fill(out["MET"], evt_weight)

            if out["Nbjets"] >= 2 and out["mbb"] > -900:
                h_mbb.Fill(out["mbb"], evt_weight)
                h_ptbb.Fill(out["ptbb"], evt_weight)
                h_drbb.Fill(out["dr_bb"], evt_weight)
                h_dphi_bb.Fill(out["dphi_bb"], evt_weight)
                h_dphi_bb_met.Fill(out["dphi_bb_met"], evt_weight)
                h_bjet1_pt.Fill(out["bjet1_pt"], evt_weight)
                h_dphi_b1_met.Fill(out["dphi_bjet1_met"], evt_weight)
                h_recoil.Fill(out["recoil"], evt_weight)
                h_balance.Fill(out["balance"], evt_weight)
                h_ptbb_minus_met.Fill(out["ptbb_minus_met"], evt_weight)

                if h_ptbb_vs_MET:
                    h_ptbb_vs_MET.Fill(out["MET"], out["ptbb"], evt_weight)
                if h_recoil_vs_HT:
                    h_recoil_vs_HT.Fill(out["HT"], out["recoil"], evt_weight)

            passed = True

            if passed and pass_lep_sr:
                acc(lep_label, evt_weight)
            else:
                passed = False

            if passed and (out["Nbjets"] >= 2):
                acc("Objects", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(out["MET"], args.minMET, None):
                acc("MinMET", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(out["Njets"], args.minJets, args.maxJets):
                acc("Njets", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(out["Nbjets"], args.minBJets, args.maxBJets):
                acc("Nbjets", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(out["balance"], args.minBalance, args.maxBalance):
                acc("Balance", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(out["dphi_bb"], args.minDPhiBB, args.maxDPhiBB):
                acc("dphi_bb", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(out["bjet1_pt"], args.minBjet1Pt, None):
                acc("bjet1_pt", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(out["dphi_bb_met"], args.minDPhiBBMET, args.maxDPhiBBMET):
                acc("dphi_bb_met", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(
                out["dphi_bjet1_met"], args.minDPhiB1MET, args.maxDPhiB1MET
            ):
                acc("dphi_b1_met", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(out["dr_bb"], args.minDRBB, args.maxDRBB):
                acc("dr_bb", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(out["mbb"], args.minMbb, args.maxMbb):
                acc("mbb", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(out["ptbb"], args.minBBPt, None):
                acc("ptbb", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(
                out["ptbb_minus_met"], args.minPtBBMinusMET, args.maxPtBBMinusMET
            ):
                acc("|ptbb-MET|", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(out["recoil"], args.minRecoil, None):
                acc("recoil", evt_weight)
            else:
                passed = False

            if passed and passes_minmax(out["HT"], args.minHT, args.maxHT):
                acc("HT", evt_weight)
            else:
                passed = False

            is_sr_local = int(passed)

            if is_sr_local:
                acc("Selected", evt_weight)

                hNlep_SR.Fill(out["Nlep"], evt_weight)
                hMET_SR.Fill(out["MET"], evt_weight)
                hNjets_SR.Fill(out["Njets"], evt_weight)
                hNbjets_SR.Fill(out["Nbjets"], evt_weight)
                hHT_SR.Fill(out["HT"], evt_weight)

                if out["Nbjets"] >= 2 and out["mbb"] > -900:
                    h_mbb_SR.Fill(out["mbb"], evt_weight)
                    h_ptbb_SR.Fill(out["ptbb"], evt_weight)
                    h_drbb_SR.Fill(out["dr_bb"], evt_weight)
                    h_dphi_bb_SR.Fill(out["dphi_bb"], evt_weight)
                    h_dphi_bb_met_SR.Fill(out["dphi_bb_met"], evt_weight)
                    h_bjet1_pt_SR.Fill(out["bjet1_pt"], evt_weight)
                    h_dphi_b1_met_SR.Fill(out["dphi_bjet1_met"], evt_weight)
                    h_recoil_SR.Fill(out["recoil"], evt_weight)
                    h_balance_SR.Fill(out["balance"], evt_weight)
                    h_ptbb_minus_met_SR.Fill(
                        out["ptbb_minus_met"], evt_weight
                    )

            br["weight"][0] = float(evt_weight)
            br["MET"][0] = float(out["MET"])
            br["Nlep"][0] = int(out["Nlep"])
            br["Njets"][0] = int(out["Njets"])
            br["Nbjets"][0] = int(out["Nbjets"])
            br["HT"][0] = float(out["HT"])

            br["mbb"][0] = float(out["mbb"])
            br["ptbb"][0] = float(out["ptbb"])
            br["dr_bb"][0] = float(out["dr_bb"])
            br["dphi_bb"][0] = float(out["dphi_bb"])
            br["dphi_bb_met"][0] = float(out["dphi_bb_met"])
            br["bjet1_pt"][0] = float(out["bjet1_pt"])
            br["dphi_bjet1_met"][0] = float(out["dphi_bjet1_met"])
            br["recoil"][0] = float(out["recoil"])
            br["balance"][0] = float(out["balance"])
            br["ptbb_minus_met"][0] = float(out["ptbb_minus_met"])

            br["isSR"][0] = int(is_sr_local)
            br["isCR_top"][0] = int(out["isCR_top"])

            tree.Fill()
            n_tree += 1

            if is_sr_local == 1:
                n_sr += 1
            if out["isCR_top"] == 1:
                n_cr += 1

    fout.cd()
    for h in hist.values():
        h.Write()
    hcf_raw.Write()
    hcf_wgt.Write()
    hcf_wgt2.Write()
    tree.Write()
    fout.Close()

    csv_path = os.path.splitext(args.output)[0] + "_cutflow.csv"
    with open(csv_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["step", "raw", "weighted"])
        for s in steps:
            w.writerow([s, cf_raw[s], cf_wgt[s]])

    print(f"Wrote: {args.output}")
    print(f"Wrote: {csv_path}")
    print(f"Tree entries: {n_tree}  SR: {n_sr}  CR_top: {n_cr}")


if __name__ == "__main__":
    main()
