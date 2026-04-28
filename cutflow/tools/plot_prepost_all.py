#!/usr/bin/env python3
"""
plot_prepost_all.py

Stacked PRE (before full selection) vs SR (after full selection) plots:
 - Backgrounds stacked
 - Signal overlaid as a red line (scaled by --sigScale)
 - Two pads: PRE and SR
 - CMS-like axis fonts/sizes + "CMS Simulation" header

Histogram naming in *_cutcount.root files (your current output):
  PRE: <var>
  SR : <var>_SR

This script is designed to be called by run_all_cutcount_clean.sh and to accept
wrapper-like arguments safely.

Tested for ROOT 6.36 + Python 3.14 (PyROOT).
"""
import os
import re
import argparse
from typing import Dict, List, Optional, Tuple

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)

# ----------------------------
# Helpers
# ----------------------------
def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else []

def discover_samples(outdir: str, tag: str) -> Dict[str, str]:
    """Find <tag>_<sample>_cutcount.root in outdir."""
    out = {}
    for fn in os.listdir(outdir):
        if fn.startswith(tag + "_") and fn.endswith("_cutcount.root"):
            key = fn[len(tag) + 1 : -len("_cutcount.root")]
            out[key] = os.path.join(outdir, fn)
    return out

def is_signal_key(k: str) -> bool:
    return k.lower() in ("signal", "sig")

def _clone(h: ROOT.TH1, newname: str) -> ROOT.TH1:
    hc = h.Clone(newname)
    hc.SetDirectory(0)
    return hc

def try_get_hist(f: ROOT.TFile, name: str) -> Optional[ROOT.TH1]:
    h = f.Get(name)
    if not h:
        return None
    if not isinstance(h, ROOT.TH1):
        return None
    return _clone(h, name + "__clone")

def get_region_hist(f: ROOT.TFile, var: str, region: str) -> Optional[ROOT.TH1]:
    """
    Your files store:
      PRE: var
      SR : var_SR
    """
    if region.lower() == "pre":
        return try_get_hist(f, var)
    else:
        return try_get_hist(f, f"{var}_SR")

def normalize(h: ROOT.TH1) -> None:
    integ = h.Integral()
    if integ > 0:
        h.Scale(1.0 / integ)

def set_axis_style(h: ROOT.TH1) -> None:
    h.SetTitle("")
    h.SetStats(0)
    ax = h.GetXaxis()
    ay = h.GetYaxis()
    ax.SetTitleFont(72); ay.SetTitleFont(72)
    ax.SetLabelFont(72); ay.SetLabelFont(72)
    ax.SetTitleSize(0.050); ay.SetTitleSize(0.050)
    ax.SetLabelSize(0.040); ay.SetLabelSize(0.040)
    ax.SetTitleOffset(1.0); ay.SetTitleOffset(1.15)
    ay.SetNdivisions(508)

def cms_header(pad: ROOT.TPad, right_text: str) -> ROOT.TLatex:
    pad.cd()
    lat = ROOT.TLatex()
    lat.SetNDC(True)
    lat.SetTextFont(72)
    lat.SetTextSize(0.040)
    lat.DrawLatex(0.150, 0.9, "CMS Simulation @ 13 TeV, mono-H (gg#rightarrow h + #chi#bar{#chi})")
    lat.SetTextAlign(31)
    lat.DrawLatex(0.95, 0.9, right_text)
    return lat

def make_canvas() -> ROOT.TCanvas:
    c = ROOT.TCanvas("c_prepost", "c_prepost", 1200, 520)
    c.Divide(2, 1, 0.01, 0.01)
    for i in (1, 2):
        p = c.cd(i)
        p.SetTopMargin(0.12)
        p.SetRightMargin(0.05)
        p.SetLeftMargin(0.14)
        p.SetBottomMargin(0.14)
        p.SetTickx(1); p.SetTicky(1)
    return c

def list_variables_from_signal(path: str) -> List[str]:
    """
    Infer variables from keys in the signal file.
    Keep only 1D TH1 variables that have both PRE (var) and SR (var_SR).
    Exclude cutflow and 2D histograms.
    """
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        return []
    keys = [k.GetName() for k in f.GetListOfKeys()]
    f.Close()

    # Build set of PRE vars and SR vars
    pre = set()
    sr  = set()
    for n in keys:
        if n in ("cutflow_raw", "cutflow_wgt", "cutflow"):
            continue
        # Exclude 2D and trees
        if n.endswith(";1"):
            n = n[:-2]
        # quick skip trees (they won't appear as TH1 keys typically, but safe)
        if n == "events":
            continue
        # Identify SR
        if n.endswith("_SR"):
            base = n[:-3]
            sr.add(base)
        else:
            pre.add(n)

    # Keep intersection, and remove any known 2D names if present (your 2D are ptbb_vs_MET, recoil_vs_HT)
    vars_ = sorted((pre & sr) - {"ptbb_vs_MET", "recoil_vs_HT"})
    return vars_

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="cutcount_out")
    ap.add_argument("--tag", default="monoHiggs")
    ap.add_argument("--sigScale", type=float, default=1.0)
    ap.add_argument("--shapeLogY", action="store_true")  # legacy
    ap.add_argument("--logy", type=int, default=None)    # wrapper alias
    ap.add_argument("--mode", choices=["shape", "yield"], default="shape")
    ap.add_argument("--only", default="")  # legacy
    ap.add_argument("--includeSamples", default="")
    ap.add_argument("--excludeSamples", default="")
    # accept wrapper synonyms if ever passed
    ap.add_argument("--plotMode", choices=["shape", "yield"], default=None)
    ap.add_argument("--include", default="")
    ap.add_argument("--exclude", default="")
    args, _unknown = ap.parse_known_args()

    # harmonize wrapper aliases
    if args.plotMode is not None:
        args.mode = args.plotMode
    if args.include and not args.includeSamples:
        args.includeSamples = args.include
    if args.exclude and not args.excludeSamples:
        args.excludeSamples = args.exclude

    outdir = os.path.abspath(args.outdir)
    if not os.path.isdir(outdir):
        raise SystemExit(f"[ERROR] outdir not found: {outdir}")

    do_shape = (args.mode == "shape")
    do_logy = bool(args.shapeLogY)
    if args.logy is not None:
        do_logy = (args.logy != 0)

    samples = discover_samples(outdir, args.tag)
    if not samples:
        raise SystemExit(f"[ERROR] No files like {args.tag}_*_cutcount.root in {outdir}")

    # filtering
    only = set(_parse_csv(args.only)) if args.only else set()
    inc  = set(_parse_csv(args.includeSamples)) if args.includeSamples else set()
    exc  = set(_parse_csv(args.excludeSamples)) if args.excludeSamples else set()

    if only:
        samples = {k: v for k, v in samples.items() if k in only}
    if inc:
        samples = {k: v for k, v in samples.items() if k in inc}
    if exc:
        samples = {k: v for k, v in samples.items() if k not in exc}

    if not samples:
        raise SystemExit("[ERROR] No samples left after filtering.")

    # signal key
    sig_key = None
    for k in samples:
        if is_signal_key(k):
            sig_key = k
            break
    if sig_key is None:
        raise SystemExit(f"[ERROR] Could not find signal key among: {sorted(samples.keys())}")

    variables = list_variables_from_signal(samples[sig_key])
    if not variables:
        raise SystemExit("[ERROR] No variables with both PRE and SR histograms found in signal file.")

    plotdir = os.path.join(outdir, "plots_prepost_all")
    os.makedirs(plotdir, exist_ok=True)
    print(f"[INFO] Plotting {len(variables)} variables into: {plotdir}")

    bkg_palette = [
        ROOT.kAzure - 9, ROOT.kSpring - 6, ROOT.kOrange - 2, ROOT.kViolet - 6,
        ROOT.kTeal - 5, ROOT.kPink - 6, ROOT.kGray + 1, ROOT.kYellow - 7
    ]

    # vars with sharper peaks -> more headroom
    hi_vars = {"dphi_bb_met", "dphi_bb", "dphi_bjet1_met", "phi_bjet1_met", "dr_bb", "mbb"}

    for var in variables:
        # open files
        files: Dict[str, ROOT.TFile] = {}
        for k, path in samples.items():
            f = ROOT.TFile.Open(path)
            if f and not f.IsZombie():
                files[k] = f

        # get signal hists
        sig_pre = get_region_hist(files[sig_key], var, "pre")
        sig_sr  = get_region_hist(files[sig_key], var, "sr")
        if not sig_pre or not sig_sr:
            for f in files.values(): f.Close()
            continue

        if do_shape:
            normalize(sig_pre); normalize(sig_sr)

        if args.sigScale != 1.0:
            sig_pre.Scale(args.sigScale)
            sig_sr.Scale(args.sigScale)

        sig_pre.SetLineColor(ROOT.kRed); sig_pre.SetLineWidth(3); sig_pre.SetFillStyle(0)
        sig_sr.SetLineColor(ROOT.kRed);  sig_sr.SetLineWidth(3);  sig_sr.SetFillStyle(0)

        def build_stack(region: str):
            st = ROOT.THStack(f"st_{region}_{var}", "")
            sumh = None
            comps = []
            bi = 0
            for k in sorted(files.keys()):
                if k == sig_key:
                    continue
                h = get_region_hist(files[k], var, region)
                if not h:
                    continue
                if do_shape:
                    normalize(h)
                col = bkg_palette[bi % len(bkg_palette)]
                bi += 1
                h.SetFillColor(col)
                h.SetLineColor(ROOT.kBlack)
                h.SetLineWidth(1)
                st.Add(h, "HIST")
                comps.append((k, h))
                if sumh is None:
                    sumh = h.Clone(f"sum_{region}_{var}")
                    sumh.SetDirectory(0)
                else:
                    sumh.Add(h)
            return st, sumh, comps

        st_pre, sum_pre, comps_pre = build_stack("pre")
        st_sr,  sum_sr,  comps_sr  = build_stack("sr")

        # y max computed from stacked sum and signal
        ymax_pre = max(float(sig_pre.GetMaximum()), float(sum_pre.GetMaximum()) if sum_pre else 0.0)
        ymax_sr  = max(float(sig_sr.GetMaximum()),  float(sum_sr.GetMaximum())  if sum_sr  else 0.0)

        head_lin = 1.60 if var in hi_vars else 1.60
        head_log = 60.0 if var in hi_vars else 40.0

        c = make_canvas()
        _keep = []  # keep ROOT objects alive (legends/latex)

        # PRE pad
        p1 = c.cd(1)
        if do_logy: p1.SetLogy()
        st_pre.SetMinimum(1e-6 if do_logy else 0.0)

        st_pre.SetMaximum((head_log if do_logy else head_lin) * max(1e-6, ymax_pre))

        st_pre.Draw("HIST")
        frame1 = st_pre.GetHistogram()
        if not frame1:
            # no bkgs -> use signal for axes
            frame1 = sig_pre
            sig_pre.Draw("HIST")
        frame1.SetMinimum(1e-6 if do_logy else 0.0)
        frame1.SetMaximum((head_log if do_logy else head_lin) * max(1e-6, ymax_pre))
        frame1.GetXaxis().SetTitle(var.replace("_", " "))
        frame1.GetYaxis().SetTitle("Normalized entries" if do_shape else "Events")
        set_axis_style(frame1)
        sig_pre.Draw("HIST SAME")

        # SR pad
        p2 = c.cd(2)
        if do_logy: p2.SetLogy()
        st_sr.SetMinimum(1e-6 if do_logy else 0.0)

        st_sr.SetMaximum((head_log if do_logy else head_lin) * max(1e-6, ymax_sr))

        st_sr.Draw("HIST")
        frame2 = st_sr.GetHistogram()
        if not frame2:
            frame2 = sig_sr
            sig_sr.Draw("HIST")
        frame2.SetMinimum(1e-6 if do_logy else 0.0)
        frame2.SetMaximum((head_log if do_logy else head_lin) * max(1e-6, ymax_sr))
        frame2.GetXaxis().SetTitle(var.replace("_", " "))
        frame2.GetYaxis().SetTitle("Normalized entries" if do_shape else "Events")
        set_axis_style(frame2)
        sig_sr.Draw("HIST SAME")

        def draw_legend(pad, comps, region_label, sig_hist):
            pad.cd()
            leg = ROOT.TLegend(0.52, 0.55, 0.95, 0.88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetTextFont(42)
            leg.SetTextSize(0.035)
            for k, h in comps:
                leg.AddEntry(h, k, "f")
            sig_lab = f"{sig_key}" + (f" (×{args.sigScale:g})" if args.sigScale != 1.0 else "")
            leg.AddEntry(sig_hist, sig_lab, "l")
            leg.Draw()
            _keep.append(leg)
            _lat = cms_header(pad, region_label)
            _keep.append(_lat)

        draw_legend(p1, comps_pre, "PRE", sig_pre)
        draw_legend(p2, comps_sr,  "SR",  sig_sr)

        outpng = os.path.join(plotdir, f"{args.tag}_{var}_PREvsSR_stack.png")
        c.SaveAs(outpng)
        c.Close()

        for f in files.values():
            f.Close()

    print(f"[OK] Saved stacked PRE/SR plots in {plotdir}")

if __name__ == "__main__":
    main()
