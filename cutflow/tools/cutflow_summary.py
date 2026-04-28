#!/usr/bin/env python3
"""
cutflow_summary.py

Combine NOMINAL cutflow histograms from per-sample *_cutcount.root outputs and write:
  1) Combined breakdown CSV
  2) Significance CSV
  3) Optional LaTeX cutflow table (Overleaf-ready)

By design, this script reads only files matching:
  <tag>_<sample>_cutcount.root

So it summarizes the nominal outputs, not the JES/JER shifted files.

NLO-safe version:
  - uses cutflow_wgt  = sum(w)
  - uses cutflow_wgt2 = sum(w^2)
  - background statistical uncertainty is sqrt(sum(w^2))

Requires: PyROOT
"""

from __future__ import annotations

import os
import math
import csv
import argparse
from typing import Dict, List, Optional

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


# ============================================================
# Small helpers
# ============================================================
def sanitize_label(label: str) -> str:
    return (label or "").replace(",", ";").strip()


def latex_escape(s: str) -> str:
    """Escape LaTeX special chars in a safe order."""
    if s is None:
        return ""
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("$", r"\$")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    return s


def split_csv(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_systs(s: str) -> List[float]:
    vals = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return vals or [0.0, 0.10, 0.20]


def safe_s_over_b(S: float, B: float) -> float:
    if abs(B) < 1e-15:
        return 0.0
    return S / B


def safe_s_over_sigmaB(S: float, sigma_B: float) -> float:
    if sigma_B <= 0:
        return 0.0
    return S / sigma_B


def approx_Z_nlo_with_syst(S: float, B: float, varB: float, rel_sys: float) -> float:
    """
    NLO-safe approximate counting significance.

    Uses:
        sigma_stat^2 = varB = sum(w^2)
        sigma_sys^2  = (rel_sys * |B|)^2
        sigma_tot^2  = sigma_stat^2 + sigma_sys^2

    Then:
        Z = S / sqrt(sigma_tot^2)

    This is safer for NLO weighted samples than using Asimov formulas that
    assume simple Poisson counting with positive yields.
    """
    if S <= 0:
        return 0.0

    sigma2 = max(varB, 0.0) + (rel_sys * abs(B)) ** 2
    if sigma2 <= 0:
        return 0.0

    return S / math.sqrt(sigma2)


# ============================================================
# ROOT helpers
# ============================================================
def open_cutflow_hist(root_path: str, histname: str):
    f = ROOT.TFile.Open(root_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {root_path}")

    h = f.Get(histname)
    if not h:
        f.Close()
        raise RuntimeError(f"Missing histogram '{histname}' in: {root_path}")
    if not h.InheritsFrom("TH1"):
        f.Close()
        raise RuntimeError(f"Object '{histname}' is not a TH1 in: {root_path}")

    hc = h.Clone(os.path.basename(root_path).replace(".root", "") + "__" + histname)
    hc.SetDirectory(0)
    f.Close()
    return hc


def get_bin_labels(h) -> List[str]:
    return [sanitize_label(h.GetXaxis().GetBinLabel(i) or f"cut{i}") for i in range(1, h.GetNbinsX() + 1)]


# ============================================================
# Sample discovery / filtering
# ============================================================
def discover_samples(indir: str, tag: str) -> Dict[str, str]:
    """
    Discover nominal files only:
      <tag>_<sample>_cutcount.root
    """
    samples = {}
    for fn in os.listdir(indir):
        if not fn.startswith(tag + "_"):
            continue
        if not fn.endswith("_cutcount.root"):
            continue
        key = fn[len(tag) + 1 : -len("_cutcount.root")]
        samples[key] = os.path.join(indir, fn)
    return samples


def filter_samples(samples: Dict[str, str], signal_key: str,
                   include: Optional[str], exclude: Optional[str]) -> Dict[str, str]:
    """
    Apply include/exclude filtering.

    - include applies to BACKGROUNDS only
    - exclude applies to ALL samples, including signal
    """
    inc = set(split_csv(include))
    exc = set(split_csv(exclude))

    keys = list(samples.keys())

    if inc:
        kept = []
        for k in keys:
            if k == signal_key:
                kept.append(k)
            elif k in inc:
                kept.append(k)
        keys = kept

    if exc:
        keys = [k for k in keys if k not in exc]

    return {k: samples[k] for k in keys if k in samples}


# ============================================================
# Output helpers
# ============================================================
def write_latex_table(out_tex: str,
                      cutlabels: List[str],
                      sig_vals: List[float],
                      bkg_vals_by_key: Dict[str, List[float]],
                      bkg_keys: List[str]) -> None:
    os.makedirs(os.path.dirname(out_tex) or ".", exist_ok=True)

    colspec = "l" + "r" * (2 + len(bkg_keys))
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by cutflow_summary.py\n")
        f.write("\\begin{table}[!htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n")
        f.write("\\hline\n")

        header = ["Cut", "Signal"] + [latex_escape(k) for k in bkg_keys] + ["B total"]
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\hline\n")

        for i, lab in enumerate(cutlabels):
            row = [latex_escape(lab), f"{sig_vals[i]:.6g}"]
            btot = 0.0
            for k in bkg_keys:
                v = bkg_vals_by_key[k][i]
                row.append(f"{v:.6g}")
                btot += v
            row.append(f"{btot:.6g}")
            f.write(" & ".join(row) + " \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Cutflow yields for signal and background samples.}\n")
        f.write("\\label{tab:cutflow}\n")
        f.write("\\end{table}\n")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Combine nominal cutflows into CSV (+ optional LaTeX).")

    ap.add_argument("--inDir", default=None, help="Directory containing <tag>_*_cutcount.root")
    ap.add_argument("--outCSV", default=None, help="Output CSV path for combined breakdown table")

    ap.add_argument("--outdir", default="cutcount_out", help="(compat) Same as --inDir")
    ap.add_argument("--tag", default="monoHiggs", help="Output file prefix tag")
    ap.add_argument("--signal", default="signal", help="Sample key to treat as signal")
    ap.add_argument("--hist", default="cutflow_wgt", help="Cutflow histogram name")
    ap.add_argument("--histVar", default="cutflow_wgt2", help="Cutflow variance histogram name (= sum w^2)")
    ap.add_argument("--systs", default="0,0.10,0.20", help="Comma-separated relative background systematics")

    ap.add_argument("--include", default=None,
                    help="Comma-separated background keys to include (signal kept by default if present).")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated sample keys to exclude (applies to signal too if listed).")

    ap.add_argument("--writeLatex", action="store_true",
                    help=r"Also write a LaTeX cutflow table for Overleaf via \input{...}")
    ap.add_argument("--outTex", default=None, help="Output .tex path (default: next to outCSV)")

    args = ap.parse_args()

    indir = os.path.abspath(args.inDir or args.outdir)
    if not os.path.isdir(indir):
        raise SystemExit(f"[ERROR] Input directory not found: {indir}")

    out_csv = os.path.abspath(args.outCSV or os.path.join(indir, "cutflow_combined.csv"))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    samples = discover_samples(indir, args.tag)
    if not samples:
        raise SystemExit(f"[ERROR] No files found matching: {indir}/{args.tag}_*_cutcount.root")

    samples = filter_samples(samples, args.signal, args.include, args.exclude)
    if not samples:
        raise SystemExit("[ERROR] No samples left after --include/--exclude filtering.")
    if args.signal not in samples:
        raise SystemExit(f"[ERROR] Signal key '{args.signal}' not found after filtering. Found: {sorted(samples.keys())}")

    # Load signal
    h_sig = open_cutflow_hist(samples[args.signal], args.hist)
    nb = h_sig.GetNbinsX()
    sig_labels = get_bin_labels(h_sig)

    # Load backgrounds and their variances
    h_bkgs = {}
    h_bkgs_var = {}

    for k in sorted(samples.keys()):
        if k == args.signal:
            continue
        try:
            h = open_cutflow_hist(samples[k], args.hist)
            hvar = open_cutflow_hist(samples[k], args.histVar)

            # bin-count consistency
            if h.GetNbinsX() != nb:
                raise RuntimeError(f"bin mismatch: signal has {nb}, {k} has {h.GetNbinsX()}")
            if hvar.GetNbinsX() != nb:
                raise RuntimeError(f"variance-bin mismatch: signal has {nb}, {k} has {hvar.GetNbinsX()}")

            # bin-label consistency
            h_labels = get_bin_labels(h)
            if h_labels != sig_labels:
                raise RuntimeError(
                    f"cut-label mismatch for sample '{k}'.\n"
                    f"signal labels = {sig_labels}\n"
                    f"{k} labels     = {h_labels}"
                )

            hvar_labels = get_bin_labels(hvar)
            if hvar_labels != sig_labels:
                raise RuntimeError(
                    f"variance cut-label mismatch for sample '{k}'.\n"
                    f"signal labels = {sig_labels}\n"
                    f"{k} variance labels = {hvar_labels}"
                )

            h_bkgs[k] = h
            h_bkgs_var[k] = hvar

        except Exception as e:
            print(f"[WARN] Skipping {k}: {e}")

    if not h_bkgs:
        raise SystemExit("[ERROR] No background cutflow histograms loaded.")

    # Sum background yield histograms
    h_bsum = None
    h_bsum_var = None

    for k in sorted(h_bkgs.keys()):
        h = h_bkgs[k]
        hvar = h_bkgs_var[k]

        if h_bsum is None:
            h_bsum = h.Clone("Bsum")
            h_bsum.SetDirectory(0)
        else:
            h_bsum.Add(h)

        if h_bsum_var is None:
            h_bsum_var = hvar.Clone("BsumVar")
            h_bsum_var.SetDirectory(0)
        else:
            h_bsum_var.Add(hvar)

    bkg_keys = sorted(h_bkgs.keys())
    cutlabels = sig_labels
    sig_vals: List[float] = []
    bkg_vals_by_key = {k: [] for k in bkg_keys}

    # ------------------------------------------------------------
    # Combined yield breakdown CSV
    # ------------------------------------------------------------
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["icut", "cutlabel", "signal"] + bkg_keys + ["B_total"])

        for i in range(1, nb + 1):
            label = cutlabels[i - 1]
            S = float(h_sig.GetBinContent(i))
            sig_vals.append(S)

            row = [i, label, S]
            btot = 0.0
            for k in bkg_keys:
                val = float(h_bkgs[k].GetBinContent(i))
                bkg_vals_by_key[k].append(val)
                row.append(val)
                btot += val
            row.append(btot)
            w.writerow(row)

    print(f"[DONE] Combined breakdown CSV: {out_csv}")

    # ------------------------------------------------------------
    # Significance CSV (NLO-safe)
    # ------------------------------------------------------------
    base, _ = os.path.splitext(out_csv)
    signif_csv = base + "_significance.csv"
    systs = parse_systs(args.systs)

    with open(signif_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["icut", "cutlabel", "S", "B", "sigma_B", "S_over_B", "S_over_sigmaB"]
        for s in systs:
            if abs(s) < 1e-12:
                header.append("Z_nlo_0pct")
            else:
                header.append(f"Z_nlo_{int(round(s * 100))}pct")
        w.writerow(header)

        for i in range(1, nb + 1):
            label = cutlabels[i - 1]
            S = float(h_sig.GetBinContent(i))
            B = float(h_bsum.GetBinContent(i)) if h_bsum else 0.0
            varB = float(h_bsum_var.GetBinContent(i)) if h_bsum_var else 0.0
            sigma_B = math.sqrt(varB) if varB > 0.0 else 0.0

            row = [
                i,
                label,
                S,
                B,
                sigma_B,
                safe_s_over_b(S, B),
                safe_s_over_sigmaB(S, sigma_B),
            ]
            for s in systs:
                row.append(approx_Z_nlo_with_syst(S, B, varB, s))
            w.writerow(row)

    print(f"[DONE] Significance CSV: {signif_csv}")

    # ------------------------------------------------------------
    # Optional LaTeX table
    # ------------------------------------------------------------
    if args.writeLatex:
        out_tex = os.path.abspath(args.outTex or (base + ".tex"))
        write_latex_table(out_tex, cutlabels, sig_vals, bkg_vals_by_key, bkg_keys)
        print(f"[DONE] LaTeX cutflow table: {out_tex}")

    # ------------------------------------------------------------
    # Final summary line
    # ------------------------------------------------------------
    last_i = nb
    last_label = cutlabels[last_i - 1]
    S_last = float(h_sig.GetBinContent(last_i))
    B_last = float(h_bsum.GetBinContent(last_i)) if h_bsum else 0.0
    varB_last = float(h_bsum_var.GetBinContent(last_i)) if h_bsum_var else 0.0
    sigma_B_last = math.sqrt(varB_last) if varB_last > 0.0 else 0.0

    print(
        f"[INFO] Last cut: {last_i} ({last_label})  "
        f"S={S_last:.6g}  B={B_last:.6g}  "
        f"sigma_B={sigma_B_last:.6g}  "
        f"S/B={safe_s_over_b(S_last, B_last):.6g}  "
        f"S/sigma_B={safe_s_over_sigmaB(S_last, sigma_B_last):.6g}"
    )


if __name__ == "__main__":
    main()
