#!/usr/bin/env python3
"""
cutcount_diagnostics.py

Create diagnostic plots from cutcount outputs produced by monohiggs_cutflow.cli.

Reads:
  <tag>_<sample>_cutcount.root

Expected:
  - cutflow_wgt   (TH1) weighted cumulative yields after each cut step
  - cutflow_wgt2  (TH1) weighted-squared cumulative yields after each cut step

Outputs:
  <outdir>/plots_cutflow/*.png
"""

import os
import math
import argparse
from typing import Dict, List, Tuple, Optional

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)


# ============================================================
# Significance helpers
# ============================================================
def asimov_Z_with_syst_positive_weights(S: float, B: float, rel_sys: float) -> float:
    """
    Original Asimov significance for positive-weight counting experiments.
    Kept for reference / compatibility, but not used for the final NLO-safe plot.
    """
    if S <= 0 or B <= 0:
        return 0.0

    sigma_b = rel_sys * B
    if sigma_b <= 0:
        z2 = 2.0 * ((S + B) * math.log(1.0 + S / B) - S)
        return math.sqrt(max(z2, 0.0))

    term1 = (S + B) * math.log(((S + B) * (B + sigma_b**2)) / (B**2 + (S + B) * sigma_b**2))
    term2 = (B**2 / sigma_b**2) * math.log(1.0 + (sigma_b**2 * S) / (B * (B + sigma_b**2)))
    z2 = 2.0 * (term1 - term2)
    return math.sqrt(max(z2, 0.0))


def nlo_Z_with_syst(S: float, B: float, varB: float, rel_sys: float) -> float:
    """
    NLO-safe approximate significance:

        sigma_stat^2 = varB = sum(w^2)
        sigma_sys^2  = (rel_sys * |B|)^2
        sigma_tot^2  = sigma_stat^2 + sigma_sys^2

        Z = S / sqrt(sigma_tot^2)

    This is consistent with the corrected cutflow_summary.py.
    """
    if S <= 0:
        return 0.0

    sigma2 = max(varB, 0.0) + (rel_sys * abs(B)) ** 2
    if sigma2 <= 0:
        return 0.0

    return S / math.sqrt(sigma2)
#================================
def nlo_Z_with_syst_and_sigstat(S: float, B: float, varS: float, varB: float, rel_sys: float) -> float:
    """
    NLO-safe approximate significance including signal statistical variance:

        sigma_stat^2 = varS + varB = sum_s(w^2) + sum_b(w^2)
        sigma_sys^2  = (rel_sys * |B|)^2
        sigma_tot^2  = sigma_stat^2 + sigma_sys^2

        Z = S / sqrt(sigma_tot^2)
    """
    if S <= 0:
        return 0.0

    sigma2 = max(varS, 0.0) + max(varB, 0.0) + (rel_sys * abs(B)) ** 2
    if sigma2 <= 0:
        return 0.0

    return S / math.sqrt(sigma2)
# ============================================================
# Sample discovery / filtering
# ============================================================
def discover_samples(outdir: str, tag: str) -> Dict[str, str]:
    out = {}
    for fn in os.listdir(outdir):
        if fn.startswith(tag + "_") and fn.endswith("_cutcount.root"):
            key = fn[len(tag) + 1 : -len("_cutcount.root")]
            out[key] = os.path.join(outdir, fn)
    return out


def split_csv(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def filter_samples(samples: Dict[str, str], signal_key: str,
                   include: Optional[str], exclude: Optional[str]) -> Dict[str, str]:
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
# ROOT helpers
# ============================================================
def load_hist_1d(path: str, hname: str) -> Optional[Tuple[List[float], List[float], List[str]]]:
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        return None

    h = f.Get(hname)
    if not h or not h.InheritsFrom("TH1"):
        f.Close()
        return None

    nb = h.GetNbinsX()
    x = [h.GetXaxis().GetBinCenter(i) for i in range(1, nb + 1)]
    y = [float(h.GetBinContent(i)) for i in range(1, nb + 1)]
    labels = [h.GetXaxis().GetBinLabel(i) for i in range(1, nb + 1)]

    f.Close()
    return x, y, labels


def list_th1_names(path: str) -> List[str]:
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        return []

    names = []
    for k in f.GetListOfKeys():
        obj = k.ReadObj()
        if obj.InheritsFrom("TH1"):
            names.append(obj.GetName())

    f.Close()
    return names


def find_hist_name(path: str, candidates: List[str]) -> Optional[str]:
    names = list_th1_names(path)
    if not names:
        return None

    s = set(names)
    for c in candidates:
        if c in s:
            return c

    for c in candidates:
        cl = c.lower()
        for n in names:
            if n.lower() == cl:
                return n

    for c in candidates:
        cl = c.lower()
        for n in names:
            if cl in n.lower():
                return n

    return None


def safe_label(lbl: str, fallback: str) -> str:
    if not lbl:
        return fallback
    return lbl.replace(",", ";").strip()


# ============================================================
# Plot helpers
# ============================================================
def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        raise SystemExit(f"[ERROR] matplotlib not available: {e}")

def savefig(plt, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()
def stacked_cutflow_plot(plt, plot_dir: str, labels: List[str], S: List[float],
                         bkg_components: List[Tuple[str, List[float]]], logy: bool):
    import numpy as np

    x = np.arange(len(labels))
    width = 0.85
    bottom = np.zeros(len(labels), dtype=float)

    plt.figure(figsize=(11.5, 5.8))

    # Stack only the positive part for readability.
    # Negative background bins can appear in NLO weighted samples;
    # they are not suitable for a standard stacked bar plot.
    for name, y in bkg_components:
        y = np.array(y, dtype=float)
        y_pos = np.clip(y, 0.0, None)
        plt.bar(x, y_pos, width=width, bottom=bottom, label=name)
        bottom += y_pos

    plt.plot(x, S, marker="o", linewidth=2.5, label="Signal", color="black")

    if logy:
        plt.yscale("log")
        ymax = max(float(max(bottom)) if len(bottom) else 1.0,
                   float(max([v for v in S if v > 0], default=1.0)))
        ymin_pos = min([v for v in bottom if v > 0] or [1e-6])
        plt.ylim(max(1e-8, 0.5 * ymin_pos), 30.0 * ymax)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Expected yield")
    plt.title("Cumulative cutflow (stacked positive backgrounds) with signal overlay")
    plt.grid(True, which="both", linestyle="--", alpha=0.25)
    plt.legend(frameon=False, fontsize=9, ncol=2)
    savefig(plt, os.path.join(plot_dir, "cutflow_stacked.png"))

def evolution_plots(
    plt,
    plot_dir: str,
    labels: List[str],
    S: List[float],
    B: List[float],
    varS: List[float],
    varB: List[float],
    systs: List[float],
    include_signal_stat: bool = False,
):
    import numpy as np
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    x = np.arange(len(labels))

    def _syst_color(srel: float) -> str:
        if abs(srel) < 1e-12:
            return "C0"
        if abs(srel - 0.10) < 1e-9:
            return "C1"
        if abs(srel - 0.20) < 1e-9:
            return "C2"
        return "C3"

    # --------------------------------------------------------
    # yields
    # --------------------------------------------------------
    plt.figure(figsize=(11.5, 5.2))
    plt.plot(x, S, marker="o", linewidth=2.2, label="Signal")
    plt.plot(x, B, marker="o", linewidth=2.2, label="Total background")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Expected yield")
    plt.title("Cumulative cutflow yields")
    plt.grid(True, which="both", linestyle="--", alpha=0.25)
    plt.legend(frameon=False)
    savefig(plt, os.path.join(plot_dir, "cutflow_yields.png"))

    # --------------------------------------------------------
    # S/B
    # --------------------------------------------------------
    SoverB = [(s / b if abs(b) > 1e-15 else 0.0) for s, b in zip(S, B)]
    plt.figure(figsize=(11.5, 5.2))
    plt.plot(x, SoverB, marker="o", linewidth=2.2)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("S / B")
    plt.title("S/B across cumulative cutflow")
    plt.grid(True, linestyle="--", alpha=0.25)
    savefig(plt, os.path.join(plot_dir, "sob_vs_cut.png"))

    # --------------------------------------------------------
    # significance (NLO-safe)
    # --------------------------------------------------------
    plt.figure(figsize=(11.5, 5.2))
    ax = plt.gca()

    nz = [s for s in systs if s > 0]

    if len(nz) >= 2:
        z_low = []
        z_high = []
        for i in range(len(labels)):
            vals = [nlo_Z_with_syst(S[i], B[i], varB[i], s) for s in nz]
            z_low.append(min(vals))
            z_high.append(max(vals))
        band_lab = f"NLO-safe Z band ({int(round(100 * min(nz)))}--{int(round(100 * max(nz)))}% bkg syst)"
        ax.fill_between(x, z_low, z_high, alpha=0.25, color="0.75", edgecolor="none", label=band_lab)

    for srel in systs:
        Z = [nlo_Z_with_syst(S[i], B[i], varB[i], srel) for i in range(len(labels))]
        lab = "NLO-safe Z (0% syst)" if abs(srel) < 1e-12 else f"NLO-safe Z ({int(round(100 * srel))}% bkg syst)"
        ax.plot(x, Z, marker="o", linewidth=2.2, label=lab, color=_syst_color(srel))

        if include_signal_stat:
            Z_sig = [
                nlo_Z_with_syst_and_sigstat(S[i], B[i], varS[i], varB[i], srel)
                for i in range(len(labels))
            ]
            lab_sig = (
                "NLO-safe Z + sig stat (0% syst)"
                if abs(srel) < 1e-12
                else f"NLO-safe Z + sig stat ({int(round(100 * srel))}% bkg syst)"
            )
            ax.plot(
                x,
                Z_sig,
                marker=None,
                linewidth=1.8,
                linestyle="--",
                label=lab_sig,
                color=_syst_color(srel),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Significance")
    ax.set_title("Significance vs cumulative cut step")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
'''
    if len(nz) >= 1:
        axins = inset_axes(
            ax,
            width="42%",
            height="38%",
            loc="lower left",
            bbox_to_anchor=(0.15, 0.08, 0.9, 0.9),
            bbox_transform=ax.transAxes,
            borderpad=0.6,
        )

        if len(nz) >= 2:
            axins.fill_between(x, z_low, z_high, alpha=0.25, color="0.75", edgecolor="none")

        zmax = 0.0
        for srel in nz:
            Z = [nlo_Z_with_syst(S[i], B[i], varB[i], srel) for i in range(len(labels))]
            zmax = max(zmax, max(Z) if Z else 0.0)
            axins.plot(x, Z, linewidth=1.6, color=_syst_color(srel))

            if include_signal_stat:
                Z_sig = [
                    nlo_Z_with_syst_and_sigstat(S[i], B[i], varS[i], varB[i], srel)
                    for i in range(len(labels))
                ]
                zmax = max(zmax, max(Z_sig) if Z_sig else 0.0)
                axins.plot(x, Z_sig, linewidth=1.3, linestyle="--", color=_syst_color(srel))

        axins.set_xlim(-0.5, len(labels) - 0.5)
        axins.set_ylim(0.0, max(0.10, 1.25 * zmax))
        axins.grid(True, linestyle="--", alpha=0.25)
        axins.set_xticks([])
        axins.set_yticks([0.0, axins.get_ylim()[1]])
        axins.tick_params(labelsize=8)
        axins.set_title("Zoom: syst-dominated", fontsize=9)
        '''
    savefig(plt, os.path.join(plot_dir, "significance_vs_cut.png"))


def efficiency_plots(plt, plot_dir: str, labels: List[str], S: List[float], B: List[float]):
    import numpy as np

    x = np.arange(len(labels))

    def cumulative(y):
        y0 = y[0] if y and abs(y[0]) > 0 else 0.0
        return [(v / y0 if abs(y0) > 0 else 0.0) for v in y]

    def step(y):
        out = []
        for i, v in enumerate(y):
            if i == 0:
                out.append(1.0)
            else:
                prev = y[i - 1]
                out.append(v / prev if abs(prev) > 0 else 0.0)
        return out

    plt.figure(figsize=(11.5, 5.2))
    plt.plot(x, cumulative(S), marker="o", linewidth=2.2, label="Signal")
    plt.plot(x, cumulative(B), marker="o", linewidth=2.2, label="Background")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Cumulative efficiency")
    plt.title("Cumulative efficiency")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend(frameon=False)
    savefig(plt, os.path.join(plot_dir, "eff_cumulative.png"))

    plt.figure(figsize=(11.5, 5.2))
    plt.plot(x, step(S), marker="o", linewidth=2.2, label="Signal")
    plt.plot(x, step(B), marker="o", linewidth=2.2, label="Background")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Step efficiency")
    plt.title("Per-cut step efficiency")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend(frameon=False)
    savefig(plt, os.path.join(plot_dir, "eff_step.png"))


# ============================================================
# ROC helpers
# ============================================================
def roc_like(sig_bins: List[float], bkg_bins: List[float], direction: str) -> Tuple[List[float], List[float]]:
    S0 = sum(sig_bins)
    B0 = sum(bkg_bins)

    epsS, epsB = [], []
    nb = len(sig_bins)

    for i in range(nb):
        if direction == "above":
            S = sum(sig_bins[i:])
            B = sum(bkg_bins[i:])
        else:
            S = sum(sig_bins[: i + 1])
            B = sum(bkg_bins[: i + 1])

        epsS.append(S / S0 if S0 > 0 else 0.0)
        epsB.append(B / B0 if B0 > 0 else 0.0)

    return epsB, epsS


def roc_module(plt, plot_dir: str, samples: Dict[str, str], signal_key: str, roc_vars: List[str]):
    candidates = {
        "dphi_bb": ["dphi_bb", "dPhi_bb", "dphiBB", "dPhiBB", "histDPhiBB", "DPhiBB", "hist_dphi_bb", "h_dphi_bb"],
        "dr_bb": ["dr_bb", "dR_bb", "drBB", "dRBB", "histDRBB", "DRBB", "hist_dr_bb", "h_dr_bb"],
    }

    bkg_keys = [k for k in sorted(samples.keys()) if k != signal_key]

    for var in roc_vars:
        if var not in candidates:
            print(f"[WARN] ROC var '{var}' not supported.")
            continue

        hname = find_hist_name(samples[signal_key], candidates[var])
        if not hname:
            print(f"[WARN] Could not find histogram for '{var}' in signal. Skipping.")
            continue

        sig = load_hist_1d(samples[signal_key], hname)
        if not sig:
            continue
        _, Sbins, _ = sig

        Btot = [0.0] * len(Sbins)
        bkg_by_comp = {}

        for bk in bkg_keys:
            tmp = load_hist_1d(samples[bk], hname)
            if not tmp:
                continue
            _, bbins, _ = tmp
            if len(bbins) != len(Btot):
                continue
            bkg_by_comp[bk] = bbins
            for i in range(len(Btot)):
                Btot[i] += bbins[i]

        if sum(Btot) <= 0:
            continue
        direction = "below" if var in ("dr_bb", "dphi_bb") else "above"

        plt.figure(figsize=(7.0, 6.5))
        eB, eS = roc_like(Sbins, Btot, direction)
        plt.plot(eB, eS, linewidth=2.8, label="Total background")

        for bk, bbins in bkg_by_comp.items():
            eB, eS = roc_like(Sbins, bbins, direction)
            plt.plot(eB, eS, linewidth=1.8, label=bk)

        plt.xlabel(r"$\epsilon_B$")
        plt.ylabel(r"$\epsilon_S$")
        plt.title(f"ROC-like curves (hist scan): {var}")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend(frameon=False, fontsize=9)
        savefig(plt, os.path.join(plot_dir, f"{var}_ROC_components.png"))


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="cutcount_out")
    ap.add_argument("--tag", default="monoHiggs")
    ap.add_argument("--signal", default="signal")
    ap.add_argument("--include", default=None,
                    help="Comma-separated background keys to include (signal kept by default if present).")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated sample keys to exclude (applies to signal too if listed).")
    ap.add_argument("--systs", default="0,0.10,0.20")
    ap.add_argument("--doStack", action="store_true")
    ap.add_argument("--doEvolution", action="store_true")
    ap.add_argument("--doEff", action="store_true")
    ap.add_argument("--doROC", action="store_true")
    ap.add_argument("--rocVars", default="dphi_bb,dr_bb")
    ap.add_argument("--includeSignalStat", action="store_true",
                    help="Also draw significance curves including signal statistical variance from cutflow_wgt2.")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    tag = args.tag
    signal_key = args.signal
    systs = [float(x.strip()) for x in args.systs.split(",") if x.strip()]

    samples = discover_samples(outdir, tag)
    if not samples:
        raise SystemExit(f"[ERROR] No {tag}_*_cutcount.root found in {outdir}")

    samples = filter_samples(samples, signal_key, args.include, args.exclude)
    if not samples:
        raise SystemExit("[ERROR] No samples left after --include/--exclude filtering.")
    if signal_key not in samples:
        raise SystemExit(f"[ERROR] Signal key '{signal_key}' not found after filtering. Available: {sorted(samples.keys())}")

    plot_dir = os.path.join(outdir, "plots_cutflow")
    os.makedirs(plot_dir, exist_ok=True)

    plt = ensure_matplotlib()

    sig = load_hist_1d(samples[signal_key], "cutflow_wgt")
    if not sig:
        raise SystemExit("[ERROR] Missing cutflow_wgt in signal file.")
    _, Svals, labels_raw = sig
    labels = [safe_label(l, f"cut{i+1}") for i, l in enumerate(labels_raw)]

    sig_var = load_hist_1d(samples[signal_key], "cutflow_wgt2")
    if not sig_var:
        print("[WARN] Missing cutflow_wgt2 in signal file; signal statistical variance will be treated as zero.")
        VarSvals = [0.0] * len(Svals)
    else:
        _, VarSvals, _ = sig_var
        if len(VarSvals) != len(Svals):
            print("[WARN] Signal cutflow_wgt2 bin mismatch; signal statistical variance will be treated as zero.")
            VarSvals = [0.0] * len(Svals)

    bkg_keys = [k for k in sorted(samples.keys()) if k != signal_key]
    bkg_components: List[Tuple[str, List[float]]] = []
    Btot = [0.0] * len(Svals)
    VarBtot = [0.0] * len(Svals)

    for bk in bkg_keys:
        tmp = load_hist_1d(samples[bk], "cutflow_wgt")
        if not tmp:
            continue
        _, bvals, _ = tmp
        if len(bvals) != len(Btot):
            continue

        tmp_var = load_hist_1d(samples[bk], "cutflow_wgt2")
        if not tmp_var:
            print(f"[WARN] Missing cutflow_wgt2 in {bk}; skipping variance contribution.")
            bvals_var = [0.0] * len(Btot)
        else:
            _, bvals_var, _ = tmp_var
            if len(bvals_var) != len(Btot):
                print(f"[WARN] cutflow_wgt2 bin mismatch in {bk}; skipping variance contribution.")
                bvals_var = [0.0] * len(Btot)

        bkg_components.append((bk, bvals))

        for i in range(len(Btot)):
            Btot[i] += bvals[i]
            VarBtot[i] += bvals_var[i]

    if args.doStack:
        stacked_cutflow_plot(plt, plot_dir, labels, Svals, bkg_components, logy=True)
    if args.doEvolution:
        evolution_plots(plt, plot_dir, labels, Svals, Btot, VarSvals, VarBtot, systs,
            include_signal_stat=args.includeSignalStat,
        )
    if args.doEff:
        efficiency_plots(plt, plot_dir, labels, Svals, Btot)
    if args.doROC:
        roc_vars = [x.strip() for x in args.rocVars.split(",") if x.strip()]
        roc_module(plt, plot_dir, samples, signal_key, roc_vars)

    print(f"[OK] Wrote plots to: {plot_dir}")


if __name__ == "__main__":
    main()
