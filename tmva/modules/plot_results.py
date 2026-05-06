#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt

# -------------------------
# Global plotting style (journal-like, HEP friendly)
# -------------------------
plt.rcParams.update({
    "figure.figsize": (7.2, 6.2),
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# -------------------------
# Helpers: cut evaluation
# -------------------------
def _to_numpy_expr(expr: str) -> str:
    """
    Convert ROOT-like cut string to a numpy-evaluable python expression.

    Key point: enforce parentheses so numpy bitwise ops (&,|,~) work elementwise
    without precedence bugs.

    Examples:
      (isSR==1 && Nlep==0)  ->  ((isSR==1) & (Nlep==0))
      (A>1 || B<2)          ->  ((A>1) | (B<2))
    """
    if expr is None:
        return ""
    s = str(expr).strip()
    if not s:
        return ""

    # Normalize spaces (helps simple replacements)
    s = " ".join(s.split())

    # Protect '!=' before handling '!'
    s = s.replace("!=", "__NE__")

    # Replace logical AND/OR with parenthesis-safe forms
    # Turn:  X && Y  into  (X) & (Y)
    # We do this by splitting with sentinels first.
    s = s.replace("&&", ") & (")
    s = s.replace("||", ") | (")

    # Handle NOT (but not != which we protected)
    # ROOT uses '!' for NOT; numpy uses '~' for elementwise NOT on bool arrays
    s = s.replace("!", "~")

    # Restore '!='
    s = s.replace("__NE__", "!=")

    # Ensure the full expression is wrapped
    # Also ensure we have outer parentheses even if user didn't provide
    if not (s.startswith("(") and s.endswith(")")):
        s = f"({s})"
    # Now add one more wrap to guarantee correct binding
    s = f"({s})"

    return s
def eval_cut_mask(arr: dict, cut: str):
    """
    Evaluate boolean mask from cut expression using numpy arrays in `arr`.
    If cut is empty -> all True.
    """
    cut = (cut or "").strip()
    if not cut:
        # all pass
        first_key = next(iter(arr.keys()))
        return np.ones_like(arr[first_key], dtype=bool)

    expr = _to_numpy_expr(cut)

    # Ensure arrays are numpy arrays
    local = {k: np.asarray(v) for k, v in arr.items()}

    try:
        # This works for expressions with &,|,~ and comparisons
        mask = eval(expr, {"__builtins__": {}}, local)
    except Exception as e:
        raise RuntimeError(f"Failed to evaluate cut:\n  cut='{cut}'\n  expr='{expr}'\nError: {e}")
    return np.asarray(mask, dtype=bool)

# -------------------------
# Helper: histogram sumw and sumw2 (MC stat band)
# -------------------------
def hist_sumw_sumw2(root_path, tree, xbranch, wbranch, bins, xlow, xhigh, cut=None, extra_branches=None):
    extra_branches = extra_branches or []
    need = [xbranch, wbranch] + list(extra_branches)

    with uproot.open(root_path) as f:
        t = f[tree]
        arr = t.arrays(need, library="np")

    x = arr[xbranch].astype(np.float64)
    w = arr[wbranch].astype(np.float64)

    # Apply cut if provided
    if cut:
        m = eval_cut_mask(arr, cut)
        x = x[m]
        w = w[m]

    edges = np.linspace(xlow, xhigh, bins + 1)
    sumw, _ = np.histogram(x, bins=edges, weights=w)
    sumw2, _ = np.histogram(x, bins=edges, weights=w * w)

    return sumw, sumw2, edges

def savefig_dual(path_noext):
    plt.savefig(path_noext + ".pdf")
    plt.savefig(path_noext + ".png", dpi=500)

# -------------------------
# Background list helpers
# -------------------------
def get_bkg_list_from_env():
    env_bkgs = os.environ.get("BKG_SAMPLES", "").strip()
    if env_bkgs:
        return env_bkgs.split()
    # fallback
    return ["tt_semilep", "tt_dilep", "tt_had", "st_tW", "st_tch_top", "st_tch_tbar", "wjets", "zvvjets"]

def nice_label(sample):
    nice = {
        "tt_semilep": "tt semilep",
        "tt_dilep":   "tt dilep",
        "tt_had":     "tt had",
        "st_tW":      "tW",
        "st_tch_top": "t-ch top",
        "st_tch_tbar":"t-ch tbar",
        "wjets":      "W+jets",
        "zvvjets":    "Z(νν)+jets",
    }
    return nice.get(sample, sample)

# -------------------------
# Plot 1: stacked BDT + stat band + ratio
# -------------------------
def plot_bdt_stack(workdir, tag, method, tree="events", wbranch="weight",
                   bins=30, xlow=-1.0, xhigh=1.0, logy=True,
                   signal_scale=5.0, make_ratio=True, sr_cut=None):

    bdt_branch = f"bdt_{method}"

    # SR cut logic:
    # 1) explicit argument sr_cut (from CLI) if provided
    # 2) else env SR_CUT
    # 3) else default isSR/Nlep
    if sr_cut is None:
        sr_cut = os.environ.get("SR_CUT", "").strip()
    if not sr_cut:
        sr_cut = "(isSR==1 && Nlep==0)"

    sig_file = os.path.join(workdir, f"{tag}_signal_withBDT.root")
    if not os.path.exists(sig_file):
        raise FileNotFoundError(f"Missing signal file: {sig_file}")

    bkg_list = get_bkg_list_from_env()

    # Build bkg map only for requested backgrounds
    bkg_map = {}
    for b in bkg_list:
        path = os.path.join(workdir, f"{tag}_{b}_withBDT.root")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing background file requested by BKG_SAMPLES: {path}\n"
                f"Either produce it in apply step, or remove '{b}' from BKG_SAMPLES."
            )
        bkg_map[nice_label(b)] = path

    # Variables needed by SR_CUT
    # (we don't parse the cut, so just include common ones; harmless if unused)
    cut_branches = ["isSR", "Nlep", "Nbjets", "Njets", "MET", "HT", "mbb", "ptbb",
                    "dr_bb", "dphi_bb", "dphi_bb_met", "dphi_bjet1_met", "recoil",
                    "balance", "ptbb_minus_met", "bjet1_pt", bdt_branch, wbranch]

    # Signal
    s_sumw, s_sumw2, edges = hist_sumw_sumw2(
        sig_file, tree, bdt_branch, wbranch, bins, xlow, xhigh,
        cut=sr_cut, extra_branches=cut_branches
    )

    # Backgrounds
    bkg_names = list(bkg_map.keys())
    b_sumw_list = []
    b_sumw2_list = []

    for name in bkg_names:
        bw, bw2, _ = hist_sumw_sumw2(
            bkg_map[name], tree, bdt_branch, wbranch, bins, xlow, xhigh,
            cut=sr_cut, extra_branches=cut_branches
        )
        b_sumw_list.append(bw)
        b_sumw2_list.append(bw2)

    # Totals
    if len(b_sumw_list) == 0:
        raise RuntimeError("No backgrounds configured (BKG_SAMPLES is empty).")

    b_tot = np.sum(np.vstack(b_sumw_list), axis=0)
    b_tot_w2 = np.sum(np.vstack(b_sumw2_list), axis=0)
    b_err = np.sqrt(b_tot_w2)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]

    if make_ratio:
        fig = plt.figure(figsize=(7.2, 7.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.2], hspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        rax = fig.add_subplot(gs[1, 0], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(7.2, 5.5))
        rax = None

    # keep your palette, but allow many bkgs (cycle)
    palette = [
        "#4C72B0", "#55A868", "#C44E52", "#8172B2",
        "#CCB974", "#64B5CD", "#8C8C8C", "#DD8452",
        "#937860", "#DA8BC3"
    ]

    bottom = np.zeros_like(b_tot, dtype=float)
    for i, (name, bw) in enumerate(zip(bkg_names, b_sumw_list)):
        col = palette[i % len(palette)]
        ax.bar(centers, bw, width=widths, bottom=bottom,
               align="center", linewidth=0.0,
               color=col, label=name)
        bottom = bottom + bw

    # Stat uncertainty band (make it visible)
    ax.bar(
        centers, 2.0 * b_err, width=widths,
        bottom=b_tot - b_err, align="center",
        facecolor="none",
        edgecolor="gray",
        linewidth=0.8,
        hatch="////",
        label="Bkg stat. unc."
    )

    # Signal (scaled)
    s_scaled = s_sumw * float(signal_scale)
    ax.step(edges, np.r_[s_scaled, s_scaled[-1]],
            where="post", color="black", linewidth=2.2,
            label=f"Signal ×{signal_scale:g}")

    ax.set_ylabel("Events")
    ax.set_xlim(xlow, xhigh)

    if logy:
        ax.set_yscale("log")
        pos = b_tot[b_tot > 0]
        ymin = (0.2 * np.min(pos)) if pos.size else 1e-2
        ymax = float(np.max(b_tot + b_err + s_scaled)) if b_tot.size else 1.0
        ax.set_ylim(max(1e-2, ymin), max(1.0, 50.0 * ymax))

    ax.legend(ncol=2, loc="upper left")

    # Text block
    text = (
        "Mono-Higgs (SR)\n"
        f"Classifier: {method}\n"
        f"Selection: {sr_cut}"
    )
    ax.text(0.02, 0.97, text, transform=ax.transAxes, va="top", ha="left")

    if make_ratio:
        ax.tick_params(labelbottom=False)
        denom = np.where(b_tot > 0, b_tot, np.nan)
        ratio = s_scaled / denom

        rax.axhline(0.0, color="black", linewidth=1.0)
        rax.plot(centers, ratio, marker="o", markersize=3.5,
                 linewidth=1.2, color="black")

        rax.set_ylabel(fr"$S\times{signal_scale:g}/B$")
        rax.set_xlabel(f"BDT score ({method})")

        # Safe y-limits
        rmax = np.nanmax(ratio)
        if not np.isfinite(rmax) or rmax <= 0:
            rax.set_ylim(0.0, 1.0)
        else:
            rax.set_ylim(0.0, 1.25 * rmax)

        rax.grid(True, which="both", axis="y", alpha=0.25)
    else:
        ax.set_xlabel(f"BDT score ({method})")

    out = os.path.join(workdir, f"bdt_stack_SR_{method}")
    savefig_dual(out)
    plt.close(fig)
    print(f"[OK] Wrote: {out}.pdf/.png")

# -------------------------
# Plot 2: Z vs cut scan
# -------------------------
def plot_scan(workdir, method):
    csv = os.path.join(workdir, f"scan_{method}.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(f"Missing: {csv}")

    df = pd.read_csv(csv)
    for col in ["cut", "S", "B", "Z"]:
        if col not in df.columns:
            raise RuntimeError(f"{csv} missing '{col}'. Found: {list(df.columns)}")

    best = df.loc[df["Z"].idxmax()]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(df["cut"], df["Z"], linewidth=2.2, color="black")
    ax.axvline(float(best["cut"]), linestyle="--", linewidth=1.6, color="red",
               label=f"Best cut = {best['cut']:.3f}")

    ax.set_xlabel(f"BDT cut ({method})")
    ax.set_ylabel(r"Significance  $Z = S/\sqrt{S+B}$")
    ax.legend(loc="best")

    txt = f"Best:\ncut={best['cut']:.3f}\nS={best['S']:.2f}\nB={best['B']:.2f}\nZ={best['Z']:.3f}"
    ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top", ha="left")

    out = os.path.join(workdir, f"scan_Z_{method}")
    savefig_dual(out)
    plt.close(fig)
    print(f"[OK] Wrote: {out}.pdf/.png")

# -------------------------
# Plot 3: Brazil plot
# -------------------------
def plot_brazil(workdir, method):
    js = os.path.join(workdir, f"limits_{method}.json")
    if not os.path.exists(js):
        raise FileNotFoundError(f"Missing: {js}")

    with open(js) as f:
        d = json.load(f)

    exp = d["expected_ul_mu"]
    obs = float(d["observed_ul_mu"])

    m2 = float(exp["minus2sigma"])
    m1 = float(exp["minus1sigma"])
    med = float(exp["median"])
    p1 = float(exp["plus1sigma"])
    p2 = float(exp["plus2sigma"])

    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    ax.fill_between([0.85, 1.15], [m2, m2], [p2, p2], color="#F5E663",
                    label=r"Expected $\pm 2\sigma$")
    ax.fill_between([0.85, 1.15], [m1, m1], [p1, p1], color="#7AC77A",
                    label=r"Expected $\pm 1\sigma$")

    ax.plot([0.85, 1.15], [med, med], color="black", linestyle="--",
            linewidth=2.0, label="Expected (median)")
    ax.plot([0.85, 1.15], [obs, obs], color="red",
            linewidth=2.2, label="Observed")

    ax.set_xlim(0.7, 1.3)
    ax.set_xticks([1.0])
    ax.set_xticklabels([f"{method} (SR-only)"])
    ax.set_ylabel(r"95% CL upper limit on signal strength $\mu$")
    ax.set_ylim(0.0, max(p2, obs) * 1.25)

    ax.legend(loc="upper right")
    ax.text(0.04, 0.95,
            f"obs: {obs:.3g}\nexp: {med:.3g}\n(-1σ,+1σ)=({m1:.3g},{p1:.3g})",
            transform=ax.transAxes, va="top", ha="left")

    out = os.path.join(workdir, f"limits_brazil_{method}")
    savefig_dual(out)
    plt.close(fig)
    print(f"[OK] Wrote: {out}.pdf/.png")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", required=True, choices=["BDT", "BDTG"])
    ap.add_argument("--tree", default="events")
    ap.add_argument("--weight", default="weight")
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--xlow", type=float, default=-1.0)
    ap.add_argument("--xhigh", type=float, default=1.0)
    ap.add_argument("--signal-scale", type=float, default=5.0)
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--no-ratio", action="store_true")
    ap.add_argument("--sr-cut", default=None,
                    help="Override SR selection (otherwise uses env SR_CUT, else default '(isSR==1 && Nlep==0)')")
    args = ap.parse_args()

    plot_bdt_stack(
        args.workdir, args.tag, args.method,
        tree=args.tree, wbranch=args.weight,
        bins=args.bins, xlow=args.xlow, xhigh=args.xhigh,
        logy=(not args.linear),
        signal_scale=args.signal_scale,
        make_ratio=(not args.no_ratio),
        sr_cut=args.sr_cut,
    )

    plot_scan(args.workdir, args.method)
    plot_brazil(args.workdir, args.method)

    print("[DONE] All journal-level plots created.")

if __name__ == "__main__":
    main()
