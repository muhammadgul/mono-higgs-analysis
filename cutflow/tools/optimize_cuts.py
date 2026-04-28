#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot


REQUIRED_BRANCHES = [
    "weight",
    "MET",
    "Nlep",
    "Njets",
    "Nbjets",
    "HT",
    "mbb",
    "ptbb",
    "dr_bb",
    "dphi_bb",
    "dphi_bb_met",
    "bjet1_pt",
    "dphi_bjet1_met",
    "recoil",
    "balance",
    "ptbb_minus_met",
    "isSR",
    "isCR_top",
]


@dataclass
class VarSpec:
    name: str
    mode: str   # "lower", "upper", "window", "int_lower"
    label: str
    center: Optional[float] = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="1D cut optimization on monoHiggs *_cutcount.root outputs."
    )
    ap.add_argument("--outdir", required=True, help="Directory containing *_cutcount.root files")
    ap.add_argument("--tag", default="monoHiggs", help="File prefix tag")
    ap.add_argument("--signal", default="signal", help="Signal sample key")
    ap.add_argument("--include", default="", help="Comma-separated background keys to include")
    ap.add_argument("--exclude", default="", help="Comma-separated sample keys to exclude")
    ap.add_argument(
        "--tree",
        default="",
        help="Tree name. If empty, auto-detect a TTree containing required branches.",
    )
    ap.add_argument(
        "--metric",
        default="asimov",
        choices=["asimov", "soversqrtb", "soversqrtsplusb"],
        help="Optimization metric",
    )
    ap.add_argument("--npoints", type=int, default=40, help="Scan points for float variables")
    ap.add_argument("--mbbCenter", type=float, default=125.0, help="Center for mbb window scan")
    ap.add_argument("--minB", type=float, default=1e-9, help="Minimum B to evaluate significance")
    ap.add_argument(
        "--writePassingCSVs",
        action="store_true",
        help="Also write CSVs of the scan curves",
    )

    # Baseline cuts
    ap.add_argument("--minLeptons", type=int, default=0)
    ap.add_argument("--maxLeptons", type=int, default=0)

    ap.add_argument("--minMET", type=float, default=5.0)
    ap.add_argument("--minJets", type=int, default=2)
    ap.add_argument("--minBJets", type=int, default=2)
    ap.add_argument("--minRecoil", type=float, default=120.0)
    ap.add_argument("--minHT", type=float, default=220.0)
    ap.add_argument("--minMbb", type=float, default=80.0)
    ap.add_argument("--maxMbb", type=float, default=150.0)
    ap.add_argument("--maxDPhiBB", type=float, default=1.40)
    ap.add_argument("--maxDRBB", type=float, default=1.6)
    ap.add_argument("--minBBPt", type=float, default=120.0)
    ap.add_argument("--maxDPhiBBMET", type=float, default=1.7)
    ap.add_argument("--maxDPhiB1MET", type=float, default=1.1)
    ap.add_argument("--minBalance", type=float, default=0.8)

    return ap.parse_args()


def split_csv(s: str) -> List[str]:
    s = s.replace(" ", "")
    if not s:
        return []
    return [x for x in s.split(",") if x]


def should_keep_sample(
    sample: str,
    signal_key: str,
    include: List[str],
    exclude: List[str],
) -> bool:
    if sample in exclude:
        return False
    if sample == signal_key:
        return True
    if include:
        return sample in include
    return True


def find_tree_name(path: str, requested: str = "") -> str:
    with uproot.open(path) as f:
        if requested:
            if requested not in f:
                raise RuntimeError(f"{path}: tree '{requested}' not found")
            return requested

        candidates = []
        for key, obj in f.items():
            try:
                if not hasattr(obj, "keys"):
                    continue
                branches = set(obj.keys())
                if all(b in branches for b in REQUIRED_BRANCHES):
                    candidates.append(key.split(";")[0])
            except Exception:
                continue

        if not candidates:
            raise RuntimeError(
                f"{path}: could not auto-detect a tree with required branches {REQUIRED_BRANCHES}"
            )
        return candidates[0]


def load_tree_as_df(path: str, tree_name: str) -> pd.DataFrame:
    with uproot.open(path) as f:
        tree = f[tree_name]
        arr = tree.arrays(REQUIRED_BRANCHES, library="np")
    return pd.DataFrame(arr)


def collect_files(
    outdir: str,
    tag: str,
    signal_key: str,
    include: List[str],
    exclude: List[str],
) -> Tuple[List[str], List[str]]:
    pattern = os.path.join(outdir, f"{tag}_*_cutcount.root")
    files = sorted(glob.glob(pattern))

    signal_files = []
    background_files = []

    for path in files:
        base = os.path.basename(path)

        if not base.endswith("_cutcount.root"):
            continue
        if "_JESUp" in base or "_JESDown" in base or "_JERUp" in base or "_JERDown" in base:
            continue

        prefix = f"{tag}_"
        suffix = "_cutcount.root"
        if not (base.startswith(prefix) and base.endswith(suffix)):
            continue

        sample = base[len(prefix):-len(suffix)]
        if not should_keep_sample(sample, signal_key, include, exclude):
            continue

        if sample == signal_key:
            signal_files.append(path)
        else:
            background_files.append(path)

    return signal_files, background_files


def load_group(paths: List[str], tree_name: str) -> pd.DataFrame:
    frames = []
    for p in paths:
        frames.append(load_tree_as_df(p, tree_name))
    if not frames:
        return pd.DataFrame(columns=REQUIRED_BRANCHES)
    return pd.concat(frames, ignore_index=True)


def passes_minmax_series(
    x: pd.Series,
    vmin: Optional[float],
    vmax: Optional[float],
) -> pd.Series:
    mask = pd.Series(True, index=x.index)
    if vmin is not None:
        mask &= x >= vmin
    if vmax is not None:
        mask &= x <= vmax
    return mask


def build_baseline_cuts(args: argparse.Namespace) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    return {
        "Nlep": (args.minLeptons, args.maxLeptons),
        "MET": (args.minMET, None),
        "Njets": (args.minJets, None),
        "Nbjets": (args.minBJets, None),
        "recoil": (args.minRecoil, None),
        "HT": (args.minHT, None),
        "mbb": (args.minMbb, args.maxMbb),
        "dphi_bb": (None, args.maxDPhiBB),
        "dr_bb": (None, args.maxDRBB),
        "ptbb": (args.minBBPt, None),
        "dphi_bb_met": (None, args.maxDPhiBBMET),
        "dphi_bjet1_met": (None, args.maxDPhiB1MET),
        "balance": (args.minBalance, None),
    }


def build_mask(
    df: pd.DataFrame,
    cuts: Dict[str, Tuple[Optional[float], Optional[float]]],
    skip_var: Optional[str] = None,
) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)

    mask = pd.Series(True, index=df.index)
    for var, (vmin, vmax) in cuts.items():
        if var == skip_var:
            continue
        mask &= passes_minmax_series(df[var], vmin, vmax)
    return mask


def significance_asimov(s: float, b: float, min_b: float) -> float:
    if s <= 0.0 or b <= min_b:
        return 0.0
    val = 2.0 * ((s + b) * math.log(1.0 + s / b) - s)
    return math.sqrt(max(val, 0.0))


def significance_s_over_sqrt_b(s: float, b: float, min_b: float) -> float:
    if s <= 0.0 or b <= min_b:
        return 0.0
    return s / math.sqrt(b)


def significance_s_over_sqrt_s_plus_b(s: float, b: float, min_b: float) -> float:
    if s <= 0.0 or (s + b) <= min_b:
        return 0.0
    return s / math.sqrt(s + b)


def compute_metric(metric: str, s: float, b: float, min_b: float) -> float:
    if metric == "asimov":
        return significance_asimov(s, b, min_b)
    if metric == "soversqrtb":
        return significance_s_over_sqrt_b(s, b, min_b)
    return significance_s_over_sqrt_s_plus_b(s, b, min_b)


def get_scan_values_lower(x: np.ndarray, npoints: int) -> np.ndarray:
    if x.size == 0:
        return np.array([])
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([])
    lo = np.nanpercentile(x, 1.0)
    hi = np.nanpercentile(x, 99.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        vals = np.unique(np.sort(x))
        return vals[: min(len(vals), npoints)]
    return np.unique(np.linspace(lo, hi, npoints))


def get_scan_values_upper(x: np.ndarray, npoints: int) -> np.ndarray:
    return get_scan_values_lower(x, npoints)


def get_scan_values_int_lower(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=int)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=int)
    vals = np.unique(np.asarray(x, dtype=int))
    return np.sort(vals)


def get_scan_values_mbb_window(x: np.ndarray, center: float, npoints: int) -> np.ndarray:
    if x.size == 0:
        return np.array([])
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([])
    distances = np.abs(x - center)
    hi = np.nanpercentile(distances, 95.0)
    lo = max(5.0, np.nanpercentile(distances, 5.0))
    if not np.isfinite(hi) or hi <= lo:
        hi = max(lo + 1.0, 40.0)
    return np.unique(np.linspace(lo, hi, npoints))


def weighted_yield(df: pd.DataFrame, mask: np.ndarray) -> float:
    if df.empty or mask.size == 0:
        return 0.0
    return float(df.loc[mask, "weight"].sum())


def scan_variable(
    s_df: pd.DataFrame,
    b_df: pd.DataFrame,
    cuts: Dict[str, Tuple[Optional[float], Optional[float]]],
    spec: VarSpec,
    metric: str,
    npoints: int,
    min_b: float,
) -> pd.DataFrame:
    s_base = build_mask(s_df, cuts, skip_var=spec.name)
    b_base = build_mask(b_df, cuts, skip_var=spec.name)

    s_x = s_df.loc[s_base, spec.name].to_numpy(dtype=float)
    b_x = b_df.loc[b_base, spec.name].to_numpy(dtype=float)
    all_x = np.concatenate([s_x, b_x]) if (s_x.size + b_x.size) > 0 else np.array([])

    if spec.mode == "lower":
        scan_vals = get_scan_values_lower(all_x, npoints)
    elif spec.mode == "upper":
        scan_vals = get_scan_values_upper(all_x, npoints)
    elif spec.mode == "int_lower":
        scan_vals = get_scan_values_int_lower(all_x)
    elif spec.mode == "window":
        if spec.center is None:
            raise ValueError(f"Window scan for {spec.name} requires a center")
        scan_vals = get_scan_values_mbb_window(all_x, spec.center, npoints)
    else:
        raise ValueError(f"Unknown scan mode: {spec.mode}")

    rows = []
    for cut_val in scan_vals:
        if spec.mode in ("lower", "int_lower"):
            s_mask = s_base & (s_df[spec.name].to_numpy() >= cut_val)
            b_mask = b_base & (b_df[spec.name].to_numpy() >= cut_val)
            cut_text = f"{spec.name} >= {cut_val:.6g}"
            cut_low = float(cut_val)
            cut_high = np.nan
            plot_x = float(cut_val)

        elif spec.mode == "upper":
            s_mask = s_base & (s_df[spec.name].to_numpy() <= cut_val)
            b_mask = b_base & (b_df[spec.name].to_numpy() <= cut_val)
            cut_text = f"{spec.name} <= {cut_val:.6g}"
            cut_low = np.nan
            cut_high = float(cut_val)
            plot_x = float(cut_val)

        elif spec.mode == "window":
            low = spec.center - cut_val
            high = spec.center + cut_val
            s_mask = s_base & (s_df[spec.name].to_numpy() >= low) & (s_df[spec.name].to_numpy() <= high)
            b_mask = b_base & (b_df[spec.name].to_numpy() >= low) & (b_df[spec.name].to_numpy() <= high)
            cut_text = f"{low:.6g} <= {spec.name} <= {high:.6g}"
            cut_low = float(low)
            cut_high = float(high)
            plot_x = float(cut_val)

        else:
            raise ValueError(f"Unsupported mode: {spec.mode}")

        s_y = weighted_yield(s_df, s_mask)
        b_y = weighted_yield(b_df, b_mask)
        z_asimov = compute_metric("asimov", s_y, b_y, min_b)
        z_sb = compute_metric("soversqrtb", s_y, b_y, min_b)
        z_ssb = compute_metric("soversqrtsplusb", s_y, b_y, min_b)
        z_main = compute_metric(metric, s_y, b_y, min_b)

        rows.append(
            {
                "variable": spec.name,
                "mode": spec.mode,
                "plot_x": plot_x,
                "cut_value": float(cut_val),
                "cut_low": cut_low,
                "cut_high": cut_high,
                "cut_text": cut_text,
                "S": s_y,
                "B": b_y,
                "S_over_sqrtB": z_sb,
                "S_over_sqrtSplusB": z_ssb,
                "Asimov": z_asimov,
                "metric_value": z_main,
            }
        )

    return pd.DataFrame(rows)


def baseline_lines_for_plot(
    spec: VarSpec,
    cuts: Dict[str, Tuple[Optional[float], Optional[float]]],
) -> Tuple[Optional[float], Optional[float], str, str]:
    vmin, vmax = cuts.get(spec.name, (None, None))

    if spec.mode in ("lower", "int_lower"):
        return vmin, None, "Baseline min", ""
    if spec.mode == "upper":
        return None, vmax, "", "Baseline max"
    if spec.mode == "window":
        if spec.center is None or vmin is None or vmax is None:
            return None, None, "", ""
        width = 0.5 * (float(vmax) - float(vmin))
        return float(width), None, "Baseline window half-width", ""
    return None, None, "", ""


def baseline_text_for_box(
    spec: VarSpec,
    cuts: Dict[str, Tuple[Optional[float], Optional[float]]],
) -> str:
    vmin, vmax = cuts.get(spec.name, (None, None))

    if spec.mode in ("lower", "int_lower"):
        if vmin is None:
            return "baseline: none"
        return f"baseline: {spec.name} >= {vmin:.6g}"

    if spec.mode == "upper":
        if vmax is None:
            return "baseline: none"
        return f"baseline: {spec.name} <= {vmax:.6g}"

    if spec.mode == "window":
        if vmin is None or vmax is None:
            return "baseline: none"
        return f"baseline: {vmin:.6g} <= {spec.name} <= {vmax:.6g}"

    return "baseline: none"


def metric_column_name(metric: str) -> str:
    return {
        "asimov": "Asimov",
        "soversqrtb": "S_over_sqrtB",
        "soversqrtsplusb": "S_over_sqrtSplusB",
    }[metric]


def plot_scan(
    df: pd.DataFrame,
    spec: VarSpec,
    cuts: Dict[str, Tuple[Optional[float], Optional[float]]],
    out_png: str,
    metric: str,
) -> None:
    if df.empty:
        return

    metric_col = metric_column_name(metric)

    x = df["plot_x"].to_numpy(dtype=float)
    y = df[metric_col].to_numpy(dtype=float)

    best_idx = int(np.nanargmax(y))
    best_row = df.iloc[best_idx]
    best_x = float(x[best_idx])
    best_y = float(y[best_idx])
    best_text = str(best_row["cut_text"])

    plt.figure(figsize=(7.4, 5.4))
    plt.plot(x, y, marker="o", label=metric_col)

    plt.axvline(best_x, linestyle="--", linewidth=1.5, label=f"Best: {best_text}")

    base_min_x, base_max_x, base_min_label, base_max_label = baseline_lines_for_plot(spec, cuts)
    if base_min_x is not None:
        plt.axvline(
            base_min_x,
            linestyle=":",
            linewidth=1.5,
            label=f"{base_min_label}: {base_min_x:.6g}",
        )
    if base_max_x is not None:
        plt.axvline(
            base_max_x,
            linestyle=":",
            linewidth=1.5,
            label=f"{base_max_label}: {base_max_x:.6g}",
        )

    if spec.mode == "window":
        xlabel = f"Half-width of {spec.label} window around {spec.center:g}"
    elif spec.mode == "upper":
        xlabel = f"Upper cut on {spec.label}"
    else:
        xlabel = f"Lower cut on {spec.label}"

    baseline_text = baseline_text_for_box(spec, cuts)

    textbox_lines = [
        f"best: {best_text}",
        baseline_text,
        f"S = {best_row['S']:.6g}",
        f"B = {best_row['B']:.6g}",
        f"{metric_col} = {best_y:.6g}",
    ]
    textbox = "\n".join(textbox_lines)

    plt.xlabel(xlabel)
    plt.ylabel(metric_col)
    plt.title(f"1D optimization: {spec.label}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    plt.text(
        0.98,
        0.98,
        textbox,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def make_best_summary_row(
    scan_df: pd.DataFrame,
    spec: VarSpec,
    metric: str,
) -> Dict[str, object]:
    metric_col = metric_column_name(metric)

    best_idx = scan_df[metric_col].idxmax()
    row = scan_df.loc[best_idx]

    out = {
        "variable": spec.name,
        "mode": spec.mode,
        "best_cut_text": row["cut_text"],
        "best_cut_value": row["cut_value"],
        "best_cut_low": row["cut_low"],
        "best_cut_high": row["cut_high"],
        "best_S": row["S"],
        "best_B": row["B"],
        "best_S_over_sqrtB": row["S_over_sqrtB"],
        "best_S_over_sqrtSplusB": row["S_over_sqrtSplusB"],
        "best_Asimov": row["Asimov"],
    }
    return out


def main() -> None:
    args = parse_args()

    include = split_csv(args.include)
    exclude = split_csv(args.exclude)

    signal_files, background_files = collect_files(
        outdir=args.outdir,
        tag=args.tag,
        signal_key=args.signal,
        include=include,
        exclude=exclude,
    )

    if not signal_files:
        raise SystemExit(f"No signal files found in {args.outdir}")
    if not background_files:
        raise SystemExit(f"No background files found in {args.outdir}")

    tree_name = find_tree_name(signal_files[0], requested=args.tree)
    print(f"[INFO] Using tree: {tree_name}")
    print(f"[INFO] Signal files: {len(signal_files)}")
    print(f"[INFO] Background files: {len(background_files)}")

    s_df = load_group(signal_files, tree_name)
    b_df = load_group(background_files, tree_name)

    print(f"[INFO] Signal entries: {len(s_df)}")
    print(f"[INFO] Background entries: {len(b_df)}")
    print(f"[INFO] Signal weighted sum: {s_df['weight'].sum():.6g}")
    print(f"[INFO] Background weighted sum: {b_df['weight'].sum():.6g}")

    cuts = build_baseline_cuts(args)

    specs = [
        VarSpec("MET", "lower", "MET"),
        VarSpec("Njets", "int_lower", "Njets"),
        VarSpec("Nbjets", "int_lower", "Nbjets"),
        VarSpec("recoil", "lower", "recoil"),
        VarSpec("HT", "lower", "HT"),
        VarSpec("mbb", "window", "mbb", center=args.mbbCenter),
        VarSpec("dphi_bb", "upper", "dphi_bb"),
        VarSpec("dr_bb", "upper", "dr_bb"),
        VarSpec("ptbb", "lower", "ptbb"),
        VarSpec("dphi_bb_met", "upper", "dphi_bb_met"),
        VarSpec("dphi_bjet1_met", "upper", "dphi_bjet1_met"),
        VarSpec("balance", "lower", "balance"),
    ]

    out_opt_dir = os.path.join(args.outdir, "cut_optimization")
    os.makedirs(out_opt_dir, exist_ok=True)

    best_rows = []
    for spec in specs:
        print(f"[INFO] Scanning {spec.name} ...")
        scan_df = scan_variable(
            s_df=s_df,
            b_df=b_df,
            cuts=cuts,
            spec=spec,
            metric=args.metric,
            npoints=args.npoints,
            min_b=args.minB,
        )

        if scan_df.empty:
            print(f"[WARN] No scan points for {spec.name}")
            continue

        best_row = make_best_summary_row(scan_df, spec, args.metric)
        best_rows.append(best_row)

        print(
            f"[BEST] {spec.name}: {best_row['best_cut_text']} | "
            f"S={best_row['best_S']:.6g} "
            f"B={best_row['best_B']:.6g} "
            f"Asimov={best_row['best_Asimov']:.6g} "
            f"S/sqrt(B)={best_row['best_S_over_sqrtB']:.6g}"
        )

        if args.writePassingCSVs:
            csv_path = os.path.join(out_opt_dir, f"scan_{spec.name}.csv")
            scan_df.to_csv(csv_path, index=False)

        png_path = os.path.join(out_opt_dir, f"scan_{spec.name}.png")
        plot_scan(
            scan_df,
            spec,
            cuts,
            png_path,
            args.metric,
        )

    best_df = pd.DataFrame(best_rows)
    if best_df.empty:
        raise SystemExit("No optimization results produced.")

    summary_csv = os.path.join(out_opt_dir, "best_cuts_summary.csv")
    best_df.to_csv(summary_csv, index=False)

    print()
    print(f"[DONE] Wrote optimization summary: {summary_csv}")
    print(f"[DONE] Plots and per-variable outputs in: {out_opt_dir}")


if __name__ == "__main__":
    main()
