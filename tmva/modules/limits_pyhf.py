#!/usr/bin/env python3
"""
limits_pyhf.py (SR + CR_top) with:
- POI: mu (signal normfactor)
- CR-constrained norms: mu_tt for tt_*, mu_st for st_*
- MC stat per bin: staterror from sumw2 (nominal)
- Correlated norms: lumi, btag (normsys)
- Shape systematics: JES/JER as histosys using *_withBDT_{SYST}{Up,Down}.root
  -> Uncorrelated per process:
     histosys name = "{SYST}_{sample}" (same across SR/CR for that sample)

ROBUSTNESS / SPEED:
A) Optional adaptive binning: merge fine bins so every SR background bin has >= BMIN.
B) Prune bins where total expectation is zero in BOTH SR and CR (keep SR signal bins).
C) IMPORTANT: do NOT apply staterror floor to bins where nominal==0.
D) Try pyhf upper_limit; if it fails or NaNs, fall back to manual CLs scan.
E) Cache arrays per (file, cut) so we don't reopen/read ROOT repeatedly.
F) Option to drop signal from CR (common in real analyses): --no-signal-in-cr
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pyhf
import uproot

pyhf.set_backend("numpy")


# -------------------------
# Cut eval: ROOT style -> numpy boolean mask
# -------------------------
def normalize_cut_to_numpy_expr(cut: str) -> str:
    if cut is None:
        return "True"
    s = str(cut).strip()
    if s == "":
        return "True"

    # strip one outer pair if it wraps whole expression
    if s.startswith("(") and s.endswith(")"):
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
        if ok:
            s = s[1:-1].strip()

    parts = re.split(r"(\&\&|\|\|)", s)
    parts = [p.strip() for p in parts if p.strip() != ""]

    out = []
    for p in parts:
        if p == "&&":
            out.append("&")
        elif p == "||":
            out.append("|")
        else:
            out.append(f"({p})")

    return "(" + " ".join(out) + ")"


def branches_in_cut(cut: str) -> List[str]:
    if not cut:
        return []
    found = re.findall(r"\b[A-Za-z_]\w*\b", cut)
    bad = {"and", "or", "not", "True", "False"}
    return sorted(set([x for x in found if x not in bad]))


def eval_cut_mask(arr: dict, cut: str) -> np.ndarray:
    if cut is None or str(cut).strip() == "":
        n = len(next(iter(arr.values())))
        return np.ones(n, dtype=bool)

    expr = normalize_cut_to_numpy_expr(cut)
    local = {k: arr[k] for k in arr.keys()}

    try:
        mask = eval(expr, {"__builtins__": {}}, local)
    except Exception as e:
        raise RuntimeError(
            "Failed to evaluate cut:\n"
            f"  cut='{cut}'\n"
            f"  expr='{expr}'\n"
            f"Error: {e}"
        )
    return np.asarray(mask, dtype=bool)


# -------------------------
# Adaptive binning (merge fine bins)
# -------------------------
def make_adaptive_edges_from_bkg(
    sumw_bkg_fine: np.ndarray,
    fine_edges: np.ndarray,
    bmin: float = 20.0,
    min_bins: int = 4,
) -> np.ndarray:
    """
    Merge fine bins into variable-width bins so each merged bin has >= bmin background.
    Merge from high-score to low-score (important for BDT tails).

    Returns sorted unique edges (including xlow and xhigh).
    Ensures at least min_bins total bins by relaxing bmin if needed.
    """
    sumw = np.asarray(sumw_bkg_fine, dtype=float)
    edges = np.asarray(fine_edges, dtype=float)
    assert len(edges) == len(sumw) + 1

    if bmin <= 0:
        return edges

    # We may need to relax bmin if too few bins are formed.
    bmin_try = float(bmin)

    for _ in range(6):  # few relax rounds
        new_edges = [edges[-1]]  # start from xhigh
        acc = 0.0

        for i in range(len(sumw) - 1, -1, -1):
            acc += sumw[i]
            if acc >= bmin_try:
                new_edges.append(edges[i])
                acc = 0.0

        if new_edges[-1] != edges[0]:
            new_edges.append(edges[0])

        new_edges = np.array(sorted(set(new_edges)), dtype=float)
        nbins = len(new_edges) - 1
        if nbins >= min_bins:
            return new_edges

        # relax requirement if it created too few bins
        bmin_try *= 0.5

    return np.array(sorted(set([edges[0], edges[-1]])), dtype=float)


# -------------------------
# Fast ROOT reading cache
# -------------------------
@dataclass
class CachedArrays:
    x: np.ndarray
    w: np.ndarray
    cut_vars: Dict[str, np.ndarray]


class RootCache:
    """
    Cache arrays read from ROOT for (root_path, tree, cut, bdt, weight).
    This eliminates huge repeated I/O for nominal and every syst variation.
    """
    def __init__(self):
        self._cache: Dict[Tuple[str, str, str, str, str], CachedArrays] = {}

    def get_arrays(
        self,
        root_path: str,
        tree: str,
        cut: str,
        bdt_branch: str,
        weight_branch: str,
    ) -> CachedArrays:
        key = (root_path, tree, cut or "", bdt_branch, weight_branch)
        if key in self._cache:
            return self._cache[key]

        with uproot.open(root_path) as f:
            t = f[tree]
            need = [bdt_branch, weight_branch]
            cut_br = [b for b in branches_in_cut(cut) if b not in need]
            existing = set(t.keys())
            cut_br = [b for b in cut_br if b in existing]
            need += cut_br

            arr = t.arrays(need, library="np")
            x = arr[bdt_branch].astype(np.float64)
            w = arr[weight_branch].astype(np.float64)
            cut_vars = {k: arr[k] for k in cut_br}

        obj = CachedArrays(x=x, w=w, cut_vars=cut_vars)
        self._cache[key] = obj
        return obj


def hist_sumw2_from_cached(
    cached: CachedArrays,
    cut: str,
    edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # build eval dict for cut
    local = {**cached.cut_vars, "__x__": cached.x, "__w__": cached.w}
    # but eval_cut_mask expects dict-like with all branches
    # create minimal dict:
    eval_dict = dict(cached.cut_vars)
    eval_dict["_bdt_"] = cached.x
    eval_dict["_w_"] = cached.w

    # Evaluate mask using original branch names in expression:
    # We need the actual arrays for names used in cut;
    # eval_cut_mask uses arr keys directly.
    # So provide ALL branches it might reference:
    arr_for_eval = {**cached.cut_vars}
    # Add also bdt/weight in case user uses them in cut (rare but possible)
    arr_for_eval["_bdt_"] = cached.x
    arr_for_eval["_w_"] = cached.w

    # Replace if user literally wrote bdt/weight in cut? (usually not)
    # We'll just evaluate with the original arrays in dict; if cut does not include them, ok.

    m = eval_cut_mask({**cached.cut_vars, **{"_bdt_": cached.x, "_w_": cached.w}}, cut)
    x = cached.x[m]
    w = cached.w[m]

    edges = np.asarray(edges, dtype=float)
    sumw, _ = np.histogram(x, bins=edges, weights=w)
    sumw2, _ = np.histogram(x, bins=edges, weights=w * w)

    sumw = np.maximum(sumw, 0.0)  # pyhf requires non-negative templates
    sumw2 = np.maximum(sumw2, 0.0)
    return sumw.astype(float), sumw2.astype(float)


# -------------------------
# pyhf modifiers
# -------------------------
def normsys(name, rel):
    rel = float(rel)
    lo = 1.0 - rel
    hi = 1.0 + rel
    if lo <= 0:
        raise ValueError(f"normsys {name}: lo={lo} <= 0 (rel={rel}) not allowed")
    return {"name": name, "type": "normsys", "data": {"lo": lo, "hi": hi}}


def staterror(name, err):
    return {"name": name, "type": "staterror", "data": err}


def histosys(name, lo_data, hi_data):
    return {"name": name, "type": "histosys", "data": {"lo_data": lo_data, "hi_data": hi_data}}


# -------------------------
# Bin pruning utilities
# -------------------------
def _apply_binmask_to_sample(sample_entry, keep_mask):
    sample_entry["data"] = (np.asarray(sample_entry["data"], dtype=float)[keep_mask]).tolist()

    for mod in sample_entry["modifiers"]:
        if mod["type"] == "staterror":
            mod["data"] = (np.asarray(mod["data"], dtype=float)[keep_mask]).tolist()
        elif mod["type"] == "histosys":
            lo = np.asarray(mod["data"]["lo_data"], dtype=float)[keep_mask].tolist()
            hi = np.asarray(mod["data"]["hi_data"], dtype=float)[keep_mask].tolist()
            mod["data"]["lo_data"] = lo
            mod["data"]["hi_data"] = hi
    return sample_entry


def prune_empty_bins(templates, keep_if_signal=True):
    sr_tot = None
    cr_tot = None
    sr_sig = None

    for sname, s in templates["SR"].items():
        v = np.asarray(s["data"], dtype=float)
        sr_tot = v if sr_tot is None else (sr_tot + v)
        if sname == "signal":
            sr_sig = v.copy()

    for _, s in templates["CR_top"].items():
        v = np.asarray(s["data"], dtype=float)
        cr_tot = v if cr_tot is None else (cr_tot + v)

    if sr_tot is None or cr_tot is None or sr_sig is None:
        raise RuntimeError("Internal error: missing SR/CR totals or SR signal")

    keep = (sr_tot > 0) | (cr_tot > 0)
    if keep_if_signal:
        keep = keep | (sr_sig > 0)

    keep = np.asarray(keep, dtype=bool)
    pruned = int(np.sum(~keep))
    return keep, pruned


# -------------------------
# Model spec builder
# -------------------------
def build_spec_sr_cr(templates, nbins, mu_bounds, mu_init, tt_bounds, st_bounds):
    channels = []
    for ch_name, samples_dict in templates.items():
        ch = {"name": ch_name, "samples": []}
        for sname, s in samples_dict.items():
            ch["samples"].append({"name": sname, "data": s["data"], "modifiers": s["modifiers"]})
        channels.append(ch)

    spec = {
        "channels": channels,
        "observations": [{"name": ch["name"], "data": [0.0] * nbins} for ch in channels],
        "measurements": [
            {
                "name": "meas",
                "config": {
                    "poi": "mu",
                    "parameters": [
                        {"name": "mu", "inits": [float(mu_init)], "bounds": [[float(mu_bounds[0]), float(mu_bounds[1])]]},
                        {"name": "mu_tt", "inits": [1.0], "bounds": [[float(tt_bounds[0]), float(tt_bounds[1])]]},
                        {"name": "mu_st", "inits": [1.0], "bounds": [[float(st_bounds[0]), float(st_bounds[1])]]},
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }
    return spec


# -------------------------
# Limits: try upper_limit; fallback to scan
# -------------------------
def expected_limit_asimov_try_upper_limit(model, level=0.95, test_stat="qtilde"):
    init = model.config.suggested_init()
    parnames = model.config.par_names

    init[model.config.poi_index] = 0.0
    for nm in ["mu_tt", "mu_st"]:
        if nm in parnames:
            init[parnames.index(nm)] = 1.0

    data = model.expected_data(init)
    if not np.all(np.isfinite(np.asarray(data, dtype=float))):
        raise RuntimeError("Asimov data contains NaN/Inf. Check templates/modifiers.")

    obs_ul, exp_ul = pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, level=level, test_stat=test_stat
    )
    return float(obs_ul), [float(x) for x in exp_ul]


def expected_limit_asimov_scan(model, mu_max, level=0.95, test_stat="qtilde", nscan=80):
    alpha = 1.0 - float(level)

    init = model.config.suggested_init()
    parnames = model.config.par_names
    init[model.config.poi_index] = 0.0
    for nm in ["mu_tt", "mu_st"]:
        if nm in parnames:
            init[parnames.index(nm)] = 1.0
    data = model.expected_data(init)

    mus = np.concatenate([
        np.linspace(0.0, min(5.0, mu_max), min(30, max(10, nscan // 2)), endpoint=True),
        np.linspace(min(5.0, mu_max), mu_max, max(10, nscan - min(30, max(10, nscan // 2))), endpoint=True),
    ])
    mus = np.unique(np.clip(mus, 0.0, mu_max))

    cls_vals = []
    exp_sets = []

    for mu in mus:
        try:
            cls_obs, cls_exp = pyhf.infer.hypotest(
                mu, data, model,
                test_stat=test_stat,
                return_expected_set=True
            )
            cls_obs = float(cls_obs)
            cls_exp = [float(x) for x in cls_exp]
        except Exception:
            cls_obs = float("nan")
            cls_exp = [float("nan")] * 5

        cls_vals.append(cls_obs)
        exp_sets.append(cls_exp)

    cls_vals = np.asarray(cls_vals, dtype=float)
    exp_sets = np.asarray(exp_sets, dtype=float)

    good = np.isfinite(cls_vals)
    mus_g = mus[good]
    cls_g = cls_vals[good]
    exp_g = exp_sets[good, :]

    if mus_g.size < 5:
        raise RuntimeError("Too many NaNs in CLs scan. Check model/systematics.")

    idx = np.where(cls_g <= alpha)[0]
    if idx.size == 0:
        raise RuntimeError(f"Did not reach CLs={alpha:.3g} by mu_max={mu_max}. Increase --mu-max.")

    i1 = idx[0]
    if i1 == 0:
        mu_ul = float(mus_g[0])
        exp_ul = exp_g[0, :]
        return mu_ul, exp_ul.tolist()

    i0 = i1 - 1
    x0, y0 = float(mus_g[i0]), float(cls_g[i0])
    x1, y1 = float(mus_g[i1]), float(cls_g[i1])
    if y1 == y0:
        mu_ul = x1
    else:
        mu_ul = x0 + (alpha - y0) * (x1 - x0) / (y1 - y0)

    j = int(np.argmin(np.abs(mus_g - mu_ul)))
    exp_ul = exp_g[j, :]

    return float(mu_ul), [float(x) for x in exp_ul]


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--workdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tree", default="events")

    ap.add_argument("--method", required=True, choices=["BDT", "BDTG"])
    ap.add_argument("--bdt-branch", default=None)
    ap.add_argument("--weight-branch", default="weight")

    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--xlow", type=float, default=-1.0)
    ap.add_argument("--xhigh", type=float, default=1.0)

    ap.add_argument("--sr-cut", default="(isSR==1 && Nlep==0)")
    ap.add_argument("--cr-cut", default="(isCR_top==1)")

    ap.add_argument("--mu-max", type=float, default=1000.0)
    ap.add_argument("--mu-init", type=float, default=1.0)

    ap.add_argument("--lumi-unc", type=float, default=0.016)
    ap.add_argument("--btag-unc", type=float, default=0.05)

    ap.add_argument("--shape-systs", default="JES,JER")
    ap.add_argument("--theory-signal", type=float, default=0.10)
    ap.add_argument("--theory-tt", type=float, default=0.10)
    ap.add_argument("--theory-st", type=float, default=0.15)
    ap.add_argument("--theory-other", type=float, default=0.20)

    # robustness / speed knobs
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--test-stat", default="qtilde", choices=["qtilde", "qmu"])
    ap.add_argument("--stat-floor-rel", type=float, default=0.0)

    ap.add_argument("--fallback-scan", action="store_true")
    ap.add_argument("--scan-points", type=int, default=80)

    # NEW: adaptive binning
    ap.add_argument("--adaptive-bins", action="store_true",
                    help="Use adaptive variable-width bins built from SR background to avoid empty tail bins.")
    ap.add_argument("--fine-bins", type=int, default=200,
                    help="Number of fine bins used to build adaptive binning (only if --adaptive-bins).")
    ap.add_argument("--bmin", type=float, default=20.0,
                    help="Minimum SR background sumw per merged bin (only if --adaptive-bins).")
    ap.add_argument("--min-bins", type=int, default=4,
                    help="Minimum number of bins after merging (only if --adaptive-bins).")

    # NEW: physics-standard speed
    ap.add_argument("--no-signal-in-cr", action="store_true",
                    help="Set signal template in CR to zeros (common in ATLAS/CMS unless signal contamination matters).")

    args = ap.parse_args()

    workdir = os.path.expanduser(args.workdir)
    bdt_branch = args.bdt_branch or f"bdt_{args.method}"

    env_bkgs = os.environ.get("BKG_SAMPLES", "").strip()
    if env_bkgs:
        bkg_names = env_bkgs.split()
    else:
        bkg_names = [
            "tt_semilep", "tt_dilep", "tt_had",
            "st_tW", "st_tch_top", "st_tch_tbar",
            "wjets", "zvvjets",
        ]

    tt_samples = [n for n in bkg_names if n.startswith("tt_")]
    st_samples = [n for n in bkg_names if n.startswith("st_")]

    def withbdt_path(sample, syst=None):
        if syst is None:
            return os.path.join(workdir, f"{args.tag}_{sample}_withBDT.root")
        return os.path.join(workdir, f"{args.tag}_{sample}_withBDT_{syst}.root")

    # Validate nominal exists
    for s in ["signal"] + bkg_names:
        fp = withbdt_path(s, None)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing nominal withBDT: {fp}")

    shape_systs = [x.strip() for x in args.shape_systs.split(",") if x.strip()]

    common_norms = [normsys("lumi", args.lumi_unc), normsys("btag", args.btag_unc)]

    theory_map = {"signal": float(args.theory_signal)}
    for n in tt_samples:
        theory_map[n] = float(args.theory_tt)
    for n in st_samples:
        theory_map[n] = float(args.theory_st)
    for n in bkg_names:
        if n not in theory_map:
            theory_map[n] = float(args.theory_other)

    # ------------------------------------------------------------
    # Build edges: either fixed bins or adaptive bins (SR bkg-driven)
    # ------------------------------------------------------------
    cache = RootCache()

    if args.adaptive_bins:
        fine_edges = np.linspace(args.xlow, args.xhigh, int(args.fine_bins) + 1)

        # build SR background total on fine bins (nominal only)
        bkg_total_fine = np.zeros(len(fine_edges) - 1, dtype=float)

        for b in bkg_names:
            fp = withbdt_path(b, None)
            cached = cache.get_arrays(fp, args.tree, args.sr_cut, bdt_branch, args.weight_branch)
            sumw_b, _ = hist_sumw2_from_cached(cached, args.sr_cut, fine_edges)
            bkg_total_fine += sumw_b

        edges = make_adaptive_edges_from_bkg(
            bkg_total_fine, fine_edges, bmin=float(args.bmin), min_bins=int(args.min_bins)
        )
        if len(edges) < 3:
            raise RuntimeError("Adaptive binning produced <2 bins. Decrease --bmin or increase --fine-bins.")
    else:
        edges = np.linspace(args.xlow, args.xhigh, int(args.bins) + 1)

    nbins_nominal = len(edges) - 1

    # ------------------------------------------------------------
    # Template builder
    # ------------------------------------------------------------
    templates: Dict[str, Dict[str, dict]] = {"SR": {}, "CR_top": {}}

    def make_sample_entry(
        ch_name: str,
        sample_name: str,
        nom_sumw: np.ndarray,
        nom_sumw2: np.ndarray,
        extra_mods: List[dict],
        cut: str,
        syst_hists: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        nom = np.maximum(nom_sumw, 0.0).astype(float)

        # staterror from sumw2 (no floor for nom==0)
        err = np.sqrt(np.maximum(nom_sumw2, 0.0)).astype(float)
        if args.stat_floor_rel and float(args.stat_floor_rel) > 0:
            rel_floor = float(args.stat_floor_rel)
            floor = rel_floor * nom
            err = np.where(nom > 0, np.maximum(err, floor), 0.0)

        mods = [staterror(f"stat_{ch_name}_{sample_name}", err.tolist())]
        mods += common_norms

        th = float(theory_map.get(sample_name, 0.0))
        if th > 0:
            mods.append(normsys(f"theory_{sample_name}", th))

        # shape systematics (precomputed, to avoid reopening files repeatedly)
        if syst_hists:
            for syst, (dn_sumw, up_sumw) in syst_hists.items():
                dn = np.maximum(dn_sumw, 0.0).astype(float).tolist()
                up = np.maximum(up_sumw, 0.0).astype(float).tolist()
                mods.append(histosys(f"{syst}_{sample_name}", lo_data=dn, hi_data=up))

        mods += extra_mods
        return {"data": nom.tolist(), "modifiers": mods}

    def build_syst_hists(sample_name: str, cut: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        out = {}
        for syst in shape_systs:
            fp_up = withbdt_path(sample_name, f"{syst}Up")
            fp_dn = withbdt_path(sample_name, f"{syst}Down")
            if os.path.exists(fp_up) and os.path.exists(fp_dn):
                cached_up = cache.get_arrays(fp_up, args.tree, cut, bdt_branch, args.weight_branch)
                cached_dn = cache.get_arrays(fp_dn, args.tree, cut, bdt_branch, args.weight_branch)
                up_sumw, _ = hist_sumw2_from_cached(cached_up, cut, edges)
                dn_sumw, _ = hist_sumw2_from_cached(cached_dn, cut, edges)
                out[syst] = (dn_sumw, up_sumw)
        return out

    # Fill SR and CR
    for ch_name, cut in [("SR", args.sr_cut), ("CR_top", args.cr_cut)]:
        # Signal
        fp_sig = withbdt_path("signal", None)
        cached_sig = cache.get_arrays(fp_sig, args.tree, cut, bdt_branch, args.weight_branch)
        s_nom, s_w2 = hist_sumw2_from_cached(cached_sig, cut, edges)

        if (ch_name == "CR_top") and args.no_signal_in_cr:
            s_nom = np.zeros_like(s_nom)
            s_w2 = np.zeros_like(s_w2)

        sig_syst = build_syst_hists("signal", cut) if (not (ch_name == "CR_top" and args.no_signal_in_cr)) else {}

        templates[ch_name]["signal"] = make_sample_entry(
            ch_name, "signal", s_nom, s_w2,
            extra_mods=[{"name": "mu", "type": "normfactor", "data": None}],
            cut=cut,
            syst_hists=sig_syst,
        )

        # Backgrounds
        for b in bkg_names:
            fp_b = withbdt_path(b, None)
            cached_b = cache.get_arrays(fp_b, args.tree, cut, bdt_branch, args.weight_branch)
            b_nom, b_w2 = hist_sumw2_from_cached(cached_b, cut, edges)

            extra = []
            if b in tt_samples:
                extra.append({"name": "mu_tt", "type": "normfactor", "data": None})
            if b in st_samples:
                extra.append({"name": "mu_st", "type": "normfactor", "data": None})

            b_syst = build_syst_hists(b, cut)

            templates[ch_name][b] = make_sample_entry(
                ch_name, b, b_nom, b_w2,
                extra_mods=extra,
                cut=cut,
                syst_hists=b_syst,
            )

    # Prune bins
    keep_mask = np.ones(nbins_nominal, dtype=bool)
    pruned = 0
    if not args.no_prune:
        keep_mask, pruned = prune_empty_bins(templates, keep_if_signal=True)
        if pruned > 0:
            for ch in templates:
                for s in templates[ch]:
                    templates[ch][s] = _apply_binmask_to_sample(templates[ch][s], keep_mask)

    nbins_eff = int(np.sum(keep_mask))

    # Checks
    s_sr = float(np.sum(np.asarray(templates["SR"]["signal"]["data"], dtype=float)))
    b_sr = float(sum(np.sum(np.asarray(templates["SR"][b]["data"], dtype=float)) for b in bkg_names))
    s_cr = float(np.sum(np.asarray(templates["CR_top"]["signal"]["data"], dtype=float)))
    b_cr = float(sum(np.sum(np.asarray(templates["CR_top"][b]["data"], dtype=float)) for b in bkg_names))

    print(f"[CHECK] bins: nominal={nbins_nominal}, kept={nbins_eff}, pruned={pruned}")
    print(f"[CHECK] SR totals: S={s_sr:.6g}  B={b_sr:.6g}")
    print(f"[CHECK] CR totals: S={s_cr:.6g}  B={b_cr:.6g}")

    if nbins_eff == 0:
        raise RuntimeError("All bins pruned. Loosen cuts or change binning.")
    if s_sr <= 0:
        raise RuntimeError("Signal yield in SR is zero. Cannot set a limit.")
    if b_sr <= 0:
        raise RuntimeError("Background yield in SR is zero. Limit ill-defined.")

    # Build model
    spec = build_spec_sr_cr(
        templates, nbins_eff,
        mu_bounds=(0.0, float(args.mu_max)),
        mu_init=float(args.mu_init),
        tt_bounds=(0.0, 5.0),
        st_bounds=(0.0, 5.0),
    )

    ws = pyhf.Workspace(spec)
    model = ws.model(measurement_name="meas")

    # Compute expected limit (Asimov)
    use_scan = False
    try:
        obs_ul, exp_ul = expected_limit_asimov_try_upper_limit(
            model, level=0.95, test_stat=args.test_stat
        )

        if (not np.isfinite(obs_ul)) or (not np.all(np.isfinite(np.asarray(exp_ul, dtype=float)))):
            raise RuntimeError("upper_limit returned NaN/Inf")

    except Exception as e:
        if not args.fallback_scan:
            raise
        print(f"[WARN] pyhf.upper_limit failed ({e}). Falling back to manual CLs scan...")
        use_scan = True
        obs_ul, exp_ul = expected_limit_asimov_scan(
            model,
            mu_max=float(args.mu_max),
            level=0.95,
            test_stat=args.test_stat,
            nscan=int(args.scan_points),
        )

    result = {
        "method": args.method,
        "channels": ["SR", "CR_top"],
        "bins_nominal": int(nbins_nominal),
        "bins_kept": int(nbins_eff),
        "bins_pruned": int(pruned),
        "edges_nominal": edges.tolist(),
        "keep_mask": keep_mask.astype(int).tolist(),
        "bdt_branch": bdt_branch,
        "weight_branch": args.weight_branch,
        "sr_cut": args.sr_cut,
        "cr_cut": args.cr_cut,
        "options": {
            "test_stat": args.test_stat,
            "stat_floor_rel": float(args.stat_floor_rel),
            "fallback_scan_used": bool(use_scan),
            "scan_points": int(args.scan_points),
            "adaptive_bins": bool(args.adaptive_bins),
            "fine_bins": int(args.fine_bins),
            "bmin": float(args.bmin),
            "min_bins": int(args.min_bins),
            "no_signal_in_cr": bool(args.no_signal_in_cr),
        },
        "nuisances": {
            "lumi_unc": float(args.lumi_unc),
            "btag_unc": float(args.btag_unc),
            "shape_systs": shape_systs,
            "shape_mode": "uncorrelated_per_process (SYST_sample), correlated SR<->CR for same sample",
            "mc_stat": "staterror (from nominal sumw2; no floor when nominal==0)",
            "constrained_norms": ["mu_tt", "mu_st"],
        },
        "yields": {
            "SR": {"S": float(s_sr), "B": float(b_sr)},
            "CR_top": {"S": float(s_cr), "B": float(b_cr)},
        },
        "observed_ul_mu": float(obs_ul),
        "expected_ul_mu": {
            "minus2sigma": float(exp_ul[0]),
            "minus1sigma": float(exp_ul[1]),
            "median": float(exp_ul[2]),
            "plus1sigma": float(exp_ul[3]),
            "plus2sigma": float(exp_ul[4]),
        },
        "samples": {
            "tt_samples": tt_samples,
            "st_samples": st_samples,
            "other_backgrounds": [n for n in bkg_names if (n not in tt_samples and n not in st_samples)],
        },
    }

    outpath = args.out or os.path.join(workdir, f"limits_{args.method}.json")
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[OK] SR+CR pyhf limit saved: {outpath}")
    print(f"[LIMIT] expected mu < {exp_ul[2]:.6g}  (-1σ {exp_ul[1]:.6g}, +1σ {exp_ul[3]:.6g})")


if __name__ == "__main__":
    main()
