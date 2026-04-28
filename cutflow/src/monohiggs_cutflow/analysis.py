from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import ROOT

from .physics import dphi, dr, passes_minmax


@dataclass
class SystConfig:
    syst: str = "Nominal"  # Nominal, JESUp, JESDown, JERUp, JERDown
    jes_frac: float = 0.02
    jer_sigma: float = 0.10
    jer_unc_frac: float = 0.10
    jer_seed: int = 12345


@dataclass
class ObjectCuts:
    jet_pt: float = 30.0
    jet_eta: float = 4.0
    bjet_pt: float = 40.0
    bjet_eta: float = 2.5
    target_mbb: float = 125.0


@dataclass
class SRConfig:
    # lepton requirement
    no_lepton_veto: bool = False
    min_leptons: int | None = None
    max_leptons: int | None = 0

    # SR cuts (all optional)
    min_met: float | None = None
    min_jets: int | None = None
    max_jets: int | None = None
    min_bjets: int | None = None
    max_bjets: int | None = None
    min_balance: float | None = None
    max_balance: float | None = None
    min_dphi_bb: float | None = None
    max_dphi_bb: float | None = None
    min_bjet1_pt: float | None = None
    min_dphi_bb_met: float | None = None
    max_dphi_bb_met: float | None = None
    min_dphi_b1_met: float | None = None
    max_dphi_b1_met: float | None = None
    min_dr_bb: float | None = None
    max_dr_bb: float | None = None
    min_mbb: float | None = None
    max_mbb: float | None = None
    min_bb_pt: float | None = None
    min_ptbb_minus_met: float | None = None
    max_ptbb_minus_met: float | None = None
    min_recoil: float | None = None
    min_ht: float | None = None
    max_ht: float | None = None


@dataclass
class CRTopConfig:
    cr_min_leptons: int = 1
    cr_max_leptons: int = 1
    cr_min_met: float = 80.0
    cr_min_jets: int = 4
    cr_max_jets: int | None = None
    cr_min_bjets: int = 2
    cr_max_bjets: int | None = None
    cr_min_ht: float = 300.0

    cr_use_mbb_window: bool = False
    cr_invert_mbb_window: bool = False
    cr_invert_dphi_bb_met: bool = False


def varied_jet_pt_mass(pt_nom: float, m_nom: float, cfg: SystConfig, ievt: int, jidx: int, rnd: ROOT.TRandom3) -> tuple[float, float]:
    pt = float(pt_nom)
    m = float(m_nom)

    if cfg.syst == "Nominal":
        return pt, m

    if cfg.syst in ("JESUp", "JESDown"):
        scale = (1.0 + cfg.jes_frac) if cfg.syst == "JESUp" else (1.0 - cfg.jes_frac)
        scale = max(0.0, scale)
        return pt * scale, m * scale

    if cfg.syst in ("JERUp", "JERDown"):
        sig = float(cfg.jer_sigma)
        if cfg.syst == "JERUp":
            sig *= (1.0 + cfg.jer_unc_frac)
        else:
            sig *= max(0.0, (1.0 - cfg.jer_unc_frac))

        seed = int(cfg.jer_seed) + int(ievt) * 1_000_003 + int(jidx) * 1009
        rnd.SetSeed(seed)
        smear = rnd.Gaus(0.0, sig)
        scale = max(0.0, 1.0 + smear)
        return pt * scale, m * scale

    return pt, m


def varied_met(
    met_nom: float,
    metphi_nom: float,
    jet_pts_nom: List[float],
    jet_phis: List[float],
    jet_masses_nom: List[float],
    cfg: SystConfig,
    ievt: int,
    rnd: ROOT.TRandom3,
) -> tuple[float, float]:
    """Propagate jet pT shifts into MET (analysis-level)."""
    metx = float(met_nom) * math.cos(float(metphi_nom))
    mety = float(met_nom) * math.sin(float(metphi_nom))

    if cfg.syst == "Nominal":
        return float(met_nom), float(metphi_nom)

    dpx = 0.0
    dpy = 0.0
    for j, (pt_nom, phi, m_nom) in enumerate(zip(jet_pts_nom, jet_phis, jet_masses_nom)):
        pt_var, _ = varied_jet_pt_mass(pt_nom, m_nom, cfg, ievt, j, rnd)
        dpt = pt_var - float(pt_nom)
        dpx += dpt * math.cos(float(phi))
        dpy += dpt * math.sin(float(phi))

    metx_var = metx - dpx
    mety_var = mety - dpy
    met_var = math.sqrt(metx_var * metx_var + mety_var * mety_var)
    metphi_var = math.atan2(mety_var, metx_var)
    return float(met_var), float(metphi_var)


def select_event(
    ievt: int,
    jets_nom: tuple[list[float], list[float], list[float], list[float], list[int]],
    leptons: tuple[int, int],
    met_nom: tuple[float, float],
    obj: ObjectCuts,
    sr: SRConfig,
    cr: CRTopConfig,
    syst: SystConfig,
    rnd: ROOT.TRandom3,
) -> Dict[str, float | int | bool]:
    """
    Return computed variables and SR/CR flags for one event.
    jets_nom: (pt, eta, phi, mass, btag)
    leptons: (nEle, nMu)
    met_nom: (MET, Phi)
    """
    jet_pt_nom, jet_eta, jet_phi, jet_mass_nom, jet_btag = jets_nom
    nlep = int(leptons[0] + leptons[1])

    # SR lepton requirement (CR is computed independently)
    pass_lep_sr = True
    if not sr.no_lepton_veto:
        minL = 0 if sr.min_leptons is None else int(sr.min_leptons)
        maxL = (10**9) if sr.max_leptons is None else int(sr.max_leptons)
        pass_lep_sr = (nlep >= minL) and (nlep <= maxL)

    # MET varied consistently with jets
    met, met_phi = varied_met(
        met_nom[0], met_nom[1],
        jet_pt_nom, jet_phi, jet_mass_nom,
        syst, ievt, rnd
    )

    # Build jets/bjets with pT variations
    jets = []   # (eta, phi, pt_var, m_var, btag)
    bjets = []  # subset
    ht = 0.0
    for j, (pt_nom, eta, phi, m_nom, btag) in enumerate(zip(jet_pt_nom, jet_eta, jet_phi, jet_mass_nom, jet_btag)):
        pt_var, m_var = varied_jet_pt_mass(pt_nom, m_nom, syst, ievt, j, rnd)
        if pt_var < obj.jet_pt or abs(float(eta)) > obj.jet_eta:
            continue
        jets.append((float(eta), float(phi), float(pt_var), float(m_var), int(btag)))
        ht += float(pt_var)
        if int(btag) > 0 and pt_var >= obj.bjet_pt and abs(float(eta)) <= obj.bjet_eta:
            bjets.append((float(eta), float(phi), float(pt_var), float(m_var), int(btag)))

    njets = len(jets)
    nbjets = len(bjets)

    # CR_top early (does not require bb)
    is_cr_top = (
        passes_minmax(nlep, cr.cr_min_leptons, cr.cr_max_leptons)
        and passes_minmax(met, cr.cr_min_met, None)
        and passes_minmax(njets, cr.cr_min_jets, cr.cr_max_jets)
        and passes_minmax(nbjets, cr.cr_min_bjets, cr.cr_max_bjets)
        and passes_minmax(ht, cr.cr_min_ht, None)
    )

    # Defaults
    mbb = -999.0
    ptbb = 0.0
    dr_bb_val = 0.0
    dphi_bb_val = 0.0
    dphi_bb_met = 0.0
    bjet1_pt = 0.0
    dphi_bjet1_met = 0.0
    recoil = 0.0
    balance = 0.0
    ptbb_minus_met = 0.0

    pass_sr = False

    have_bb = nbjets >= 2
    if have_bb:
        # choose best bb pair by |mbb-target|
        best = None
        best_dm = 1e18

        for i in range(nbjets):
            for k in range(i + 1, nbjets):
                e1, p1, pt1, m1, _ = bjets[i]
                e2, p2, pt2, m2, _ = bjets[k]

                v1 = ROOT.TLorentzVector()
                v2 = ROOT.TLorentzVector()
                v1.SetPtEtaPhiM(pt1, e1, p1, m1)
                v2.SetPtEtaPhiM(pt2, e2, p2, m2)
                bb = v1 + v2
                mbb_tmp = float(bb.M())
                dm = abs(mbb_tmp - obj.target_mbb)
                if dm < best_dm:
                    best_dm = dm
                    best = (e1, p1, pt1, m1, e2, p2, pt2, m2, bb)

        if best is not None:
            e1, p1, pt1, m1, e2, p2, pt2, m2, bb = best
            mbb = float(bb.M())
            ptbb = float(bb.Pt())
            dr_bb_val = float(dr(e1, p1, e2, p2))
            dphi_bb_val = float(dphi(p1, p2))
            dphi_bb_met = float(dphi(float(bb.Phi()), met_phi))

            # leading bjet by varied pT
            bjet1_pt = float(max(pt for _e, _p, pt, _m, _b in bjets))
            # leading bjet phi (find corresponding)
            lead = max(bjets, key=lambda x: x[2])
            dphi_bjet1_met = float(dphi(lead[1], met_phi))

            metvec = ROOT.TLorentzVector()
            metvec.SetPtEtaPhiM(met, 0.0, met_phi, 0.0)
            recoil = float((bb + metvec).Pt())

            balance = float(ptbb / met) if met > 1e-9 else 0.0
            ptbb_minus_met = float(abs(ptbb - met))

            # SR selection
            pass_sr = (
                pass_lep_sr
                and passes_minmax(met, sr.min_met, None)
                and passes_minmax(njets, sr.min_jets, sr.max_jets)
                and passes_minmax(nbjets, sr.min_bjets, sr.max_bjets)
                and passes_minmax(balance, sr.min_balance, sr.max_balance)
                and passes_minmax(dphi_bb_val, sr.min_dphi_bb, sr.max_dphi_bb)
                and passes_minmax(bjet1_pt, sr.min_bjet1_pt, None)
                and passes_minmax(dphi_bb_met, sr.min_dphi_bb_met, sr.max_dphi_bb_met)
                and passes_minmax(dphi_bjet1_met, sr.min_dphi_b1_met, sr.max_dphi_b1_met)
                and passes_minmax(dr_bb_val, sr.min_dr_bb, sr.max_dr_bb)
                and passes_minmax(mbb, sr.min_mbb, sr.max_mbb)
                and passes_minmax(ptbb, sr.min_bb_pt, None)
                and passes_minmax(ptbb_minus_met, sr.min_ptbb_minus_met, sr.max_ptbb_minus_met)
                and passes_minmax(recoil, sr.min_recoil, None)
                and passes_minmax(ht, sr.min_ht, sr.max_ht)
            )

    # Optional CR sideband toggles (only meaningful if bb exists and SR window is defined)
    if have_bb and (sr.min_mbb is not None) and (sr.max_mbb is not None):
        if cr.cr_invert_mbb_window:
            is_cr_top = is_cr_top and (not (mbb >= float(sr.min_mbb) and mbb <= float(sr.max_mbb)))
        elif cr.cr_use_mbb_window:
            is_cr_top = is_cr_top and (mbb >= float(sr.min_mbb) and mbb <= float(sr.max_mbb))

    if have_bb and cr.cr_invert_dphi_bb_met and (sr.max_dphi_bb_met is not None):
        is_cr_top = is_cr_top and (dphi_bb_met > float(sr.max_dphi_bb_met))

    return {
        "MET": met,
        "MET_phi": met_phi,
        "Nlep": nlep,
        "Njets": njets,
        "Nbjets": nbjets,
        "HT": ht,
        "mbb": mbb,
        "ptbb": ptbb,
        "dr_bb": dr_bb_val,
        "dphi_bb": dphi_bb_val,
        "dphi_bb_met": dphi_bb_met,
        "bjet1_pt": bjet1_pt,
        "dphi_bjet1_met": dphi_bjet1_met,
        "recoil": recoil,
        "balance": balance,
        "ptbb_minus_met": ptbb_minus_met,
        "isSR": int(1 if pass_sr else 0),
        "isCR_top": int(1 if is_cr_top else 0),
        "pass_lep_sr": int(1 if pass_lep_sr else 0),
    }
