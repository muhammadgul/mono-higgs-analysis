import ROOT
from array import array

ROOT.TH1.AddDirectory(False)

def _h1(name, title, nb, lo, hi):
    h = ROOT.TH1F(name, title, nb, lo, hi)
    h.Sumw2()
    return h

def _h2(name, title, nbx, xlo, xhi, nby, ylo, yhi):
    h = ROOT.TH2F(name, title, nbx, xlo, xhi, nby, ylo, yhi)
    h.Sumw2()
    return h

def define_histograms():
    """Histograms for mono-Higgs analysis (cut-and-count + diagnostics).

    Convention:
      * pre-selection shapes:   <name>
      * post-selection (SR):    <name>_SR  (filled only after all cuts pass)
    """
    hist = {}

    # Core kinematics (pre)
    hist["MET"] = _h1("MET", "Missing E_{T};MET [GeV];Events", 120, 0, 600)
    hist["Nlep"] = _h1("Nlep", "Lepton multiplicity (e/#mu);N_{lep};Events", 6, 0, 6)
    hist["Njets"]  = _h1("Njets", "Jet multiplicity;N_{jets};Events", 12, 0, 12)
    hist["Nbjets"] = _h1("Nbjets", "b-jet multiplicity;N_{bjets};Events", 8, 0, 8)
    hist["HT"]     = _h1("HT", "Hadronic activity;H_{T} [GeV];Events", 120, 0, 1200)

    # Higgs candidate (best bb pair)
    hist["mbb"]   = _h1("mbb", "bb invariant mass;m_{bb} [GeV];Events", 80, 0, 400)
    hist["ptbb"]  = _h1("ptbb", "bb transverse momentum;p_{T}(bb) [GeV];Events", 120, 0, 600)
    hist["dr_bb"] = _h1("dr_bb", "bb separation;#DeltaR(b,b);Events", 60, 0, 6)
    hist["dphi_bb"] = _h1("dphi_bb", "#Delta#phi(b,b);#Delta#phi(b,b);Events", 64, 0, 3.2)
    hist["dphi_bb_met"] = _h1("dphi_bb_met", "#Delta#phi(bb, MET);#Delta#phi(bb,MET);Events", 64, 0, 3.2)

    # Leading b-jet
    hist["bjet1_pt"] = _h1("bjet1_pt", "Leading b-jet p_{T};p_{T}(b1) [GeV];Events", 120, 0, 600)
    hist["dphi_bjet1_met"] = _h1("dphi_bjet1_met", "#Delta#phi(b1, MET);#Delta#phi(b1,MET);Events", 64, 0, 3.2)

    # Recoil / balance variables
    hist["recoil"] = _h1("recoil", "Recoil p_{T};recoil [GeV];Events", 120, 0, 800)
    hist["balance"] = _h1("balance", "Balance p_{T}(bb)/MET; p_{T}(bb)/MET;Events", 80, 0, 8)
    hist["ptbb_minus_met"] = _h1("ptbb_minus_met", "|p_{T}(bb) - MET|;|p_{T}(bb)-MET| [GeV];Events", 120, 0, 600)

    # 2D diagnostics (pre)
    hist["ptbb_vs_MET"] = _h2("ptbb_vs_MET", "p_{T}(bb) vs MET;MET [GeV];p_{T}(bb) [GeV]",
                              60, 0, 600, 60, 0, 600)
    hist["recoil_vs_HT"] = _h2("recoil_vs_HT", "Recoil vs H_{T};H_{T} [GeV];recoil [GeV]",
                               60, 0, 1200, 60, 0, 800)

    # SR mirrors for key 1D shapes
    sr_list = [
        "MET","Nlep","Njets","Nbjets","HT",
        "mbb","ptbb","dr_bb","dphi_bb","dphi_bb_met",
        "bjet1_pt","dphi_bjet1_met",
        "recoil","balance","ptbb_minus_met",
    ]
    for name in sr_list:
        pre = hist.get(name)
        if not pre:
            continue
        hsr = pre.Clone(f"{name}_SR")
        hsr.SetTitle(pre.GetTitle() + " (after full selection)")
        hsr.Sumw2()
        hsr.SetDirectory(0)
        hist[f"{name}_SR"] = hsr

    return hist

def define_cutflow_hists(step_labels):
    """Returns (cutflow_raw, cutflow_wgt)."""
    n = len(step_labels)
    h_raw = ROOT.TH1F("cutflow_raw", "Cutflow (raw counts);Cut;Events", n, 0, n)
    h_wgt = ROOT.TH1F("cutflow_wgt", "Cutflow (weighted yield);Cut;Yield", n, 0, n)
    h_raw.Sumw2(); h_wgt.Sumw2()
    for i, lab in enumerate(step_labels, start=1):
        h_raw.GetXaxis().SetBinLabel(i, lab)
        h_wgt.GetXaxis().SetBinLabel(i, lab)
    return h_raw, h_wgt

def define_tree():
    """Event-level tree for ML / debugging."""
    t = ROOT.TTree("events", "Mono-Higgs analysis tree (for ML / cutflow)")

    br = {
        "weight": array('f', [1.0]),

        # Region flags
        "isSR": array('i', [0]),
        "isCR_top": array('i', [0]),

        "MET": array('f', [0.0]),
        "Nlep": array('i', [0]),
        "Njets": array('i', [0]),
        "Nbjets": array('i', [0]),
        "HT": array('f', [0.0]),

        "mbb": array('f', [0.0]),
        "ptbb": array('f', [0.0]),
        "dr_bb": array('f', [0.0]),
        "dphi_bb": array('f', [0.0]),
        "dphi_bb_met": array('f', [0.0]),

        "bjet1_pt": array('f', [0.0]),
        "dphi_bjet1_met": array('f', [0.0]),

        "recoil": array('f', [0.0]),
        "balance": array('f', [0.0]),
        "ptbb_minus_met": array('f', [0.0]),
    }

    t.Branch("weight", br["weight"], "weight/F")

    t.Branch("isSR", br["isSR"], "isSR/I")
    t.Branch("isCR_top", br["isCR_top"], "isCR_top/I")

    t.Branch("MET", br["MET"], "MET/F")
    t.Branch("Nlep", br["Nlep"], "Nlep/I")
    t.Branch("Njets", br["Njets"], "Njets/I")
    t.Branch("Nbjets", br["Nbjets"], "Nbjets/I")
    t.Branch("HT", br["HT"], "HT/F")

    t.Branch("mbb", br["mbb"], "mbb/F")
    t.Branch("ptbb", br["ptbb"], "ptbb/F")
    t.Branch("dr_bb", br["dr_bb"], "dr_bb/F")
    t.Branch("dphi_bb", br["dphi_bb"], "dphi_bb/F")
    t.Branch("dphi_bb_met", br["dphi_bb_met"], "dphi_bb_met/F")

    t.Branch("bjet1_pt", br["bjet1_pt"], "bjet1_pt/F")
    t.Branch("dphi_bjet1_met", br["dphi_bjet1_met"], "dphi_bjet1_met/F")

    t.Branch("recoil", br["recoil"], "recoil/F")
    t.Branch("balance", br["balance"], "balance/F")
    t.Branch("ptbb_minus_met", br["ptbb_minus_met"], "ptbb_minus_met/F")

    return t, br
