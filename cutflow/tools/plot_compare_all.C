// plot_compare_all.C
// Compare shapes of signal vs backgrounds from cutcount ROOT outputs.
// Supports include/exclude filtering and 2D plotting with a readable palette + logZ.
//
// Usage example:
//   root -l -b -q 'plot_compare_all.C("cutcount_out","monoHiggs",100,false,true,true,"","wjets,zvvjets")'

#include "TFile.h"
#include "TKey.h"
#include "TClass.h"
#include "TSystem.h"
#include "TString.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TLatex.h"
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <iostream>
#include "TGaxis.h"

static std::vector<std::string> split_csv(const std::string &s) {
  std::vector<std::string> out;
  std::string tmp;
  for (char c : s) {
    if (c == ',') {
      if (!tmp.empty()) out.push_back(tmp);
      tmp.clear();
    } else {
      tmp.push_back(c);
    }
  }
  if (!tmp.empty()) out.push_back(tmp);

  for (auto &x : out) {
    x.erase(0, x.find_first_not_of(" \t\r\n"));
    x.erase(x.find_last_not_of(" \t\r\n") + 1);
  }
  out.erase(std::remove_if(out.begin(), out.end(),
                           [](const std::string &x){return x.empty();}),
            out.end());
  return out;
}

static bool in_list(const std::string &k, const std::vector<std::string> &lst) {
  return std::find(lst.begin(), lst.end(), k) != lst.end();
}

static bool sample_allowed(const std::string &sample,
                           const std::vector<std::string> &include,
                           const std::vector<std::string> &exclude) {
  if (!exclude.empty() && in_list(sample, exclude)) return false;
  if (!include.empty() && !in_list(sample, include) && sample != "signal") return false;
  return true;
}

static std::map<std::string, std::string> sample_pretty() {
  return {
    {"signal","Signal"},
    {"tt_semilep","t#bar{t} (semi-lep)"},
    {"tt_dilep","t#bar{t} (di-lep)"},
    {"tt_had","t#bar{t} (had)"},
    {"st_tch_top","single-top t-ch (t)"},
    {"st_tch_tbar","single-top t-ch (#bar{t})"},
    {"st_tW","single-top tW"},
    {"wjets","W+jets"},
    {"zvvjets","Z(#nu#nu)+jets"}
  };
}


static TString pretty_var(const TString &hname) {
  if (hname == "dphi_bb") return "#Delta#phi(b,#bar{b})";
  if (hname == "dr_bb") return "#DeltaR(b,#bar{b})";
  if (hname == "mbb") return "m_{bb} [GeV]";
  if (hname == "bb_pt") return "p_{T}(bb) [GeV]";
  if (hname == "bjet1_pt") return "p_{T}(b_{1}) [GeV]";
  if (hname == "dphi_bb_met") return "#Delta#phi(bb, E_{T}^{miss})";
  if (hname == "dphi_b1_met") return "#Delta#phi(b_{1}, E_{T}^{miss})";
  if (hname == "min_dphi_met_jet") return "min #Delta#phi(E_{T}^{miss}, j)";
  if (hname == "met") return "E_{T}^{miss} [GeV]";
  if (hname == "ht") return "H_{T} [GeV]";
  if (hname == "recoil") return "Recoil [GeV]";
  TString s = hname;
  s.ReplaceAll("_", " ");
  return s;
}

static int sample_color(const std::string &k) {
  if (k=="signal") return kRed+1;
  if (k=="tt_semilep") return kAzure+2;
  if (k=="tt_dilep") return kAzure+4;
  if (k=="tt_had") return kAzure+6;
  if (k=="st_tch_top") return kGreen+2;
  if (k=="st_tch_tbar") return kGreen+3;
  if (k=="st_tW") return kGreen+4;
  if (k=="wjets") return kOrange+7;
  if (k=="zvvjets") return kViolet+2;
  return kGray+2;
}

// ---- SIGNATURE MUST MATCH YOUR RUNNER CALL (8 args) ----
void plot_compare_all(const char* outdir="cutcount_out",
                      const char* tag="monoHiggs",
                      double lumi=100.0,
                      bool doNorm=false,
                      bool doLogy=true,
                      bool do2D=true,
                      const char* includeSamples="",
                      const char* excludeSamples="") {

  // Style
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetTitleFont(42, "XYZ");
  gStyle->SetLabelFont(42, "XYZ");
  gStyle->SetTitleSize(0.050, "X");
  gStyle->SetTitleSize(0.050, "Y");
  gStyle->SetLabelSize(0.040, "X");
  gStyle->SetLabelSize(0.040, "Y");
  gStyle->SetTitleOffset(1.05, "X");
  gStyle->SetTitleOffset(1.20, "Y");
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetPalette(kViridis);
  gStyle->SetNumberContours(100);
  TGaxis::SetMaxDigits(2);                 // compact scientific notation
  TGaxis::SetExponentOffset(-0.02, 0.005, "y");  // put 10^n near top-left
  
  std::string OUTDIR(outdir);
  std::string TAG(tag);

  std::vector<std::string> include = split_csv(includeSamples ? includeSamples : "");
  std::vector<std::string> exclude = split_csv(excludeSamples ? excludeSamples : "");

  // Input files expected: <outdir>/<tag>_<sample>_cutcount.root
  std::map<std::string, std::string> files = {
    {"signal",     OUTDIR + "/" + TAG + "_signal_cutcount.root"},
    {"tt_semilep", OUTDIR + "/" + TAG + "_tt_semilep_cutcount.root"},
    {"tt_dilep",   OUTDIR + "/" + TAG + "_tt_dilep_cutcount.root"},
    {"tt_had",     OUTDIR + "/" + TAG + "_tt_had_cutcount.root"},
    {"st_tch_top", OUTDIR + "/" + TAG + "_st_tch_top_cutcount.root"},
    {"st_tch_tbar",OUTDIR + "/" + TAG + "_st_tch_tbar_cutcount.root"},
    {"st_tW",      OUTDIR + "/" + TAG + "_st_tW_cutcount.root"},
    {"wjets",      OUTDIR + "/" + TAG + "_wjets_cutcount.root"},
    {"zvvjets",    OUTDIR + "/" + TAG + "_zvvjets_cutcount.root"}
  };

  // Output dirs
  std::string out_base = OUTDIR + "/plots_compare_all";
  std::string out_1d   = out_base + "/plots1D";
  std::string out_2d   = out_base + "/plots2D";
  gSystem->mkdir(out_base.c_str(), true);
  gSystem->mkdir(out_1d.c_str(), true);
  gSystem->mkdir(out_2d.c_str(), true);

  // Open files
  std::map<std::string, TFile*> f;
  for (auto &kv : files) {
    const std::string &samp = kv.first;
    if (!sample_allowed(samp, include, exclude)) continue;

    TFile *tf = TFile::Open(kv.second.c_str(), "READ");
    if (!tf || tf->IsZombie()) {
      std::cerr << "[WARN] cannot open " << kv.second << " (skip)\n";
      continue;
    }
    f[samp] = tf;
  }

  if (f.find("signal") == f.end()) {
    std::cerr << "[ERROR] signal file missing or excluded. Cannot compare.\n";
    for (auto &kv : f) kv.second->Close();
    return;
  }

  // Collect histogram names from signal file
  std::vector<std::string> hnames;
  TIter nextkey(f["signal"]->GetListOfKeys());
  while (TKey *key = (TKey*)nextkey()) {
    TObject *obj = key->ReadObj();
    if (!obj) continue;
    if (obj->InheritsFrom(TH1::Class())) {
      hnames.push_back(obj->GetName());
    }
    delete obj;
  }
  std::sort(hnames.begin(), hnames.end());
  hnames.erase(std::unique(hnames.begin(), hnames.end()), hnames.end());

  auto pretty = sample_pretty();

  for (const auto &hname : hnames) {
    TObject *test = f["signal"]->Get(hname.c_str());
    if (!test) continue;
    bool is2D = test->InheritsFrom(TH2::Class());
    delete test;

    if (is2D && !do2D) continue;
if (!is2D) {
      // ----------------------------
      // 1D overlay (CMS-like style)
      // ----------------------------
      TCanvas *c = new TCanvas(("c_"+TString(hname)).Data(), hname.c_str(), 900, 700);
      c->cd();
      if (doLogy) c->SetLogy();
      c->SetTopMargin(0.12);
      c->SetRightMargin(0.05);
      c->SetLeftMargin(0.12);
      c->SetBottomMargin(0.12);

      // Legend: 2 columns, compact, top-right
      TLegend *leg = new TLegend(0.52, 0.62, 0.88, 0.88);
      leg->SetNColumns(2);
      leg->SetBorderSize(0);
      leg->SetFillStyle(0);
      leg->SetTextFont(72);
      leg->SetTextSize(0.038);

      TH1 *hs = (TH1*)f["signal"]->Get(hname.c_str());
      if (!hs) { delete leg; delete c; continue; }

      // Clone + (optional) normalize
      TH1 *hsd = (TH1*)hs->Clone(("signal_"+TString(hname)).Data());
      hsd->SetDirectory(0);
      hsd->SetLineColor(sample_color("signal"));
      hsd->SetLineWidth(3);
      hsd->SetFillStyle(0);
      hsd->SetTitle("");
      hsd->SetStats(0);
      if (doNorm && hsd->Integral() > 0) hsd->Scale(1.0 / hsd->Integral());

      // Axis titles (and force style on the object, not only gStyle)
      hsd->GetXaxis()->SetTitle(pretty_var(hname.c_str()));
      if (doNorm) hsd->GetYaxis()->SetTitle("Normalized entries");
      else        hsd->GetYaxis()->SetTitle("Events");

      hsd->GetXaxis()->SetTitleFont(72);
      hsd->GetYaxis()->SetTitleFont(72);
      hsd->GetXaxis()->SetLabelFont(72);
      hsd->GetYaxis()->SetLabelFont(72);

      hsd->GetXaxis()->SetTitleSize(0.0450);
      hsd->GetYaxis()->SetTitleSize(0.0450);
      hsd->GetXaxis()->SetLabelSize(0.040);
      hsd->GetYaxis()->SetLabelSize(0.040);

      hsd->GetXaxis()->SetTitleOffset(1.0);
      hsd->GetYaxis()->SetTitleOffset(1.1);

      // Prepare background clones, track global max so nothing gets clipped
      std::vector<TH1*> hbgs;
      hbgs.reserve(f.size());

      double maxY = hsd->GetMaximum();

      leg->AddEntry(hsd, pretty["signal"].c_str(), "l");

      for (auto &kv : f) {
        const std::string &samp = kv.first;
        if (samp == "signal") continue;

        TH1 *hb = (TH1*)kv.second->Get(hname.c_str());
        if (!hb) continue;

        TH1 *hbd = (TH1*)hb->Clone((samp+"_"+hname).c_str());
        hbd->SetDirectory(0);
        hbd->SetLineColor(sample_color(samp));
        hbd->SetLineWidth(2);
        hbd->SetFillStyle(0);
        hbd->SetTitle("");
        hbd->SetStats(0);
        if (doNorm && hbd->Integral() > 0) hbd->Scale(1.0 / hbd->Integral());

        maxY = std::max(maxY, (double)hbd->GetMaximum());
        hbgs.push_back(hbd);

        std::string lab = pretty.count(samp) ? pretty[samp] : samp;
        leg->AddEntry(hbd, lab.c_str(), "l");
      }


      // Y-range with headroom (prevents backgrounds going out of canvas)
      if (doLogy) {
        hsd->SetMinimum(1e-6);
        hsd->SetMaximum(50.0 * maxY); // log padding
      } else {
        hsd->SetMinimum(0.0);
        hsd->SetMaximum(1.35 * maxY);
      }

      // Draw
      hsd->Draw("HIST");

      for (auto *hbd : hbgs) hbd->Draw("HIST SAME");
      leg->Draw();

      // CMS-like top text
      TLatex lat;
      lat.SetNDC(true);
      lat.SetTextFont(72);
      lat.SetTextSize(0.040);
      lat.DrawLatex(0.2, 0.89, "CMS Simulation @ 13TeV, mono-H  gg#rightarrow h + #chi#bar{#chi}");

      c->RedrawAxis();
      c->Modified();
      c->Update();

      std::string outpng = out_1d + "/" + hname + ".png";
      c->Print(outpng.c_str());

      for (auto *hbd : hbgs) delete hbd;
      delete hsd;
      delete leg;
      delete c;
    } else {
      // 2D: per-sample COLZ with logZ to avoid saturation
      for (auto &kv : f) {
        const std::string &samp = kv.first;
        TH2 *h2 = (TH2*)kv.second->Get(hname.c_str());
        if (!h2) continue;

        TCanvas *c = new TCanvas(("c2_"+TString(hname)+"_"+TString(samp)).Data(), hname.c_str(), 900, 750);
        c->SetRightMargin(0.14);
        /* linear Z for readability */
        // c->SetLogz();
        c->SetLogz(0);
        c->SetTopMargin(0.10);
        c->SetRightMargin(0.14);
        c->SetLeftMargin(0.12);
        c->SetBottomMargin(0.12);


        TH2 *h2d = (TH2*)h2->Clone((samp+"_"+hname+"_2d").c_str());
        h2d->SetDirectory(0);
        // (no logZ) keep default minimum

                // Axis titles (make sure they are meaningful for your 2D hists)
        if (TString(hname) == "recoil_vs_HT") {
          h2d->GetXaxis()->SetTitle("H_{T} [GeV]");
          h2d->GetYaxis()->SetTitle("Recoil [GeV]");
        } else if (TString(hname) == "ptbb_vs_MET") {
          h2d->GetXaxis()->SetTitle("E_{T}^{miss} [GeV]");
          h2d->GetYaxis()->SetTitle("p_{T}(bb) [GeV]");
        }

// ---------- CMS-like axis style (2D) ----------
	h2d->SetTitle("");
	h2d->SetStats(0);

// X axis
	h2d->GetXaxis()->SetTitleFont(72);
	h2d->GetXaxis()->SetLabelFont(72);
	h2d->GetXaxis()->SetTitleSize(0.04);
	h2d->GetXaxis()->SetLabelSize(0.038);
	h2d->GetXaxis()->SetTitleOffset(1.050);

// Y axis
	h2d->GetYaxis()->SetTitleFont(72);
	h2d->GetYaxis()->SetLabelFont(72);
	h2d->GetYaxis()->SetTitleSize(0.04);
	h2d->GetYaxis()->SetLabelSize(0.038);
	h2d->GetYaxis()->SetTitleOffset(1.050);

// Z axis (color bar)
	h2d->GetZaxis()->SetTitleFont(72);
	h2d->GetZaxis()->SetLabelFont(72);
	h2d->GetZaxis()->SetTitleSize(0.04);
	h2d->GetZaxis()->SetLabelSize(0.038);
  	h2d->GetZaxis()->SetTitleOffset(1.20);
	
        h2d->Draw("COLZ");
        // Top annotations (like your reference plot)
        TLatex lat;
        lat.SetNDC(true);
        lat.SetTextFont(72);
        lat.SetTextSize(0.04);

        // Left: process label
        lat.DrawLatex(0.18, 0.92, "CMS Simulation @ 13 TeV, #it{mono-H}  gg#rightarrow h + #chi#bar{#chi}");

        // Right: sample label (signal/background)
     //   std::string sampLabel = pretty.count(samp) ? pretty[samp] : samp;
     //   TString rightTxt = TString("#it{mono-H} ") + TString(sampLabel.c_str());
    //    lat.DrawLatex(0.70, 0.96, rightTxt);

        // Bottom-right: variable name
  //      lat.SetTextSize(0.040);
 //       lat.DrawLatex(0.70, 0.04, TString(hname).ReplaceAll("_", " "));

        std::string outpng = out_2d + "/" + hname + "__" + samp + ".png";
        c->Print(outpng.c_str());

        delete h2d;
        delete c;
      }
    }
  }

  for (auto &kv : f) kv.second->Close();
  std::cout << "[OK] plot_compare_all.C finished\n";
}
