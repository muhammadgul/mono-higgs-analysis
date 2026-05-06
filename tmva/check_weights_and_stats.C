// check_weights_and_stats.C
// Usage:
//   root -l -q 'check_weights_and_stats.C("../cutcount_out_bdt","monoHiggs","events","(Nlep==0 && Nbjets>=2)","weight",1e4)'
//
// Prints per-sample stats + weight diagnostics.

#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <TString.h>
#include <TMath.h>
#include <TH1D.h>
#include <TROOT.h>
#include <iostream>
#include <vector>

static double SumW(TTree* t, const TString& w, const TString& cut) {
  TString hname = Form("h_sumw_%p", (void*)t);
  TH1D* h = (TH1D*)gROOT->FindObject(hname);
  if (!h) h = new TH1D(hname, hname, 1, 0, 1);
  h->Reset();
  TString expr = "0.5>>" + hname;
  TString sel  = cut.Length() ? Form("(%s)*(%s)", w.Data(), cut.Data()) : w;
  t->Draw(expr, sel, "goff");
  return h->GetSumOfWeights();
}

static Long64_t Count(TTree* t, const TString& cut) {
  // Fast count using TTree::Draw with goff
  TString hname = Form("h_cnt_%p", (void*)t);
  TH1D* h = (TH1D*)gROOT->FindObject(hname);
  if (!h) h = new TH1D(hname, hname, 1, 0, 1);
  h->Reset();
  TString expr = "0.5>>" + hname;
  TString sel  = cut.Length() ? cut : "1";
  t->Draw(expr, sel, "goff");
  return (Long64_t)h->GetEntries(); // entries filled (selected rows)
}

static void WeightQuantiles(TTree* t, const TString& w, const TString& cut,
                            double& q50, double& q95, double& q99, double& q999,
                            double& wmin, double& wmax, Long64_t& nneg, Long64_t& nspike,
                            double spike_thr) {
  // Use TTree::Draw to fill a temporary histogram of weights for quantiles.
  // Auto-binning range based on min/max is tricky; we first compute min/max with Draw.
  // Step 1: get min/max weights in selection
  TString htmpname = Form("h_wtmp_%p", (void*)t);
  TH1D* htmp = (TH1D*)gROOT->FindObject(htmpname);
  if (htmp) { htmp->Reset(); }
  else      { htmp = new TH1D(htmpname, htmpname, 1, 0, 1); }

  // Get min/max via Draw into arrays
  TString drawExpr = w;  // draw the weight values
  TString selExpr  = cut.Length() ? cut : "1";
  Long64_t nsel = t->Draw(drawExpr, selExpr, "goff");
  const double* arr = t->GetV1();
  wmin = +1e300; wmax = -1e300;
  nneg = 0; nspike = 0;
  for (Long64_t i = 0; i < nsel; i++) {
    double wi = arr[i];
    if (wi < wmin) wmin = wi;
    if (wi > wmax) wmax = wi;
    if (wi < 0) nneg++;
    if (TMath::Abs(wi) > spike_thr) nspike++;
  }
  if (nsel <= 0) {
    q50=q95=q99=q999=0;
    wmin=0; wmax=0;
    return;
  }

  // Step 2: histogram weights for quantiles (robust range)
  // Use symmetric range around 0 if negatives exist, else [0, wmax]
  double lo, hi;
  if (wmin < 0) {
    double a = TMath::Max(TMath::Abs(wmin), TMath::Abs(wmax));
    lo = -a; hi = +a;
  } else {
    lo = 0.0; hi = wmax;
  }
  // Avoid degenerate range
  if (hi <= lo) { lo -= 1.0; hi += 1.0; }

  TString hqname = Form("h_wq_%p", (void*)t);
  TH1D* hq = (TH1D*)gROOT->FindObject(hqname);
  if (hq) { hq->Reset(); hq->SetBins(2000, lo, hi); }
  else    { hq = new TH1D(hqname, hqname, 2000, lo, hi); }

  TString expr = Form("%s>>%s", w.Data(), hqname.Data());
  t->Draw(expr, selExpr, "goff");

  double probs[4] = {0.50, 0.95, 0.99, 0.999};
  double qs[4]    = {0,0,0,0};
  hq->GetQuantiles(4, qs, probs);
  q50 = qs[0]; q95 = qs[1]; q99 = qs[2]; q999 = qs[3];
}

void check_weights_and_stats(const char* in_dir="../cutcount_out_bdt",
                             const char* tag="monoHiggs",
                             const char* tree="events",
                             const char* cut="(Nlep==0 && Nbjets>=2)",
                             const char* wbranch="weight",
                             double spike_thr=1e4)
{
  gROOT->SetBatch(true);

  std::vector<TString> samples = {
    "signal",
    "tt_semilep","tt_dilep","tt_had",
    "st_tW","st_tch_top","st_tch_tbar",
    "wjets","zvvjets"
  };

  std::cout << "Input dir: " << in_dir << "\n";
  std::cout << "Tree: " << tree << "\n";
  std::cout << "Cut: " << cut << "\n";
  std::cout << "Weight branch: " << wbranch << "\n";
  std::cout << "Spike threshold |w| > " << spike_thr << "\n\n";

  std::cout
    << "Sample"
    << "\tEntries"
    << "\tPassCut"
    << "\tSumW"
    << "\tSumW(Cut)"
    << "\tWmin"
    << "\tWmax"
    << "\tNneg"
    << "\tNspike"
    << "\tQ50"
    << "\tQ95"
    << "\tQ99"
    << "\tQ99.9"
    << "\n";

  for (auto& s : samples) {
    TString path = Form("%s/%s_%s_cutcount.root", in_dir, tag, s.Data());
    if (gSystem->AccessPathName(path)) {
      std::cout << s << "\t[MISSING] " << path << "\n";
      continue;
    }

    TFile* f = TFile::Open(path, "READ");
    if (!f || f->IsZombie()) {
      std::cout << s << "\t[ZOMBIE] " << path << "\n";
      continue;
    }

    TTree* t = (TTree*)f->Get(tree);
    if (!t) {
      std::cout << s << "\t[NO TREE] " << tree << " in " << path << "\n";
      f->Close();
      continue;
    }

    Long64_t nall = t->GetEntries();
    Long64_t npass = Count(t, cut);

    double sumw_all = SumW(t, wbranch, "1");
    double sumw_cut = SumW(t, wbranch, cut);

    double q50,q95,q99,q999,wmin,wmax;
    Long64_t nneg,nspike;
    WeightQuantiles(t, wbranch, cut, q50,q95,q99,q999, wmin,wmax, nneg,nspike, spike_thr);

    std::cout
      << s
      << "\t" << nall
      << "\t" << npass
      << "\t" << sumw_all
      << "\t" << sumw_cut
      << "\t" << wmin
      << "\t" << wmax
      << "\t" << nneg
      << "\t" << nspike
      << "\t" << q50
      << "\t" << q95
      << "\t" << q99
      << "\t" << q999
      << "\n";

    f->Close();
  }

  std::cout << "\nDone.\n";
}
