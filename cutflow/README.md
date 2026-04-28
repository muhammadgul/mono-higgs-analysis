# monohiggs-cutflow

Delphes ROOT (TTree `Delphes`) **cutflow + cut-and-count** for mono-Higgs (H→bb)+MET, designed to run in a normal ROOT+Python environment **without** needing Delphes' `libDelphes` / `libExRootAnalysis`.

## What you get
- Event loop reading Delphes output via **uproot** (no Delphes libs).
- Consistent JES/JER variations (analysis-level) + MET propagation (from jet pT shifts).
- ROOT output:
  - Histograms (pre-selection + `_SR` mirrors)
  - `events` TTree with ML-friendly branches and SR/CR tags
  - `cutflow_raw` and `cutflow_wgt` histograms

## Install (editable)
```bash
python -m pip install -U pip
python -m pip install -e .
```

## Run
```bash
monohiggs-cutflow input.root -o monoHiggs_cutcount.root --minMET 200 --minBJets 2 --minMbb 90 --maxMbb 150
```

## Notes
Branch names assume standard Delphes:
`Jet.PT/Eta/Phi/Mass/BTag`, `Electron.*`, `Muon.*`, `MissingET.MET/Phi`.

# First push from WSL
cd ~/madgraph/MG5_aMC_v3_5_13/mono-higgs-analysis
git init
git add .
git commit -m "Initial commit"
git remote add origin git@github.com:muhammadgul/mono-higgs-analysis.git
git branch -M main
git push -u origin main

# Later updates
cd ~/madgraph/MG5_aMC_v3_5_13/mono-higgs-analysis
git status
git add .
git commit -m "Update analysis"
git push

# Check remote
git remote -v

# Test SSH
ssh -T git@github.com
