#!/usr/bin/env bash

# ---------- Inputs ----------
export IN_DIR="$HOME/madgraph/MG5_aMC_v3_5_13/monohiggs-cutflow/outputs/cutcount_out_bdt"
export TAG="monoHiggs"

# Which TMVA method
export METHOD="BDTG"     # or BDT

# Tree name
export TREE="events"

# Output working directory
export OUT_DIR="work_tmva_${METHOD}"
mkdir -p "${OUT_DIR}"

# ---------- Physics selections ----------
# TRAIN_CUT: only for TMVA training (more stats but SR-like topology)
# SR_CUT: used for scan/yields/limits/plots (your physics SR)
export TRAIN_CUT="(Nlep==0 && Nbjets>=2)"
export SR_CUT="(isSR==1 && Nlep==0)"
export CR_CUT="(isCR_top==1 && Nlep>=1 && Njets>=4 && Nbjets>=2)"
# ---------- Background switches ----------
export USE_WJETS=0      # 1 = include, 0 = exclude
export USE_ZVVJETS=0    # 1 = include, 0 = exclude

# Core backgrounds (always on)
BKG_CORE="tt_semilep tt_dilep tt_had st_tW st_tch_top st_tch_tbar"

# Optional ones
BKG_OPT=""
if [[ "${USE_WJETS}" -eq 1 ]]; then
  BKG_OPT="${BKG_OPT} wjets"
fi
if [[ "${USE_ZVVJETS}" -eq 1 ]]; then
  BKG_OPT="${BKG_OPT} zvvjets"
fi

# Export final background list used by ALL steps that need explicit background lists
export BKG_SAMPLES="${BKG_CORE}${BKG_OPT:+ ${BKG_OPT}}"

# For apply step: list of all samples to process (signal + backgrounds)
export APPLY_SAMPLES="signal ${BKG_SAMPLES}"

export SYSTS="Nominal JESUp JESDown JERUp JERDown"

echo "[CONFIG] METHOD=${METHOD}"
echo "[CONFIG] TRAIN_CUT=${TRAIN_CUT}"
echo "[CONFIG] SR_CUT=${SR_CUT}"
echo "[CONFIG] BKG_SAMPLES=${BKG_SAMPLES}"
