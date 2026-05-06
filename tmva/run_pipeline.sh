#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/config.sh"

usage() {
  cat <<EOF
Usage:
  ./run_pipeline.sh [--train] [--apply] [--scan] [--yields] [--limits] [--plots] [--all]
                    [--dry-run] [--force] [--gui]

Steps:
  --train   Train TMVA and write: ${OUT_DIR}/TMVA_\${METHOD}.root + weights in ${OUT_DIR}/dataset/weights/
  --apply   Apply TMVA and write: ${OUT_DIR}/*_withBDT.root
  --scan    Scan BDT cut -> ${OUT_DIR}/scan_\${METHOD}.csv
  --yields  Compute yields -> ${OUT_DIR}/yields_\${METHOD}.json
  --limits  Compute pyhf limit -> ${OUT_DIR}/limits_\${METHOD}.json
  --plots   Make publication plots

Options:
  --dry-run   Print commands only (do not execute)
  --force     Run even if output files exist
  --gui       Open TMVAGui after training (needs display, not batch)

Examples:
  ./run_pipeline.sh --all
  ./run_pipeline.sh --train --gui
  ./run_pipeline.sh --apply --scan
  ./run_pipeline.sh --limits
  ./run_pipeline.sh --all --dry-run
EOF
}

DRYRUN=0
FORCE=0
GUI=0

DO_TRAIN=0
DO_APPLY=0
DO_SCAN=0
DO_YIELDS=0
DO_LIMITS=0
DO_PLOTS=0

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRYRUN=1; shift;;
    --force)   FORCE=1; shift;;
    --gui)     GUI=1; shift;;

    --train)   DO_TRAIN=1; shift;;
    --apply)   DO_APPLY=1; shift;;
    --scan)    DO_SCAN=1; shift;;
    --yields)  DO_YIELDS=1; shift;;
    --limits)  DO_LIMITS=1; shift;;
    --plots)   DO_PLOTS=1; shift;;

    --all)
      DO_TRAIN=1; DO_APPLY=1; DO_SCAN=1; DO_YIELDS=1; DO_LIMITS=1; DO_PLOTS=1
      shift
      ;;
    -h|--help) usage; exit 0;;
    *)
      echo "[ERROR] Unknown arg: $1 (use -h for help)"
      exit 2
      ;;
  esac
done
run() {
  echo "+ $*"
  if [[ "${DRYRUN}" -eq 0 ]]; then
    "$@"
  fi
}

need_or_skip() {
  # args: output_file stepname
  local out="$1"
  local step="$2"
  if [[ "${FORCE}" -eq 1 ]]; then
    return 0
  fi
  if [[ -e "${out}" ]]; then
    echo "[SKIP] ${step}: output exists -> ${out}  (use --force to rerun)"
    return 1
  fi
  return 0
}

mkdir -p "${OUT_DIR}"

TMVA_ROOT="${OUT_DIR}/TMVA_${METHOD}.root"

if [[ "${DO_TRAIN}" -eq 1 ]]; then
  echo "[TRAIN] TMVA ${METHOD} -> ${TMVA_ROOT}"
  if need_or_skip "${TMVA_ROOT}" "train"; then
    GUI_FLAG=""
    if [[ "${GUI}" -eq 1 ]]; then
      GUI_FLAG="--gui"
    fi
    run python3 -m modules.train_tmva \
      --in-dir "${IN_DIR}" --tag "${TAG}" --tree "${TREE}" \
      --method "${METHOD}" --out-dir "${OUT_DIR}" \
      --out "TMVA_${METHOD}.root" \
      --sr-cut "${TRAIN_CUT}" \
      ${GUI_FLAG}
  fi
fi

if [[ "${DO_APPLY}" -eq 1 ]]; then
  echo "[APPLY] TMVA ${METHOD} -> *_withBDT*.root (Nominal + systs)"
  # existence check for nominal
  ONE_OUT="${OUT_DIR}/${TAG}_signal_withBDT.root"
  if need_or_skip "${ONE_OUT}" "apply"; then

    # list of systs (space-separated). Configurable from config.sh if you want.
    SYSTS="${SYSTS:-Nominal JESUp JESDown JERUp JERDown}"

    for S in ${SYSTS}; do
      echo "[APPLY] syst=${S}"
      run python3 -m modules.apply_tmva \
        --in-dir "${IN_DIR}" --out-dir "${OUT_DIR}" \
        --tag "${TAG}" --tree "${TREE}" \
        --method "${METHOD}" \
        --weights-root "${TMVA_ROOT}" \
        --bdt-branch "bdt_${METHOD}" \
        --syst "${S}"
    done
  fi
fi

if [[ "${DO_SCAN}" -eq 1 ]]; then
  echo "[SCAN] -> ${OUT_DIR}/scan_${METHOD}.csv"
  OUTCSV="${OUT_DIR}/scan_${METHOD}.csv"
  if need_or_skip "${OUTCSV}" "scan"; then
    run python3 -m modules.scan_cut \
      --workdir "${OUT_DIR}" \
      --tag "${TAG}" \
      --tree "${TREE}" \
      --sr-cut "${SR_CUT}" \
      --weight "weight" \
      --bdt-branch "bdt_${METHOD}" \
      --cut-min -1.0 \
      --cut-max 1.0 \
      --cut-step 0.01 \
      --out-csv "${OUTCSV}"
  fi
fi

if [[ "${DO_YIELDS}" -eq 1 ]]; then
  echo "[YIELDS] -> ${OUT_DIR}/yields_${METHOD}.json"
  OUTJSON="${OUT_DIR}/yields_${METHOD}.json"
  if need_or_skip "${OUTJSON}" "yields"; then
    run python3 -m modules.yields \
      --workdir "${OUT_DIR}" \
      --tag "${TAG}" \
      --tree "${TREE}" \
      --sr-cut "${SR_CUT}" \
      --weight "weight" \
      --bdt-branch "bdt_${METHOD}" \
      --scan-csv "${OUT_DIR}/scan_${METHOD}.csv" \
      --out-json "${OUTJSON}"
  fi
fi

# ...

if [[ "${DO_LIMITS}" -eq 1 ]]; then
  echo "[LIMITS] -> ${OUT_DIR}/limits_${METHOD}.json (SR + CR_top, with nuisances)"
  OUTLIM="${OUT_DIR}/limits_${METHOD}.json"
  if need_or_skip "${OUTLIM}" "limits"; then
    echo "[LIMITS] SR_CUT=${SR_CUT}"
    echo "[LIMITS] CR_CUT=${CR_CUT}"
    cmd=(python3 -m modules.limits_pyhf \
      --workdir "${OUT_DIR}" --tag "${TAG}" --tree "${TREE}" \
      --method "${METHOD}" \
      --bdt-branch "bdt_${METHOD}" \
      --weight-branch "weight" \
      --xlow -1.0 --xhigh 1.0 \
      --sr-cut "${SR_CUT}" \
      --cr-cut "${CR_CUT}" \
      --shape-systs "JES,JER" \
      --adaptive-bins --fine-bins 200 --bmin 50 --min-bins 5 \
      --no-signal-in-cr \
      --fallback-scan --scan-points 40 \
      --out "${OUTLIM}"
      )

    run "${cmd[@]}"
  fi
fi

if [[ "${DO_PLOTS}" -eq 1 ]]; then
  echo "[PLOTS] publication plots"
  # existence check: one output plot
  ONEPLOT="${OUT_DIR}/bdt_stack_SR_${METHOD}.pdf"
  if need_or_skip "${ONEPLOT}" "plots"; then
    run python3 -m modules.plot_results \
      --workdir "${OUT_DIR}" \
      --tag "${TAG}" \
      --method "${METHOD}" \
      --bins 30 \
      --xlow -1.0 --xhigh 1.0 \
      --signal-scale 5
  fi
fi

echo "[DONE] Pipeline finished."
echo "Outputs in: ${OUT_DIR}"
echo "  - ${OUT_DIR}/TMVA_${METHOD}.root"
echo "  - ${OUT_DIR}/*_withBDT.root"
echo "  - ${OUT_DIR}/scan_${METHOD}.csv"
echo "  - ${OUT_DIR}/yields_${METHOD}.json"
echo "  - ${OUT_DIR}/limits_${METHOD}.json"
echo "  - ${OUT_DIR}/bdt_stack_SR_${METHOD}.pdf/.png, scan_Z_${METHOD}.pdf/.png, limits_brazil_${METHOD}.pdf/.png"
