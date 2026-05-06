from dataclasses import dataclass
from typing import Dict, List
import os

# --- These MUST match the variable names used in TMVA training (weights.xml) ---
TMVA_VARIABLES: List[str] = [
    "MET",
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
]

@dataclass
class Samples:
    signal: str
    backgrounds: Dict[str, str]

def sample_files(in_dir: str, tag: str) -> Samples:
    """
    Build full paths to your cutcount ROOT files.
    Expected filenames like:
      {tag}_signal_cutcount.root
      {tag}_tt_semilep_cutcount.root
      ...
    """
    in_dir = os.path.expanduser(in_dir)

    signal = os.path.join(in_dir, f"{tag}_signal_cutcount.root")

    backgrounds = {
        "tt_semilep": os.path.join(in_dir, f"{tag}_tt_semilep_cutcount.root"),
        "tt_dilep":   os.path.join(in_dir, f"{tag}_tt_dilep_cutcount.root"),
        "tt_had":     os.path.join(in_dir, f"{tag}_tt_had_cutcount.root"),
        "st_tW":      os.path.join(in_dir, f"{tag}_st_tW_cutcount.root"),
        "st_tch_top": os.path.join(in_dir, f"{tag}_st_tch_top_cutcount.root"),
        "st_tch_tbar":os.path.join(in_dir, f"{tag}_st_tch_tbar_cutcount.root"),
        "wjets":      os.path.join(in_dir, f"{tag}_wjets_cutcount.root"),
        "zvvjets":    os.path.join(in_dir, f"{tag}_zvvjets_cutcount.root"),
    }

    return Samples(signal=signal, backgrounds=backgrounds)
