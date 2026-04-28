from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import awkward as ak
import uproot


@dataclass(frozen=True)
class DelphesBranches:
    jet_pt: str = "Jet.PT"
    jet_eta: str = "Jet.Eta"
    jet_phi: str = "Jet.Phi"
    jet_mass: str = "Jet.Mass"
    jet_btag: str = "Jet.BTag"

    ele_pt: str = "Electron.PT"
    mu_pt: str = "Muon.PT"

    met: str = "MissingET.MET"
    met_phi: str = "MissingET.Phi"


def _empty_jagged(n_events: int, dtype: str = "float32") -> ak.Array:
    """
    Return an empty jagged array with one list per event.
    Shape: n_events * var * dtype
    """
    # ak.Array([[]] * n_events) is fine: lists are immutable to ak; no aliasing issues for content
    return ak.Array([[] for _ in range(int(n_events))])


def _empty_flat(n_events: int, fill: float = 0.0) -> ak.Array:
    """Return a flat array of length n_events filled with `fill`."""
    return ak.Array([fill for _ in range(int(n_events))])


def iter_events(
    filename: str,
    tree_name: str = "Delphes",
    branches: Optional[DelphesBranches] = None,
    step_size: int = 50_000,
    max_events: int = -1,
) -> Iterator[Dict[str, ak.Array]]:
    """
    Yield chunks of awkward arrays from a Delphes ROOT file using uproot.

    This function intentionally does *not* require libDelphes / ExRootAnalysis.
    It reads branches directly from the Delphes TTree and normalizes them to
    simple keys expected by the analysis layer.

    Robustness:
      - If a branch is missing, we substitute an empty jagged array (for collections)
        or a flat zero array (for MET scalars), with the correct number of events.
    """
    br = branches or DelphesBranches()

    with uproot.open(filename) as f:
        tree = f[tree_name]

        wanted = [
            br.jet_pt, br.jet_eta, br.jet_phi, br.jet_mass, br.jet_btag,
            br.ele_pt, br.mu_pt,
            br.met, br.met_phi,
        ]

        n_total = int(tree.num_entries)
        if max_events and max_events > 0:
            n_total = min(n_total, int(max_events))

        for start in range(0, n_total, int(step_size)):
            stop = min(n_total, start + int(step_size))

            arrays = tree.arrays(wanted, entry_start=start, entry_stop=stop, library="ak")
            fields = set(arrays.fields)

            # Determine n_events in this chunk from any available field
            n_events = 0
            if len(fields) > 0:
                # pick first field
                first = next(iter(fields))
                n_events = len(arrays[first])

            def get_jagged(field: str) -> ak.Array:
                return arrays[field] if field in fields else _empty_jagged(n_events)

            def get_flat(field: str) -> ak.Array:
                return arrays[field] if field in fields else _empty_flat(n_events, 0.0)
            def get_flat(field: str) -> ak.Array:
                if field not in fields:
                    return _empty_flat(n_events, 0.0)

                arr = arrays[field]

    # If it's a jagged array (e.g. MissingET.MET is 1-element per event),
    # take first element per event -> flat.
    # ak.firsts returns None for empty lists; fill with 0.
                try:
        # If arr has axis=1 (list per event), this succeeds.
                    flat = ak.fill_none(ak.firsts(arr), 0.0)
                    return flat
                except Exception:
        # If arr is already flat, return it
                    return arr
            yield {
                "jet_pt": get_jagged(br.jet_pt),
                "jet_eta": get_jagged(br.jet_eta),
                "jet_phi": get_jagged(br.jet_phi),
                "jet_mass": get_jagged(br.jet_mass),  # if missing -> empty jagged (consistent event shape)
                "jet_btag": get_jagged(br.jet_btag),
                "ele_pt": get_jagged(br.ele_pt),
                "mu_pt": get_jagged(br.mu_pt),
                "met": get_flat(br.met),
                "met_phi": get_flat(br.met_phi),
                "start": start,
            }
