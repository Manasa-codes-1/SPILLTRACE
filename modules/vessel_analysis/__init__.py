"""
Module 4 - Vessel Investigation

Public interface for other SPILLTRACE modules (dashboard, Module 5
ranking, alerts) to use. Always import from here, not from the
internal files directly - that way, internal refactors to this module
won't ever break anything that depends on it.
"""

from .models import OriginEstimate, VesselPing, CandidateVessel
from .ais_loader import load_ais_csv, dataframe_to_pings
from .candidate_filter import find_candidate_vessels
from .module3_connector import build_origin_estimate_from_drift_results
from .module3_json_loader import load_module3_output

__all__ = [
    "OriginEstimate",
    "VesselPing",
    "CandidateVessel",
    "load_ais_csv",
    "dataframe_to_pings",
    "find_candidate_vessels",
    "build_origin_estimate",
    "load_module3_output",
]
