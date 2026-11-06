"""Experimental marker-based evidence; not certified data removal or DP."""
from pathlib import Path as _Path

# Generic research verification stays here; device modules live in system/.
__path__.append(str(_Path(__file__).resolve().parents[1] / "system" / "verification"))
del _Path
