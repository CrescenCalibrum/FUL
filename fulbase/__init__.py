"""Experiment package with stable imports for relocated system modules."""
from pathlib import Path as _Path

# Keep public names such as fulbase.equipment_models stable after relocation.
# Isolated code bundles may still place these modules directly beside this file.
__path__.append(str(_Path(__file__).resolve().parent / "system"))
del _Path
