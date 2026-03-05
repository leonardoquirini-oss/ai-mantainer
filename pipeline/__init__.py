"""
Pipeline per calcolo feature e scoring.

Moduli:
- features: calcolo feature per veicoli (km, intervalli, costi)
"""

from .features import compute_km_features, compute_all_features, update_vehicle_features

__all__ = [
    "compute_km_features",
    "compute_all_features",
    "update_vehicle_features",
]
