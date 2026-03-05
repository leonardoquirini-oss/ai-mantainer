"""
Connettori per API esterne.

- AdHocConnector: API manutenzioni (192.168.0.12:9100)
- TIRConnector: API viaggi/km (192.168.0.12:9090)
"""

from .adhoc_connector import AdHocConnector, get_adhoc_connector
from .tir_connector import TIRConnector, get_tir_connector

__all__ = [
    "AdHocConnector",
    "get_adhoc_connector",
    "TIRConnector",
    "get_tir_connector",
]
