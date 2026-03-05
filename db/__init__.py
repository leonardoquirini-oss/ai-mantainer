"""
Database SQLite per storico manutenzioni.

Moduli:
- init: connessione e inizializzazione DB
- ingest: import dati manutenzioni da API AdHoc
- ingest_trips: import viaggi da CSV/Excel
"""

from .init import get_connection, init_db, DB_PATH, get_db_stats
from .ingest import ingest_from_adhoc, get_import_stats
from .ingest_trips import ingest_trips, ingest_from_tir, get_trips_stats

__all__ = [
    "get_connection",
    "init_db",
    "DB_PATH",
    "get_db_stats",
    "ingest_from_adhoc",
    "get_import_stats",
    "ingest_trips",
    "ingest_from_tir",
    "get_trips_stats",
]
