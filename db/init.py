"""
Inizializzazione database SQLite per manutenzioni.

Uso:
    from db.init import get_connection, init_db

    # Inizializza DB (crea tabelle se non esistono)
    init_db()

    # Ottieni connessione
    conn = get_connection()
"""

import sqlite3
from pathlib import Path

# Path del database
DB_PATH = Path(__file__).parent.parent / "data" / "maintenance.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    """
    Restituisce una connessione SQLite con le impostazioni consigliate.

    - row_factory = sqlite3.Row per accesso per nome colonna
    - WAL mode per scritture concorrenti più sicure
    - foreign_keys attive
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """
    Inizializza il database creando le tabelle se non esistono.

    Legge lo schema da db/schema.sql e lo esegue.
    """
    # Crea directory data/ se non esiste
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection()

    # Leggi ed esegui schema
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())

    conn.commit()
    conn.close()

    print(f"DB inizializzato: {DB_PATH}")


def get_db_stats() -> dict:
    """
    Restituisce statistiche sul database.
    """
    conn = get_connection()

    stats = {}

    # Conta record per tabella
    for table in ["maintenance_history", "vehicle_features", "risk_scores"]:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = count
        except sqlite3.OperationalError:
            stats[table] = 0

    # Intervallo date maintenance_history
    try:
        row = conn.execute("""
            SELECT
                MIN(data_intervento) as min_date,
                MAX(data_intervento) as max_date,
                COUNT(DISTINCT targa) as n_veicoli
            FROM maintenance_history
        """).fetchone()
        stats["date_range"] = {
            "min": row["min_date"],
            "max": row["max_date"]
        }
        stats["n_veicoli"] = row["n_veicoli"]
    except sqlite3.OperationalError:
        pass

    conn.close()
    return stats


if __name__ == "__main__":
    init_db()
