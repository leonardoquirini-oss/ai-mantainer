"""
Import dati da API AdHoc in SQLite.

Uso:
    from db.ingest import ingest_from_adhoc

    # Importa tutto lo storico
    ingest_from_adhoc()

    # Importa solo un periodo
    ingest_from_adhoc(date(2024, 1, 1), date(2024, 12, 31))
"""

import logging
from datetime import date, datetime
from typing import Optional, List, Dict, Any

from .init import get_connection, init_db

logger = logging.getLogger("maintenance-agent.ingest")


def _parse_date(value: Any) -> Optional[str]:
    """
    Converte un valore in formato DATE SQLite (YYYY-MM-DD).

    Gestisce:
    - datetime/date Python
    - stringhe ISO (2024-01-15T00:00:00)
    - stringhe date (2024-01-15, 15/01/2024)
    - None
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    value_str = str(value).strip()
    if not value_str or value_str.lower() == "none":
        return None

    # Prova vari formati
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value_str[:19], fmt).date().isoformat()
        except ValueError:
            continue

    return None


def _clean_targa(targa: str) -> str:
    """Pulisce e normalizza la targa."""
    return targa.strip().upper() if targa else ""


def _clean_azienda(azienda: str) -> str:
    """Pulisce e valida l'azienda."""
    azienda = azienda.strip().upper() if azienda else ""
    if azienda in ("G", "B", "C"):
        return azienda
    return ""


def ingest_from_adhoc(
    data_start: Optional[date] = None,
    data_stop: Optional[date] = None,
    skip_duplicates: bool = True
) -> int:
    """
    Importa dati dall'API AdHoc in maintenance_history.

    Args:
        data_start: Data inizio periodo (default: 01/01/2015)
        data_stop: Data fine periodo (default: oggi)
        skip_duplicates: Se True, evita di reinserire record già presenti

    Returns:
        Numero di record inseriti
    """
    # Import qui per evitare dipendenze circolari
    import sys
    sys.path.insert(0, str(__file__).rsplit("/db/", 1)[0])
    from agent.connectors import AdHocConnector

    if data_start is None:
        data_start = date(2015, 1, 1)
    if data_stop is None:
        data_stop = date.today()

    logger.info(f"Ingest da AdHoc: {data_start} - {data_stop}")

    # Inizializza DB se necessario
    init_db()

    # Fetch dati da API
    connector = AdHocConnector()
    try:
        raw_data = connector.get_manutenzioni(data_start, data_stop)
    finally:
        connector.close()

    logger.info(f"Ricevuti {len(raw_data)} record da API")

    # Prepara record per insert
    records = []
    for row in raw_data:
        targa = _clean_targa(row.get("TARGA", ""))
        azienda = _clean_azienda(row.get("AZIENDA", ""))
        data_intervento = _parse_date(row.get("DATA_INTERVENTO"))

        # Skip se mancano campi obbligatori
        if not targa or not azienda or not data_intervento:
            continue

        records.append({
            "azienda": azienda,
            "seriale_doc": str(row.get("SERIALE_DOC", "")).strip(),
            "descrizione": row.get("DESCRIZIONE", ""),
            "dettaglio": row.get("DETTAGLIO", ""),
            "targa": targa,
            "data_intervento": data_intervento,
            "causale": row.get("CAUSALE", ""),
            "costo": float(row.get("COSTO", 0) or 0),
            "data_imm": _parse_date(row.get("DATA_IMM")),
        })

    logger.info(f"Record validi: {len(records)}")

    if not records:
        logger.warning("Nessun record da inserire")
        return 0

    conn = get_connection()

    if skip_duplicates:
        # Recupera chiavi esistenti per deduplicazione
        existing = set()
        cursor = conn.execute("""
            SELECT azienda, seriale_doc, targa, data_intervento
            FROM maintenance_history
        """)
        for row in cursor:
            existing.add((row[0], row[1], row[2], row[3]))

        # Filtra solo nuovi record
        new_records = [
            r for r in records
            if (r["azienda"], r["seriale_doc"], r["targa"], r["data_intervento"]) not in existing
        ]
        logger.info(f"Record già presenti: {len(records) - len(new_records)}")
    else:
        new_records = records

    if not new_records:
        logger.info("Nessun nuovo record da inserire")
        conn.close()
        return 0

    # Insert batch
    conn.executemany("""
        INSERT INTO maintenance_history
            (azienda, seriale_doc, descrizione, dettaglio, targa,
             data_intervento, causale, costo, data_imm)
        VALUES
            (:azienda, :seriale_doc, :descrizione, :dettaglio, :targa,
             :data_intervento, :causale, :costo, :data_imm)
    """, new_records)

    conn.commit()
    conn.close()

    logger.info(f"Inseriti {len(new_records)} nuovi record")
    print(f"Inseriti {len(new_records)} nuovi record in maintenance_history")

    return len(new_records)


def get_import_stats() -> Dict[str, Any]:
    """
    Restituisce statistiche sull'import.
    """
    conn = get_connection()

    stats = {}

    # Totale record
    stats["total_records"] = conn.execute(
        "SELECT COUNT(*) FROM maintenance_history"
    ).fetchone()[0]

    # Per azienda
    rows = conn.execute("""
        SELECT azienda, COUNT(*) as count
        FROM maintenance_history
        GROUP BY azienda
        ORDER BY count DESC
    """).fetchall()
    stats["per_azienda"] = {r[0]: r[1] for r in rows}

    # Range date
    row = conn.execute("""
        SELECT
            MIN(data_intervento) as min_date,
            MAX(data_intervento) as max_date
        FROM maintenance_history
    """).fetchone()
    stats["date_range"] = {"min": row[0], "max": row[1]}

    # Veicoli unici
    stats["unique_vehicles"] = conn.execute(
        "SELECT COUNT(DISTINCT targa) FROM maintenance_history"
    ).fetchone()[0]

    conn.close()
    return stats


if __name__ == "__main__":
    # Esegui import completo
    import sys
    logging.basicConfig(level=logging.INFO)

    # Aggiungi path progetto
    project_root = str(__file__).rsplit("/db/", 1)[0]
    sys.path.insert(0, project_root)

    n = ingest_from_adhoc()
    print(f"\nStatistiche:")
    stats = get_import_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
