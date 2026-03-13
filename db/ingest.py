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
        "%m/%d/%Y %H:%M:%S",  # US format con time (CSV AdHoc)
        "%m/%d/%Y",           # US format
        "%d/%m/%Y",           # EU format
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
) -> int:
    """
    Importa dati dall'API AdHoc in maintenance_history.

    La deduplicazione è gestita dal vincolo UNIQUE a livello DB
    su (azienda, seriale_doc, targa, data_intervento).

    Args:
        data_start: Data inizio periodo (default: 01/01/2015)
        data_stop: Data fine periodo (default: oggi)

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

    # Debug: mostra prima riga per vedere nomi colonne
    if raw_data:
        first_row = raw_data[0]
        logger.info(f"Colonne CSV: {list(first_row.keys())}")
        logger.debug(f"Prima riga: {first_row}")

    # Prepara record per insert
    records = []
    skipped = {"no_targa": 0, "no_azienda": 0, "no_data": 0}

    for idx, row in enumerate(raw_data):
        # Normalizza chiavi a uppercase per compatibilità JSON/CSV
        row = {k.upper(): v for k, v in row.items()}
        targa = _clean_targa(row.get("TARGA", ""))
        azienda = _clean_azienda(row.get("AZIENDA", ""))
        data_intervento = _parse_date(row.get("DATA_INTERVENTO"))

        # Debug prima riga
        if idx == 0:
            logger.info(f"Prima riga - TARGA='{row.get('TARGA')}' -> '{targa}'")
            logger.info(f"Prima riga - AZIENDA='{row.get('AZIENDA')}' -> '{azienda}'")
            logger.info(f"Prima riga - DATA_INTERVENTO='{row.get('DATA_INTERVENTO')}' -> '{data_intervento}'")

        # Skip se mancano campi obbligatori
        if not targa:
            skipped["no_targa"] += 1
            continue
        if not azienda:
            skipped["no_azienda"] += 1
            continue
        if not data_intervento:
            skipped["no_data"] += 1
            continue

        # DATA_IMM può essere in vari campi
        data_imm = (
            _parse_date(row.get("DATA_IMM"))
            or _parse_date(row.get("DATA_IMM_MEZZO"))
            or _parse_date(row.get("DATA_IMM_CTR"))
        )

        records.append({
            "azienda": azienda,
            "seriale_doc": str(row.get("SERIALE_DOC", "")).strip(),
            "descrizione": row.get("DESCRIZIONE", ""),
            "dettaglio": row.get("DETTAGLIO", ""),
            "targa": targa,
            "data_intervento": data_intervento,
            "causale": row.get("CAUSALE", ""),
            "costo": float(row.get("COSTO", 0) or 0),
            "data_imm": data_imm,
        })

    logger.info(f"Record validi: {len(records)}")
    if any(skipped.values()):
        logger.info(f"Record scartati: {skipped}")

    if not records:
        logger.warning("Nessun record da inserire")
        return 0

    conn = get_connection()

    # Assicura che l'indice UNIQUE esista per la deduplicazione a livello DB
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_mh_unique
        ON maintenance_history(azienda, seriale_doc, targa, data_intervento)
    """)

    # Conta record prima dell'inserimento
    count_before = conn.execute(
        "SELECT COUNT(*) FROM maintenance_history"
    ).fetchone()[0]

    # INSERT OR IGNORE: il vincolo UNIQUE gestisce la deduplicazione
    conn.executemany("""
        INSERT OR IGNORE INTO maintenance_history
            (azienda, seriale_doc, descrizione, dettaglio, targa,
             data_intervento, causale, costo, data_imm)
        VALUES
            (:azienda, :seriale_doc, :descrizione, :dettaglio, :targa,
             :data_intervento, :causale, :costo, :data_imm)
    """, records)

    conn.commit()

    count_after = conn.execute(
        "SELECT COUNT(*) FROM maintenance_history"
    ).fetchone()[0]
    conn.close()

    inserted = count_after - count_before
    skipped_dupes = len(records) - inserted
    if skipped_dupes > 0:
        logger.info(f"Record duplicati ignorati: {skipped_dupes}")

    logger.info(f"Inseriti {inserted} nuovi record")
    print(f"Inseriti {inserted} nuovi record in maintenance_history")

    return inserted


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
