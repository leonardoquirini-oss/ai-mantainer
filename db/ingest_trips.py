"""
Import viaggi da CSV/Excel in SQLite.

Uso:
    from db.ingest_trips import ingest_trips

    # Da file CSV o Excel
    ingest_trips("viaggi.csv")
    ingest_trips("viaggi.xlsx")
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .init import get_connection, init_db

logger = logging.getLogger("maintenance-agent.ingest_trips")

# Mapping colonne fonte → DB
COLUMN_MAP = {
    'BG': 'bg',
    'TargaMotrice': 'targa_motrice',
    'TargaSemirimorchio': 'targa_semirimorchio',
    'Km': 'km',
    'DataViaggio': 'data_viaggio',
    'Data': 'data_viaggio',  # alias
}


def _parse_date(value: Any) -> Optional[str]:
    """Converte un valore in formato DATE SQLite (YYYY-MM-DD)."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    value_str = str(value).strip()
    if not value_str or value_str.lower() in ("none", "nan", "nat"):
        return None

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",  # TIR format EU
        "%m/%d/%Y %H:%M:%S",  # TIR format US
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value_str[:19], fmt).date().isoformat()
        except ValueError:
            continue

    return None


def _clean_targa(targa: Any) -> Optional[str]:
    """Pulisce e normalizza la targa. Ritorna None se vuota/invalida."""
    if targa is None:
        return None
    targa_str = str(targa).strip().upper()
    if not targa_str or targa_str in ("NAN", "NONE", "NULL", ""):
        return None
    return targa_str


def ingest_trips(filepath: str) -> Dict[str, int]:
    """
    Importa viaggi da file CSV o Excel.

    Colonne attese:
    - BG: ID univoco del viaggio
    - TargaMotrice: targa della motrice
    - TargaSemirimorchio: targa del semirimorchio (opzionale)
    - Km: km percorsi
    - DataViaggio o Data: data del viaggio

    Args:
        filepath: Path al file CSV o Excel

    Returns:
        Dict con conteggi: inserted, skipped, invalid
    """
    import pandas as pd

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File non trovato: {filepath}")

    logger.info(f"Caricamento viaggi da {filepath}")

    # Leggi file
    if filepath.suffix.lower() == '.csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    # Rinomina colonne
    df = df.rename(columns=COLUMN_MAP)

    # Verifica colonne obbligatorie
    required = ['bg', 'targa_motrice', 'km', 'data_viaggio']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti: {missing}")

    # Pulizia dati
    df['bg'] = df['bg'].astype(str).str.strip()
    df['targa_motrice'] = df['targa_motrice'].apply(_clean_targa)
    df['targa_semirimorchio'] = df['targa_semirimorchio'].apply(_clean_targa) if 'targa_semirimorchio' in df.columns else None
    df['data_viaggio'] = df['data_viaggio'].apply(_parse_date)
    df['km'] = pd.to_numeric(df['km'], errors='coerce')

    # Inizializza DB se necessario
    init_db()

    conn = get_connection()
    inserted = 0
    skipped = 0
    invalid = 0

    for _, row in df.iterrows():
        # Valida record
        bg = row.get('bg')
        targa_motrice = row.get('targa_motrice')
        data_viaggio = row.get('data_viaggio')
        km = row.get('km')

        if not bg or not targa_motrice or not data_viaggio or pd.isna(km) or km <= 0:
            invalid += 1
            continue

        try:
            conn.execute("""
                INSERT INTO trips (bg, targa_motrice, targa_semirimorchio, km, data_viaggio)
                VALUES (?, ?, ?, ?, ?)
            """, (
                bg,
                targa_motrice,
                row.get('targa_semirimorchio'),
                float(km),
                data_viaggio
            ))
            inserted += 1
        except Exception as e:
            # BG duplicato o altro errore
            skipped += 1
            logger.debug(f"Skip BG {bg}: {e}")

    conn.commit()
    conn.close()

    result = {
        "inserted": inserted,
        "skipped": skipped,
        "invalid": invalid,
        "total_processed": len(df)
    }

    logger.info(f"Import viaggi completato: {result}")
    print(f"Viaggi inseriti: {inserted} | Duplicati saltati: {skipped} | Invalidi: {invalid}")

    return result


def ingest_from_tir(
    data_start: Optional[date] = None,
    data_stop: Optional[date] = None
) -> Dict[str, int]:
    """
    Importa viaggi dall'API TIR in tabella trips.

    Usa il template "viaggi" con outputFormat="csv" per efficienza.

    Args:
        data_start: Data inizio periodo (default: 01/01/2020)
        data_stop: Data fine periodo (default: oggi)

    Returns:
        Dict con conteggi: inserted, skipped, invalid
    """
    # Import qui per evitare dipendenze circolari
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agent.connectors import TIRConnector

    if data_start is None:
        data_start = date(2020, 1, 1)
    if data_stop is None:
        data_stop = date.today()

    logger.info(f"Ingest viaggi da TIR: {data_start} - {data_stop}")

    # Inizializza DB se necessario
    init_db()

    # Fetch dati da API TIR
    connector = TIRConnector()
    try:
        raw_data = connector.get_viaggi(data_start, data_stop, output_format="csv")
    finally:
        connector.close()

    logger.info(f"Ricevuti {len(raw_data)} viaggi da TIR API")

    if not raw_data:
        logger.warning("Nessun viaggio ricevuto da TIR")
        return {"inserted": 0, "skipped": 0, "invalid": 0, "total_processed": 0}

    conn = get_connection()
    inserted = 0
    skipped = 0
    invalid = 0

    # Mapping colonne TIR → DB
    # TIR: BG, TargaMotrice, TargaSemirimorchio, Km, Data
    for row in raw_data:
        try:
            # Estrai campi (supporta varianti di naming)
            bg = str(row.get('BG') or row.get('bg') or '').strip()
            targa_motrice = _clean_targa(row.get('TargaMotrice') or row.get('targa_motrice'))
            targa_semirimorchio = _clean_targa(row.get('TargaSemirimorchio') or row.get('targa_semirimorchio'))
            data_viaggio = _parse_date(row.get('Data') or row.get('DataViaggio') or row.get('data_viaggio'))

            # Parse km
            km_raw = row.get('Km') or row.get('km') or 0
            try:
                km = float(str(km_raw).replace(',', '.'))
            except (ValueError, TypeError):
                km = 0

            # Valida record
            if not bg or not targa_motrice or not data_viaggio or km <= 0:
                invalid += 1
                continue

            # Insert
            conn.execute("""
                INSERT INTO trips (bg, targa_motrice, targa_semirimorchio, km, data_viaggio)
                VALUES (?, ?, ?, ?, ?)
            """, (
                bg,
                targa_motrice,
                targa_semirimorchio,
                km,
                data_viaggio
            ))
            inserted += 1

        except Exception as e:
            # BG duplicato o altro errore
            skipped += 1
            logger.debug(f"Skip viaggio: {e}")

    conn.commit()
    conn.close()

    result = {
        "inserted": inserted,
        "skipped": skipped,
        "invalid": invalid,
        "total_processed": len(raw_data)
    }

    logger.info(f"Import viaggi TIR completato: {result}")
    print(f"Viaggi inseriti: {inserted} | Duplicati saltati: {skipped} | Invalidi: {invalid}")

    return result


def get_trips_stats() -> Dict[str, Any]:
    """Restituisce statistiche sui viaggi importati."""
    conn = get_connection()

    stats = {}

    # Totale viaggi
    stats["total_trips"] = conn.execute(
        "SELECT COUNT(*) FROM trips"
    ).fetchone()[0]

    # Totale km
    stats["total_km"] = conn.execute(
        "SELECT COALESCE(SUM(km), 0) FROM trips"
    ).fetchone()[0]

    # Range date
    row = conn.execute("""
        SELECT
            MIN(data_viaggio) as min_date,
            MAX(data_viaggio) as max_date
        FROM trips
    """).fetchone()
    stats["date_range"] = {"min": row[0], "max": row[1]}

    # Motrici uniche
    stats["unique_motrici"] = conn.execute(
        "SELECT COUNT(DISTINCT targa_motrice) FROM trips"
    ).fetchone()[0]

    # Semirimorchi unici
    stats["unique_semirimorchi"] = conn.execute(
        "SELECT COUNT(DISTINCT targa_semirimorchio) FROM trips WHERE targa_semirimorchio IS NOT NULL"
    ).fetchone()[0]

    conn.close()
    return stats


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Uso: python -m db.ingest_trips <file.csv|file.xlsx>")
        sys.exit(1)

    result = ingest_trips(sys.argv[1])
    print(f"\nStatistiche:")
    stats = get_trips_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
