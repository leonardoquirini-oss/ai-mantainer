"""
Costruzione del target per il modello ML.

Il target è binario: "questa targa avrà un intervento di tipo X entro N giorni?"

Granularità: (targa, tipo_guasto, data_intervento)
Un intervento con tag multipli genera più righe nel dataset.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# Setup path per import
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from db.init import get_connection
from agent.utils.categorizzatore import categorizza_riga, categoria_to_tipo_guasto

logger = logging.getLogger("maintenance-agent.scoring.target")

# Tipi guasto disponibili (mappati dal categorizzatore)
TIPI_GUASTO = [
    'freni', 'pneumatici', 'motore', 'elettrico', 'carrozzeria',
    'sospensioni', 'idraulico', 'revisione', 'altro', 'tagliando'
]


def load_classified_interventions() -> pd.DataFrame:
    """
    Carica gli interventi deduplicati e classificati per tipo_guasto.

    Deduplicazione per (azienda, seriale_doc, targa):
    più righe con lo stesso seriale rappresentano lo stesso intervento
    (righe contabili distinte — manodopera, ricambi, ecc.).

    Un intervento con tag multipli genera più righe (una per tipo_guasto).

    Returns:
        DataFrame con colonne: azienda, targa, data_intervento, tipo_guasto, costo_totale
    """
    conn = get_connection()

    logger.info("Caricamento interventi deduplicati...")

    df = pd.read_sql("""
        SELECT
            azienda,
            seriale_doc,
            targa,
            MIN(data_intervento) AS data_intervento,
            SUM(costo)           AS costo_totale,
            -- Concatena descrizione e dettaglio per il categorizzatore
            GROUP_CONCAT(DISTINCT descrizione) AS descrizione,
            GROUP_CONCAT(DISTINCT dettaglio)   AS dettaglio
        FROM maintenance_history
        GROUP BY azienda, seriale_doc, targa
        ORDER BY targa, data_intervento
    """, conn)
    conn.close()

    logger.info(f"Caricati {len(df)} interventi deduplicati")

    df['data_intervento'] = pd.to_datetime(df['data_intervento'])

    # Applica il categorizzatore e mappa a tipo_guasto
    righe_espanse = []
    for _, row in df.iterrows():
        categorie = categorizza_riga(row['descrizione'] or '', row['dettaglio'] or '')
        tipi_guasto = set(categoria_to_tipo_guasto(c) for c in categorie)

        for tipo in tipi_guasto:
            righe_espanse.append({
                'azienda': row['azienda'],
                'targa': row['targa'],
                'data_intervento': row['data_intervento'],
                'tipo_guasto': tipo,
                'costo_totale': row['costo_totale'],
            })

    result = pd.DataFrame(righe_espanse)
    logger.info(f"Espansi a {len(result)} righe (per tipo_guasto)")

    return result


def build_target(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    Per ogni (targa, tipo_guasto) alla data X, il target è 1
    se esiste un intervento dello stesso tipo entro horizon_days giorni.

    Args:
        df: DataFrame da load_classified_interventions()
        horizon_days: Orizzonte temporale (7, 30, 90)

    Returns:
        DataFrame con colonna aggiuntiva fail_{horizon}d
    """
    df = df.sort_values(['targa', 'tipo_guasto', 'data_intervento']).copy()

    # Per ogni (targa, tipo_guasto), trova la data del prossimo intervento
    df['next_intervention_date'] = df.groupby(
        ['targa', 'tipo_guasto']
    )['data_intervento'].shift(-1)

    df['days_to_next'] = (df['next_intervention_date'] - df['data_intervento']).dt.days
    df[f'fail_{horizon_days}d'] = (df['days_to_next'] <= horizon_days).astype(int)

    return df


def build_training_dataset(
    min_date: Optional[str] = None,
    max_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Costruisce il dataset completo per training.

    Carica interventi, applica categorizzazione, costruisce target
    per i tre orizzonti (7, 30, 90 giorni).

    Args:
        min_date: Data minima (es. '2018-01-01')
        max_date: Data massima (es. '2025-06-30')

    Returns:
        DataFrame pronto per training con target multipli
    """
    df = load_classified_interventions()

    # Filtra per date
    if min_date:
        df = df[df['data_intervento'] >= pd.Timestamp(min_date)]
    if max_date:
        df = df[df['data_intervento'] <= pd.Timestamp(max_date)]

    logger.info(f"Dataset filtrato: {len(df)} righe ({min_date} - {max_date})")

    # Costruisci target per i tre orizzonti
    df = build_target(df, 7)    # urgente
    df = build_target(df, 30)   # pianificabile
    df = build_target(df, 90)   # strategico

    # Escludi ultime righe per (targa, tipo_guasto) — futuro ignoto
    df = df.dropna(subset=['next_intervention_date'])

    logger.info(f"Dataset finale per training: {len(df)} righe")

    return df


def get_target_stats(df: pd.DataFrame) -> dict:
    """
    Statistiche sul dataset di training.

    Args:
        df: DataFrame da build_training_dataset()

    Returns:
        Dict con statistiche
    """
    stats = {
        'total_rows': len(df),
        'unique_targhe': df['targa'].nunique(),
        'unique_tipi_guasto': df['tipo_guasto'].nunique(),
        'per_tipo_guasto': df['tipo_guasto'].value_counts().to_dict(),
        'class_balance': {},
    }

    for horizon in [7, 30, 90]:
        col = f'fail_{horizon}d'
        if col in df.columns:
            stats['class_balance'][f'{horizon}d'] = {
                'positive': int(df[col].sum()),
                'negative': int((df[col] == 0).sum()),
                'ratio': round(df[col].mean(), 3),
            }

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    df = build_training_dataset(min_date='2020-01-01')
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColonne: {df.columns.tolist()}")

    stats = get_target_stats(df)
    print(f"\nStatistiche:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\nPrime 10 righe:")
    print(df[['targa', 'tipo_guasto', 'data_intervento', 'days_to_next', 'fail_7d', 'fail_30d', 'fail_90d']].head(10))
