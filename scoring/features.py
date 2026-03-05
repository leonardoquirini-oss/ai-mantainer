"""
Feature engineering per il modello ML di scoring.

build_features() costruisce il dataset di feature per tutte le targhe attive.
Legge da SQLite: maintenance_history e vehicle_km (vista).
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

# Setup path per import
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from db.init import get_connection

logger = logging.getLogger("maintenance-agent.scoring.features")

# Feature columns per training
FEATURE_COLS = [
    'days_since_last',
    'avg_days_between',
    'days_ratio',
    'vehicle_age_days',
    'interventions_last_90d',
    'recurrence_12m',
    'cost_trend',
    'km_dal_ultimo_intervento',
    'avg_km_per_intervento',
    'km_ratio',
    'km_ultimi_30d',
    'km_ultimi_90d',
    'km_stimati_settimana',
    'month_sin',
    'month_cos',
]


def build_features(reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Costruisce il dataset di feature per tutte le targhe attive.

    Legge da SQLite: maintenance_history e vehicle_km (vista).

    Args:
        reference_date: Data di riferimento per il calcolo. Default: oggi.

    Returns:
        DataFrame con una riga per targa e tutte le feature.
    """
    conn = get_connection()
    ref = pd.Timestamp(reference_date or pd.Timestamp.today())

    logger.info(f"Building features con reference_date={ref.date()}")

    # --- Storico manutenzioni ---
    maint = pd.read_sql("""
        SELECT targa, data_intervento, costo, data_imm, azienda
        FROM maintenance_history
        ORDER BY targa, data_intervento
    """, conn)
    maint['data_intervento'] = pd.to_datetime(maint['data_intervento'])
    maint['data_imm'] = pd.to_datetime(maint['data_imm'])

    # --- Km reali da vehicle_km (vista che unifica motrice + semirimorchio) ---
    trips = pd.read_sql("""
        SELECT targa, data_viaggio, km
        FROM vehicle_km
        ORDER BY targa, data_viaggio
    """, conn)
    trips['data_viaggio'] = pd.to_datetime(trips['data_viaggio'])

    conn.close()

    # Targhe attive = tutte quelle presenti nel DB
    all_targhe = pd.Series(
        pd.concat([maint['targa'], trips['targa']]).unique(),
        name='targa'
    )

    logger.info(f"Trovate {len(all_targhe)} targhe uniche")

    features = []
    for targa in all_targhe:
        m = maint[maint['targa'] == targa].sort_values('data_intervento')
        t = trips[trips['targa'] == targa].sort_values('data_viaggio')

        row = {'targa': targa}

        # Azienda (prendi la più recente)
        if not m.empty:
            row['azienda'] = m['azienda'].iloc[-1]
        else:
            row['azienda'] = None

        # --- Feature temporali da manutenzioni ---
        if not m.empty:
            last_date = m['data_intervento'].max()
            row['days_since_last'] = (ref - last_date).days
            row['avg_days_between'] = m['data_intervento'].diff().dt.days.mean()
            row['days_ratio'] = row['days_since_last'] / max(row['avg_days_between'] or 1, 1)

            # Età veicolo
            if pd.notna(m['data_imm'].iloc[0]):
                row['vehicle_age_days'] = (ref - m['data_imm'].iloc[0]).days
            else:
                row['vehicle_age_days'] = None

            # Interventi recenti
            row['interventions_last_90d'] = int((m['data_intervento'] >= ref - pd.Timedelta(days=90)).sum())
            row['recurrence_12m'] = int((m['data_intervento'] >= ref - pd.Timedelta(days=365)).sum())

            # Trend costo
            cost_hist = m['costo'].mean()
            cost_3m = m[m['data_intervento'] >= ref - pd.Timedelta(days=90)]['costo'].mean()
            row['cost_trend'] = (cost_3m / cost_hist) if cost_hist and pd.notna(cost_3m) else None

            # Km dal ultimo intervento
            if not t.empty:
                km_post = t[t['data_viaggio'] >= last_date]['km'].sum()
                row['km_dal_ultimo_intervento'] = round(float(km_post), 1)

                # Km medi storici per intervallo tra manutenzioni
                if len(m) >= 2:
                    km_per_intervallo = []
                    for i in range(1, len(m)):
                        d_start = m['data_intervento'].iloc[i - 1]
                        d_end = m['data_intervento'].iloc[i]
                        km_int = t[(t['data_viaggio'] >= d_start) &
                                   (t['data_viaggio'] < d_end)]['km'].sum()
                        km_per_intervallo.append(km_int)
                    avg_km = np.mean(km_per_intervallo) if km_per_intervallo else None
                    row['avg_km_per_intervento'] = avg_km
                    row['km_ratio'] = (km_post / avg_km) if avg_km and avg_km > 0 else None
                else:
                    row['avg_km_per_intervento'] = None
                    row['km_ratio'] = None
            else:
                row['km_dal_ultimo_intervento'] = None
                row['avg_km_per_intervento'] = None
                row['km_ratio'] = None
        else:
            row.update({
                'days_since_last': None, 'avg_days_between': None, 'days_ratio': None,
                'vehicle_age_days': None, 'interventions_last_90d': 0, 'recurrence_12m': 0,
                'cost_trend': None, 'km_dal_ultimo_intervento': None,
                'avg_km_per_intervento': None, 'km_ratio': None,
            })

        # --- Feature km da viaggi ---
        if not t.empty:
            cutoff_30 = ref - pd.Timedelta(days=30)
            cutoff_90 = ref - pd.Timedelta(days=90)
            recent_90 = t[t['data_viaggio'] >= cutoff_90]

            row['km_ultimi_30d'] = round(float(t[t['data_viaggio'] >= cutoff_30]['km'].sum()), 1)
            row['km_ultimi_90d'] = round(float(recent_90['km'].sum()), 1)
            row['km_stimati_settimana'] = round(float(recent_90['km'].sum()) / 90 * 7, 1) if not recent_90.empty else None
            row['km_totali_storici'] = round(float(t['km'].sum()), 1)
        else:
            row.update({
                'km_ultimi_30d': None, 'km_ultimi_90d': None,
                'km_stimati_settimana': None, 'km_totali_storici': None,
            })

        # --- Stagionalità ciclica ---
        row['month_sin'] = np.sin(2 * np.pi * ref.month / 12)
        row['month_cos'] = np.cos(2 * np.pi * ref.month / 12)

        features.append(row)

    df = pd.DataFrame(features)
    logger.info(f"Costruite feature per {len(df)} targhe")

    return df


def build_features_for_targa(targa: str, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Costruisce le feature per una singola targa.

    Versione ottimizzata per inference runtime.

    Args:
        targa: Targa del veicolo
        reference_date: Data di riferimento

    Returns:
        DataFrame con una riga (la targa) e tutte le feature.
    """
    conn = get_connection()
    ref = pd.Timestamp(reference_date or pd.Timestamp.today())
    targa_clean = targa.strip().upper()

    # Manutenzioni per questa targa
    maint = pd.read_sql("""
        SELECT data_intervento, costo, data_imm, azienda
        FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
        ORDER BY data_intervento
    """, conn, params=(targa_clean,))
    maint['data_intervento'] = pd.to_datetime(maint['data_intervento'])
    maint['data_imm'] = pd.to_datetime(maint['data_imm'])

    # Viaggi per questa targa
    trips = pd.read_sql("""
        SELECT data_viaggio, km
        FROM vehicle_km
        WHERE UPPER(TRIM(targa)) = ?
        ORDER BY data_viaggio
    """, conn, params=(targa_clean,))
    trips['data_viaggio'] = pd.to_datetime(trips['data_viaggio'])

    conn.close()

    row = {'targa': targa_clean}

    if not maint.empty:
        row['azienda'] = maint['azienda'].iloc[-1]
        last_date = maint['data_intervento'].max()
        row['days_since_last'] = (ref - last_date).days
        row['avg_days_between'] = maint['data_intervento'].diff().dt.days.mean()
        row['days_ratio'] = row['days_since_last'] / max(row['avg_days_between'] or 1, 1)

        if pd.notna(maint['data_imm'].iloc[0]):
            row['vehicle_age_days'] = (ref - maint['data_imm'].iloc[0]).days
        else:
            row['vehicle_age_days'] = None

        row['interventions_last_90d'] = int((maint['data_intervento'] >= ref - pd.Timedelta(days=90)).sum())
        row['recurrence_12m'] = int((maint['data_intervento'] >= ref - pd.Timedelta(days=365)).sum())

        cost_hist = maint['costo'].mean()
        cost_3m = maint[maint['data_intervento'] >= ref - pd.Timedelta(days=90)]['costo'].mean()
        row['cost_trend'] = (cost_3m / cost_hist) if cost_hist and pd.notna(cost_3m) else None

        if not trips.empty:
            km_post = trips[trips['data_viaggio'] >= last_date]['km'].sum()
            row['km_dal_ultimo_intervento'] = round(float(km_post), 1)

            if len(maint) >= 2:
                km_per_intervallo = []
                for i in range(1, len(maint)):
                    d_start = maint['data_intervento'].iloc[i - 1]
                    d_end = maint['data_intervento'].iloc[i]
                    km_int = trips[(trips['data_viaggio'] >= d_start) &
                                   (trips['data_viaggio'] < d_end)]['km'].sum()
                    km_per_intervallo.append(km_int)
                avg_km = np.mean(km_per_intervallo) if km_per_intervallo else None
                row['avg_km_per_intervento'] = avg_km
                row['km_ratio'] = (km_post / avg_km) if avg_km and avg_km > 0 else None
            else:
                row['avg_km_per_intervento'] = None
                row['km_ratio'] = None
        else:
            row['km_dal_ultimo_intervento'] = None
            row['avg_km_per_intervento'] = None
            row['km_ratio'] = None
    else:
        row.update({
            'azienda': None,
            'days_since_last': None, 'avg_days_between': None, 'days_ratio': None,
            'vehicle_age_days': None, 'interventions_last_90d': 0, 'recurrence_12m': 0,
            'cost_trend': None, 'km_dal_ultimo_intervento': None,
            'avg_km_per_intervento': None, 'km_ratio': None,
        })

    if not trips.empty:
        cutoff_30 = ref - pd.Timedelta(days=30)
        cutoff_90 = ref - pd.Timedelta(days=90)
        recent_90 = trips[trips['data_viaggio'] >= cutoff_90]

        row['km_ultimi_30d'] = round(float(trips[trips['data_viaggio'] >= cutoff_30]['km'].sum()), 1)
        row['km_ultimi_90d'] = round(float(recent_90['km'].sum()), 1)
        row['km_stimati_settimana'] = round(float(recent_90['km'].sum()) / 90 * 7, 1) if not recent_90.empty else None
    else:
        row['km_ultimi_30d'] = None
        row['km_ultimi_90d'] = None
        row['km_stimati_settimana'] = None

    row['month_sin'] = np.sin(2 * np.pi * ref.month / 12)
    row['month_cos'] = np.cos(2 * np.pi * ref.month / 12)

    return pd.DataFrame([row])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    df = build_features()
    print(f"\nFeature matrix: {df.shape}")
    print(f"Colonne: {df.columns.tolist()}")
    print(f"\nPrime 5 righe:")
    print(df.head())
