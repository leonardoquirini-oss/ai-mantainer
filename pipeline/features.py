"""
Pipeline per il calcolo delle feature dei veicoli.

Eseguita ogni notte per aggiornare la tabella vehicle_features.

Uso:
    from pipeline.features import update_vehicle_features

    # Aggiorna tutte le feature per tutti i veicoli
    update_vehicle_features()

    # Calcola feature per un singolo veicolo
    features = compute_all_features("AA 93252")
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd

# Import database
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from db.init import get_connection

logger = logging.getLogger("maintenance-agent.pipeline.features")


def compute_km_features(targa: str, reference_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Calcola le feature di percorrenza per una targa (motrice o semirimorchio).

    Usa la vista vehicle_km che unifica i due ruoli.

    Args:
        targa: Targa del veicolo
        reference_date: Data di riferimento (default: oggi)

    Returns:
        Dict con feature km
    """
    conn = get_connection()
    targa_clean = targa.strip().upper()

    if reference_date is None:
        reference_date = date.today()

    ref = pd.Timestamp(reference_date)

    # Tutti i viaggi di questa targa (qualunque ruolo)
    trips = pd.read_sql("""
        SELECT data_viaggio, km, ruolo
        FROM vehicle_km
        WHERE UPPER(TRIM(targa)) = ?
        ORDER BY data_viaggio
    """, conn, params=(targa_clean,))

    # Km dall'ultimo intervento di manutenzione
    last_intervention = conn.execute("""
        SELECT MAX(data_intervento)
        FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
    """, (targa_clean,)).fetchone()[0]

    conn.close()

    if trips.empty:
        return {
            'km_stimati_settimana': None,
            'km_dal_ultimo_intervento': None,
            'km_totali_storici': None,
            'km_ultimi_30d': None,
            'km_ultimi_90d': None,
        }

    trips['data_viaggio'] = pd.to_datetime(trips['data_viaggio'])

    cutoff_90 = ref - pd.Timedelta(days=90)
    cutoff_30 = ref - pd.Timedelta(days=30)

    recent_90 = trips[trips['data_viaggio'] >= cutoff_90]
    recent_30 = trips[trips['data_viaggio'] >= cutoff_30]

    # Km medi settimanali (basati sugli ultimi 90 giorni)
    km_week = None
    if not recent_90.empty:
        km_week = recent_90['km'].sum() / 90 * 7

    # Km dall'ultimo intervento
    km_dal_intervento = None
    if last_intervention:
        km_dal_intervento = trips[
            trips['data_viaggio'] >= pd.Timestamp(last_intervention)
        ]['km'].sum()

    return {
        'km_stimati_settimana': round(km_week, 1) if km_week else None,
        'km_dal_ultimo_intervento': round(float(km_dal_intervento), 1) if km_dal_intervento else None,
        'km_totali_storici': round(float(trips['km'].sum()), 1),
        'km_ultimi_30d': round(float(recent_30['km'].sum()), 1),
        'km_ultimi_90d': round(float(recent_90['km'].sum()), 1),
    }


def compute_maintenance_features(targa: str, reference_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Calcola le feature basate sullo storico manutenzioni.

    Args:
        targa: Targa del veicolo
        reference_date: Data di riferimento (default: oggi)

    Returns:
        Dict con feature manutenzione
    """
    conn = get_connection()
    targa_clean = targa.strip().upper()

    if reference_date is None:
        reference_date = date.today()

    # Info base
    base_info = conn.execute("""
        SELECT
            azienda,
            MAX(data_imm) as data_imm,
            MAX(data_intervento) as ultimo_intervento,
            MIN(data_intervento) as primo_intervento,
            COUNT(*) as n_interventi,
            SUM(costo) as costo_totale
        FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
        GROUP BY azienda
    """, (targa_clean,)).fetchone()

    if not base_info:
        conn.close()
        return {
            'azienda': None,
            'data_imm': None,
            'vehicle_age_days': None,
            'days_since_last_intervention': None,
            'avg_days_between_interventions': None,
            'days_ratio': None,
            'interventions_last_90d': 0,
            'interventions_last_365d': 0,
            'cost_last_intervention': None,
            'cost_avg_12m': None,
            'cost_trend': None,
        }

    # Calcola età veicolo
    vehicle_age_days = None
    if base_info['data_imm']:
        data_imm = datetime.strptime(base_info['data_imm'], '%Y-%m-%d').date()
        vehicle_age_days = (reference_date - data_imm).days

    # Giorni dall'ultimo intervento
    days_since_last = None
    if base_info['ultimo_intervento']:
        ultimo = datetime.strptime(base_info['ultimo_intervento'], '%Y-%m-%d').date()
        days_since_last = (reference_date - ultimo).days

    # Media giorni tra interventi
    interventions = pd.read_sql("""
        SELECT data_intervento
        FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
        ORDER BY data_intervento
    """, conn, params=(targa_clean,))

    avg_days_between = None
    if len(interventions) > 1:
        interventions['data_intervento'] = pd.to_datetime(interventions['data_intervento'])
        diffs = interventions['data_intervento'].diff().dropna()
        avg_days_between = diffs.dt.days.mean()

    # Days ratio
    days_ratio = None
    if days_since_last and avg_days_between and avg_days_between > 0:
        days_ratio = days_since_last / avg_days_between

    # Interventi ultimi 90/365 giorni
    ref_str = reference_date.isoformat()
    int_90d = conn.execute("""
        SELECT COUNT(*) FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
          AND data_intervento >= DATE(?, '-90 days')
    """, (targa_clean, ref_str)).fetchone()[0]

    int_365d = conn.execute("""
        SELECT COUNT(*) FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
          AND data_intervento >= DATE(?, '-365 days')
    """, (targa_clean, ref_str)).fetchone()[0]

    # Costo ultimo intervento
    cost_last = conn.execute("""
        SELECT costo FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
        ORDER BY data_intervento DESC
        LIMIT 1
    """, (targa_clean,)).fetchone()
    cost_last_intervention = cost_last[0] if cost_last else None

    # Costo medio ultimi 12 mesi
    cost_12m = conn.execute("""
        SELECT AVG(costo) FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
          AND data_intervento >= DATE(?, '-365 days')
    """, (targa_clean, ref_str)).fetchone()[0]

    # Costo medio ultimi 3 mesi
    cost_3m = conn.execute("""
        SELECT AVG(costo) FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
          AND data_intervento >= DATE(?, '-90 days')
    """, (targa_clean, ref_str)).fetchone()[0]

    # Costo medio storico
    cost_storico = conn.execute("""
        SELECT AVG(costo) FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
    """, (targa_clean,)).fetchone()[0]

    # Cost trend (costo recente / costo storico)
    cost_trend = None
    if cost_3m and cost_storico and cost_storico > 0:
        cost_trend = cost_3m / cost_storico

    conn.close()

    return {
        'azienda': base_info['azienda'],
        'data_imm': base_info['data_imm'],
        'vehicle_age_days': vehicle_age_days,
        'days_since_last_intervention': days_since_last,
        'avg_days_between_interventions': round(avg_days_between, 1) if avg_days_between else None,
        'days_ratio': round(days_ratio, 2) if days_ratio else None,
        'interventions_last_90d': int_90d,
        'interventions_last_365d': int_365d,
        'cost_last_intervention': cost_last_intervention,
        'cost_avg_12m': round(cost_12m, 2) if cost_12m else None,
        'cost_trend': round(cost_trend, 2) if cost_trend else None,
    }


def compute_all_features(targa: str, reference_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Calcola tutte le feature per un veicolo.

    Combina feature km e feature manutenzione.

    Args:
        targa: Targa del veicolo
        reference_date: Data di riferimento (default: oggi)

    Returns:
        Dict con tutte le feature
    """
    if reference_date is None:
        reference_date = date.today()

    targa_clean = targa.strip().upper()

    # Feature manutenzione
    maint_features = compute_maintenance_features(targa_clean, reference_date)

    # Feature km
    km_features = compute_km_features(targa_clean, reference_date)

    return {
        'targa': targa_clean,
        'computed_at': datetime.now().isoformat(),
        **maint_features,
        **km_features,
    }


def update_vehicle_features(targhe: Optional[List[str]] = None) -> int:
    """
    Aggiorna la tabella vehicle_features per tutti i veicoli (o una lista specifica).

    Args:
        targhe: Lista di targhe da aggiornare. Se None, aggiorna tutti.

    Returns:
        Numero di veicoli aggiornati
    """
    conn = get_connection()

    # Se non specificate, prendi tutte le targhe dallo storico manutenzioni
    if targhe is None:
        rows = conn.execute("""
            SELECT DISTINCT targa FROM maintenance_history
        """).fetchall()
        targhe = [r[0] for r in rows]

    logger.info(f"Aggiornamento feature per {len(targhe)} veicoli")

    updated = 0
    for targa in targhe:
        try:
            features = compute_all_features(targa)

            if not features.get('azienda'):
                continue

            # Upsert (INSERT OR REPLACE)
            conn.execute("""
                INSERT OR REPLACE INTO vehicle_features (
                    targa, azienda, computed_at,
                    data_imm, vehicle_age_days,
                    days_since_last_intervention, avg_days_between_interventions, days_ratio,
                    interventions_last_90d, interventions_last_365d,
                    cost_last_intervention, cost_avg_12m, cost_trend,
                    km_stimati_settimana, km_dal_ultimo_intervento
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                features['targa'],
                features['azienda'],
                features['computed_at'],
                features['data_imm'],
                features['vehicle_age_days'],
                features['days_since_last_intervention'],
                features['avg_days_between_interventions'],
                features['days_ratio'],
                features['interventions_last_90d'],
                features['interventions_last_365d'],
                features['cost_last_intervention'],
                features['cost_avg_12m'],
                features['cost_trend'],
                features.get('km_stimati_settimana'),
                features.get('km_dal_ultimo_intervento'),
            ))
            updated += 1

        except Exception as e:
            logger.error(f"Errore aggiornamento feature per {targa}: {e}")

    conn.commit()
    conn.close()

    logger.info(f"Aggiornate feature per {updated} veicoli")
    return updated


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        # Aggiorna feature per una targa specifica
        targa = sys.argv[1]
        features = compute_all_features(targa)
        print(f"Feature per {targa}:")
        for k, v in features.items():
            print(f"  {k}: {v}")
    else:
        # Aggiorna tutte le feature
        n = update_vehicle_features()
        print(f"Aggiornate feature per {n} veicoli")
