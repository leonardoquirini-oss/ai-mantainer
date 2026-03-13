"""
Predizione del risk score per singola targa o intera flotta.

Il risk score combina le probabilità dei tre orizzonti (7d, 30d, 90d)
con pesi decrescenti e scala a 0-100.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# Setup path per import
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from db.init import get_connection
from .features import build_features, build_features_for_targa, FEATURE_COLS
from .target import TIPI_GUASTO

logger = logging.getLogger("maintenance-agent.scoring.predict")

# Directory modelli
MODELS_DIR = project_root / "models"

# Pesi per combinazione probabilità
HORIZON_WEIGHTS = {
    7: 0.50,   # Urgente: peso maggiore
    30: 0.35,  # Pianificabile
    90: 0.15,  # Strategico
}

# Soglie per risk level
RISK_THRESHOLDS = {
    'rosso': 75,    # >= 75
    'arancio': 50,  # >= 50
    'giallo': 25,   # >= 25
    'verde': 0,     # < 25
}


def normalize_targa(targa: str) -> str:
    """
    Normalizza targa per matching con lista_mezzi_attivi (TARGATRIM).

    Il template lista_mezzi_attivi restituisce:
    - Veicoli/Semirimorchi: targa senza spazi (AA48020)
    - Container: solo parte numerica (1234 invece di GBTU 1234.5)

    Args:
        targa: Targa originale dal DB manutenzioni

    Returns:
        Targa normalizzata per matching
    """
    targa = targa.strip().upper()

    if targa.startswith('GBTU'):
        # Estrai parte numerica: "GBTU 1234.5" -> "1234", "GBTU 1234-3" -> "1234"
        match = re.search(r'GBTU\s*(\d+)', targa)
        if match:
            return match.group(1)

    # Per altri veicoli: rimuovi spazi
    return targa.replace(' ', '')


def load_models() -> Dict[str, Any]:
    """
    Carica tutti i modelli salvati dalla directory models/.

    Returns:
        Dict con chiavi 'risk_{horizon}d_{tipo_guasto}' e valori i modelli
    """
    models = {}

    if not MODELS_DIR.exists():
        logger.warning(f"Directory modelli non trovata: {MODELS_DIR}")
        return models

    for model_file in MODELS_DIR.glob("risk_*d_*.joblib"):
        # Estrai nome modello dal filename
        name = model_file.stem  # risk_30d_freni
        try:
            models[name] = joblib.load(model_file)
            logger.debug(f"Caricato modello: {name}")
        except Exception as e:
            logger.error(f"Errore caricamento {model_file}: {e}")

    logger.info(f"Caricati {len(models)} modelli")
    return models


def risk_level(score: float) -> str:
    """
    Converte risk score numerico in livello categorico.

    Args:
        score: Risk score 0-100

    Returns:
        Livello: 'verde', 'giallo', 'arancio', 'rosso'
    """
    if score >= RISK_THRESHOLDS['rosso']:
        return 'rosso'
    elif score >= RISK_THRESHOLDS['arancio']:
        return 'arancio'
    elif score >= RISK_THRESHOLDS['giallo']:
        return 'giallo'
    else:
        return 'verde'


def compute_risk_score(
    targa: str,
    tipo_guasto: str,
    models: Dict[str, Any],
    X_current: Optional[pd.DataFrame] = None,
    include_shap: bool = True
) -> Dict[str, Any]:
    """
    Calcola il risk score per una coppia (targa, tipo_guasto).

    Combina le probabilità dei tre orizzonti con pesi:
    - 7d (urgente): 50%
    - 30d (pianificabile): 35%
    - 90d (strategico): 15%

    Args:
        targa: Targa del veicolo
        tipo_guasto: Tipo di guasto da predire
        models: Dict dei modelli caricati
        X_current: Feature pre-calcolate (opzionale)
        include_shap: Se True, include top factors SHAP

    Returns:
        Dict con: risk_score, risk_level, prob_7d, prob_30d, prob_90d, top_factors
    """
    targa_clean = targa.strip().upper()

    # Carica feature se non fornite
    if X_current is None:
        X_current = build_features_for_targa(targa_clean)

    if X_current.empty:
        logger.warning(f"Nessuna feature trovata per targa {targa_clean}")
        return {
            'targa': targa_clean,
            'tipo_guasto': tipo_guasto,
            'risk_score': None,
            'risk_level': 'unknown',
            'error': 'Nessuna feature disponibile',
        }

    # Prepara feature nel formato corretto
    feature_cols = [c for c in FEATURE_COLS if c in X_current.columns]
    X = X_current[feature_cols].copy()

    # NaN lasciati intatti: XGBoost gestisce NaN nativamente
    # imparando la direzione di split ottimale per i dati mancanti

    # Calcola probabilità per ogni orizzonte
    probs = {}
    missing_models = []

    for horizon in [7, 30, 90]:
        model_name = f"risk_{horizon}d_{tipo_guasto}"

        if model_name not in models:
            missing_models.append(model_name)
            probs[horizon] = None
            continue

        model = models[model_name]

        try:
            # Verifica che il modello abbia le feature attese
            model_features = getattr(model, 'feature_names_in_', None)
            if model_features is not None:
                # Allinea le feature
                for col in model_features:
                    if col not in X.columns:
                        X[col] = 0
                X_aligned = X[model_features]
            else:
                X_aligned = X

            prob = model.predict_proba(X_aligned)[0, 1]
            probs[horizon] = float(prob)
        except Exception as e:
            logger.error(f"Errore predizione {model_name}: {e}")
            probs[horizon] = None

    # Verifica che abbiamo almeno una probabilità
    valid_probs = {h: p for h, p in probs.items() if p is not None}

    if not valid_probs:
        return {
            'targa': targa_clean,
            'tipo_guasto': tipo_guasto,
            'risk_score': None,
            'risk_level': 'unknown',
            'error': f'Nessun modello disponibile per {tipo_guasto}',
            'missing_models': missing_models,
        }

    # Calcola score combinato
    weighted_sum = 0
    weight_sum = 0

    for horizon, prob in valid_probs.items():
        weight = HORIZON_WEIGHTS[horizon]
        weighted_sum += prob * weight
        weight_sum += weight

    # Normalizza e scala a 0-100
    combined_prob = weighted_sum / weight_sum if weight_sum > 0 else 0
    risk_score = round(combined_prob * 100, 1)

    result = {
        'targa': targa_clean,
        'tipo_guasto': tipo_guasto,
        'risk_score': risk_score,
        'risk_level': risk_level(risk_score),
        'prob_7d': probs.get(7),
        'prob_30d': probs.get(30),
        'prob_90d': probs.get(90),
        'computed_at': datetime.now().isoformat(),
    }

    # Aggiungi SHAP factors se richiesto
    if include_shap and valid_probs:
        try:
            from .shap_explain import get_shap_factors

            # Usa il modello a 30d come riferimento per SHAP
            if 30 in valid_probs and f"risk_30d_{tipo_guasto}" in models:
                model = models[f"risk_30d_{tipo_guasto}"]
                top_factors = get_shap_factors(model, X_aligned, top_n=3)
                result['top_factors'] = top_factors
        except ImportError:
            pass  # SHAP module non ancora implementato
        except Exception as e:
            logger.debug(f"Errore SHAP: {e}")

    return result


def compute_all_risk_scores(
    targa: str,
    models: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Calcola il risk score per tutti i tipi_guasto di una targa.

    Args:
        targa: Targa del veicolo
        models: Dict dei modelli (caricato se non fornito)

    Returns:
        Lista di dict con risk score per ogni tipo_guasto
    """
    if models is None:
        models = load_models()

    # Carica feature una volta sola
    X_current = build_features_for_targa(targa)

    results = []
    for tipo_guasto in TIPI_GUASTO:
        result = compute_risk_score(targa, tipo_guasto, models, X_current)
        results.append(result)

    # Ordina per risk_score decrescente
    results.sort(key=lambda x: x.get('risk_score') or 0, reverse=True)

    return results


def score_fleet(
    save_to_db: bool = True,
    tipi_guasto: Optional[List[str]] = None,
    filter_active: bool = True
) -> int:
    """
    Calcola risk score per tutte le targhe della flotta.

    Salva i risultati nella tabella risk_scores in SQLite.

    Args:
        save_to_db: Se True, salva su DB
        tipi_guasto: Lista di tipi_guasto da calcolare (default: tutti)
        filter_active: Se True, valuta solo mezzi attivi (default: True)

    Returns:
        Numero di score calcolati
    """
    logger.info("Inizio scoring flotta...")

    models = load_models()

    if not models:
        logger.error("Nessun modello caricato")
        return 0

    # Carica feature per tutte le targhe
    all_features = build_features()

    if all_features.empty:
        logger.error("Nessuna feature disponibile")
        return 0

    total_targhe = len(all_features['targa'].unique())

    # Filtra solo mezzi attivi se richiesto
    if filter_active:
        try:
            from agent.connectors.adhoc_connector import AdHocConnector

            connector = AdHocConnector()
            mezzi_attivi = connector.get_mezzi_attivi()  # set di TARGATRIM
            connector.close()

            # Normalizza targhe del DB per matching con TARGATRIM
            # Container GBTU: estrai parte numerica
            # Altri: rimuovi spazi
            targhe_db = all_features['targa'].unique()
            targhe_attive = set()

            for targa in targhe_db:
                normalized = normalize_targa(targa)
                if normalized in mezzi_attivi:
                    targhe_attive.add(targa)

            all_features = all_features[all_features['targa'].isin(targhe_attive)]
            logger.info(f"Mezzi attivi da AdHoc: {len(mezzi_attivi)}")
            logger.info(f"Targhe matchate: {len(targhe_attive)} su {total_targhe} nel DB")
        except Exception as e:
            logger.warning(f"Impossibile recuperare mezzi attivi: {e}")
            logger.warning("Continuo con tutte le targhe...")

    targhe = all_features['targa'].unique()
    logger.info(f"Scoring {len(targhe)} targhe...")

    tipi = tipi_guasto or TIPI_GUASTO
    results = []

    for idx, targa in enumerate(targhe):
        X_current = all_features[all_features['targa'] == targa]

        for tipo in tipi:
            try:
                score = compute_risk_score(
                    targa, tipo, models, X_current, include_shap=False
                )
                if score.get('risk_score') is not None:
                    results.append(score)
            except Exception as e:
                logger.debug(f"Errore {targa}/{tipo}: {e}")

        if (idx + 1) % 100 == 0:
            logger.info(f"Processate {idx + 1}/{len(targhe)} targhe...")

    logger.info(f"Calcolati {len(results)} score")

    # Salva su DB
    if save_to_db and results:
        _save_scores_to_db(results)

    return len(results)


def _save_scores_to_db(results: List[Dict]) -> None:
    """
    Salva i risk score nella tabella risk_scores.

    Args:
        results: Lista di dict con risk score
    """
    conn = get_connection()

    # Crea tabella se non esiste
    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            targa TEXT NOT NULL,
            tipo_guasto TEXT NOT NULL,
            risk_score REAL,
            risk_level TEXT,
            prob_7d REAL,
            prob_30d REAL,
            prob_90d REAL,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(targa, tipo_guasto)
        )
    """)

    # Crea indici
    conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_scores_targa ON risk_scores(targa)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_scores_level ON risk_scores(risk_level)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_scores_score ON risk_scores(risk_score DESC)")

    # Insert/Update
    for r in results:
        conn.execute("""
            INSERT INTO risk_scores (targa, tipo_guasto, risk_score, risk_level, prob_7d, prob_30d, prob_90d, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(targa, tipo_guasto) DO UPDATE SET
                risk_score = excluded.risk_score,
                risk_level = excluded.risk_level,
                prob_7d = excluded.prob_7d,
                prob_30d = excluded.prob_30d,
                prob_90d = excluded.prob_90d,
                computed_at = excluded.computed_at
        """, (
            r['targa'],
            r['tipo_guasto'],
            r.get('risk_score'),
            r.get('risk_level'),
            r.get('prob_7d'),
            r.get('prob_30d'),
            r.get('prob_90d'),
            r.get('computed_at', datetime.now().isoformat()),
        ))

    conn.commit()
    conn.close()

    logger.info(f"Salvati {len(results)} score su DB")


def get_high_risk_vehicles(
    min_score: float = 50,
    limit: int = 20
) -> List[Dict]:
    """
    Recupera i veicoli ad alto rischio dal DB.

    Args:
        min_score: Score minimo per essere considerato alto rischio
        limit: Numero massimo di risultati

    Returns:
        Lista di dict con targa, tipo_guasto, risk_score, risk_level
    """
    conn = get_connection()

    rows = conn.execute("""
        SELECT targa, tipo_guasto, risk_score, risk_level, prob_7d, prob_30d, prob_90d, computed_at
        FROM risk_scores
        WHERE risk_score >= ?
        ORDER BY risk_score DESC
        LIMIT ?
    """, (min_score, limit)).fetchall()

    conn.close()

    return [
        {
            'targa': r[0],
            'tipo_guasto': r[1],
            'risk_score': r[2],
            'risk_level': r[3],
            'prob_7d': r[4],
            'prob_30d': r[5],
            'prob_90d': r[6],
            'computed_at': r[7],
        }
        for r in rows
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    models = load_models()
    print(f"Modelli caricati: {len(models)}")

    if models:
        # Test su una targa
        targa = "FE065SW"
        scores = compute_all_risk_scores(targa, models)

        print(f"\nRisk scores per {targa}:")
        for s in scores:
            if s.get('risk_score') is not None:
                print(f"  {s['tipo_guasto']}: {s['risk_score']:.1f} ({s['risk_level']})")
