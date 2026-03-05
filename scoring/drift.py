"""
Monitoring del model drift.

Verifica se le performance del modello degradano nel tempo
confrontando PR-AUC su dati recenti vs baseline.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score

# Setup path per import
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .features import build_features, FEATURE_COLS
from .target import build_training_dataset, TIPI_GUASTO
from .predict import load_models

logger = logging.getLogger("maintenance-agent.scoring.drift")

# Directory per log drift
MODELS_DIR = project_root / "models"
DRIFT_LOG_PATH = MODELS_DIR / "drift_log.json"

# Soglie default
DEFAULT_DRIFT_THRESHOLD = 0.10  # 10% di degradazione
DEFAULT_LOOKBACK_DAYS = 90


def check_model_drift(
    model: Any,
    X_recent: pd.DataFrame,
    y_recent: pd.Series,
    baseline_pr_auc: float,
    threshold: float = DEFAULT_DRIFT_THRESHOLD
) -> Dict[str, Any]:
    """
    Verifica se il modello ha subito drift.

    Args:
        model: Modello addestrato
        X_recent: Feature su dati recenti
        y_recent: Target su dati recenti
        baseline_pr_auc: PR-AUC baseline (dal training)
        threshold: Soglia di degradazione (default 10%)

    Returns:
        Dict con: drift_detected, current_pr_auc, baseline_pr_auc, degradation
    """
    if len(y_recent) < 10:
        return {
            'drift_detected': False,
            'reason': 'Campioni insufficienti per valutare drift',
            'n_samples': len(y_recent),
        }

    if y_recent.sum() < 2:
        return {
            'drift_detected': False,
            'reason': 'Troppo pochi campioni positivi',
            'n_positive': int(y_recent.sum()),
        }

    try:
        y_proba = model.predict_proba(X_recent)[:, 1]
        current_pr_auc = average_precision_score(y_recent, y_proba)
    except Exception as e:
        return {
            'drift_detected': False,
            'error': str(e),
        }

    degradation = (baseline_pr_auc - current_pr_auc) / baseline_pr_auc if baseline_pr_auc > 0 else 0
    drift_detected = degradation > threshold

    result = {
        'drift_detected': drift_detected,
        'current_pr_auc': round(current_pr_auc, 4),
        'baseline_pr_auc': round(baseline_pr_auc, 4),
        'degradation': round(degradation, 4),
        'threshold': threshold,
        'n_samples': len(y_recent),
        'n_positive': int(y_recent.sum()),
    }

    if drift_detected:
        logger.warning(
            f"DRIFT DETECTED: PR-AUC {baseline_pr_auc:.3f} → {current_pr_auc:.3f} "
            f"({degradation*100:.1f}% degradazione)"
        )

    return result


def check_all_models_drift(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    threshold: float = DEFAULT_DRIFT_THRESHOLD
) -> Dict[str, Dict]:
    """
    Verifica drift per tutti i modelli.

    Args:
        lookback_days: Giorni di dati recenti da usare
        threshold: Soglia di degradazione

    Returns:
        Dict con risultati per ogni modello
    """
    logger.info(f"Checking drift per tutti i modelli (ultimi {lookback_days} giorni)...")

    models = load_models()
    if not models:
        logger.error("Nessun modello caricato")
        return {}

    # Carica dati recenti
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=lookback_days)

    try:
        df = build_training_dataset(
            min_date=start_date.strftime('%Y-%m-%d'),
            max_date=end_date.strftime('%Y-%m-%d')
        )
    except Exception as e:
        logger.error(f"Errore caricamento dati recenti: {e}")
        return {}

    if df.empty:
        logger.warning("Nessun dato recente disponibile")
        return {}

    # Merge con feature
    all_features = build_features()
    df = df.merge(all_features, on='targa', how='left', suffixes=('', '_feat'))
    df = df.dropna(subset=['days_since_last'])

    results = {}

    for model_name, model in models.items():
        # Parse model name: risk_30d_freni
        parts = model_name.split('_')
        if len(parts) < 3:
            continue

        horizon_str = parts[1]  # "30d"
        tipo_guasto = '_'.join(parts[2:])  # "freni" o "altro_tipo"

        horizon_days = int(horizon_str.replace('d', ''))
        target_col = f'fail_{horizon_days}d'

        if target_col not in df.columns:
            continue

        # Filtra per tipo_guasto
        df_tipo = df[df['tipo_guasto'] == tipo_guasto].copy()

        if len(df_tipo) < 20:
            results[model_name] = {
                'drift_detected': False,
                'reason': f'Campioni insufficienti ({len(df_tipo)})',
            }
            continue

        # Prepara X, y
        feature_cols = [c for c in FEATURE_COLS if c in df_tipo.columns]
        X_recent = df_tipo[feature_cols].fillna(0)
        y_recent = df_tipo[target_col]

        # Carica baseline PR-AUC dalle metriche salvate
        metrics_path = MODELS_DIR / f"{model_name}_metrics.json"
        baseline_pr_auc = 0.5  # Default

        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
                baseline_pr_auc = metrics.get('pr_auc', 0.5)

        # Check drift
        drift_result = check_model_drift(
            model, X_recent, y_recent, baseline_pr_auc, threshold
        )
        drift_result['tipo_guasto'] = tipo_guasto
        drift_result['horizon_days'] = horizon_days

        results[model_name] = drift_result

    # Log risultati
    _log_drift_results(results)

    # Riepilogo
    drifted = [k for k, v in results.items() if v.get('drift_detected')]
    if drifted:
        logger.warning(f"\nMODELLI CON DRIFT: {', '.join(drifted)}")
    else:
        logger.info("\nNessun drift significativo rilevato")

    return results


def _log_drift_results(results: Dict[str, Dict]) -> None:
    """
    Salva i risultati del drift check nel log.

    Args:
        results: Risultati del drift check
    """
    # Carica log esistente
    drift_log = []
    if DRIFT_LOG_PATH.exists():
        try:
            with open(DRIFT_LOG_PATH) as f:
                drift_log = json.load(f)
        except:
            drift_log = []

    # Aggiungi entry
    entry = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {
            'total_models': len(results),
            'drifted': len([k for k, v in results.items() if v.get('drift_detected')]),
        }
    }

    drift_log.append(entry)

    # Mantieni solo ultimi 100 entry
    drift_log = drift_log[-100:]

    # Salva
    with open(DRIFT_LOG_PATH, 'w') as f:
        json.dump(drift_log, f, indent=2, default=str)


def get_drift_history(last_n: int = 10) -> List[Dict]:
    """
    Recupera la storia dei drift check.

    Args:
        last_n: Numero di entry da restituire

    Returns:
        Lista di entry del log
    """
    if not DRIFT_LOG_PATH.exists():
        return []

    with open(DRIFT_LOG_PATH) as f:
        drift_log = json.load(f)

    return drift_log[-last_n:]


def send_drift_alert(drifted_models: List[str], results: Dict[str, Dict]) -> None:
    """
    Invia alert per modelli con drift.

    Placeholder per integrazione con sistema di notifiche.

    Args:
        drifted_models: Lista di modelli con drift
        results: Risultati completi del drift check
    """
    # TODO: Integrare con sistema di notifiche (email, Slack, etc.)

    alert_msg = f"""
⚠️ MODEL DRIFT ALERT

I seguenti modelli hanno subito drift significativo:

"""
    for model_name in drifted_models:
        r = results[model_name]
        alert_msg += f"""
- {model_name}:
  Baseline PR-AUC: {r.get('baseline_pr_auc', 'N/A')}
  Current PR-AUC: {r.get('current_pr_auc', 'N/A')}
  Degradation: {r.get('degradation', 0)*100:.1f}%
"""

    alert_msg += """
Azione consigliata: riaddestramento dei modelli con dati recenti.
"""

    logger.warning(alert_msg)

    # Salva alert su file
    alert_path = MODELS_DIR / "drift_alert.txt"
    with open(alert_path, 'w') as f:
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(alert_msg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check drift per tutti i modelli
    results = check_all_models_drift(lookback_days=90)

    print("\nRisultati drift check:")
    for model_name, result in results.items():
        drift_status = "⚠️ DRIFT" if result.get('drift_detected') else "✅ OK"
        print(f"  {model_name}: {drift_status}")
        if 'current_pr_auc' in result:
            print(f"    PR-AUC: {result['baseline_pr_auc']:.3f} → {result['current_pr_auc']:.3f}")
