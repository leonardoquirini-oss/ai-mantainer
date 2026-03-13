"""
Training dei modelli ML per scoring del rischio.

- Time-based split (mai random split!)
- Walk-forward cross-validation
- XGBoost/LightGBM con scale_pos_weight
- Ottimizzazione iperparametri con Optuna
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
)

# Setup path per import
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .features import build_features, FEATURE_COLS
from .target import build_training_dataset, TIPI_GUASTO, get_target_stats

logger = logging.getLogger("maintenance-agent.scoring.train")

# Directory per salvare i modelli
MODELS_DIR = project_root / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Configurazione split temporale
HISTORY_START = pd.Timestamp('2018-01-01')
TRAIN_CUTOFF = pd.Timestamp('2025-06-30')
GAP_DAYS = 30
TEST_START = TRAIN_CUTOFF + pd.Timedelta(days=GAP_DAYS)

# Walk-forward folds
WALK_FORWARD_FOLDS = [
    ('2018-01-01', '2021-12-31', '2022-02-01', '2022-07-31'),
    ('2018-01-01', '2022-12-31', '2023-02-01', '2023-07-31'),
    ('2018-01-01', '2023-12-31', '2024-02-01', '2024-07-31'),
    ('2018-01-01', '2024-06-30', '2024-08-01', '2024-12-31'),
    ('2018-01-01', '2024-12-31', '2025-02-01', '2025-06-30'),
]


def merge_features_with_target(
    target_df: pd.DataFrame,
    reference_dates: Optional[List[pd.Timestamp]] = None
) -> pd.DataFrame:
    """
    Unisce il dataset target con le feature calcolate per coorti mensili.

    Per ogni coorte mensile, calcola le feature con reference_date = fine mese,
    evitando data leakage (le feature non vedono eventi futuri).

    Args:
        target_df: DataFrame con target (da build_training_dataset)
        reference_dates: Date uniche per cui calcolare feature (ottimizzazione)

    Returns:
        DataFrame con target + feature
    """
    logger.info("Merging features with target (per-cohort, no data leakage)...")

    # Assicura che data_intervento sia datetime
    target_df = target_df.copy()
    target_df['data_intervento'] = pd.to_datetime(target_df['data_intervento'])

    # Raggruppa per mese di data_intervento
    target_df['cohort'] = target_df['data_intervento'].dt.to_period('M')
    cohorts = target_df['cohort'].unique()

    logger.info(f"Calcolo feature per {len(cohorts)} coorti mensili...")

    merged_parts = []
    for cohort in sorted(cohorts):
        # Fine mese come reference_date
        ref_date = cohort.to_timestamp(how='end')

        # Feature calcolate solo con dati <= ref_date
        cohort_features = build_features(reference_date=ref_date)

        # Righe target di questa coorte
        cohort_target = target_df[target_df['cohort'] == cohort]

        # Merge per targa
        merged = cohort_target.merge(
            cohort_features,
            on='targa',
            how='left',
            suffixes=('', '_feat')
        )
        merged_parts.append(merged)

    result = pd.concat(merged_parts, ignore_index=True)

    # Rimuovi colonna ausiliaria
    result = result.drop(columns=['cohort'])

    # Rimuovi righe senza feature
    before = len(result)
    result = result.dropna(subset=['days_since_last'])
    after = len(result)

    if before > after:
        logger.warning(f"Rimosse {before - after} righe senza feature")

    logger.info(f"Dataset merged: {len(result)} righe")

    return result


def prepare_training_data(
    tipo_guasto: str,
    horizon_days: int = 30,
    min_samples: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara dati per training di un singolo modello.

    Args:
        tipo_guasto: Tipo di guasto da predire
        horizon_days: Orizzonte temporale (7, 30, 90)
        min_samples: Minimo numero di campioni positivi richiesti

    Returns:
        (X_train, X_test, y_train, y_test)

    Raises:
        ValueError: Se non ci sono abbastanza campioni
    """
    target_col = f'fail_{horizon_days}d'

    # Carica e prepara dataset
    df = build_training_dataset(
        min_date=HISTORY_START.strftime('%Y-%m-%d'),
        max_date=None  # Usa tutti i dati disponibili
    )

    # Filtra per tipo_guasto
    df = df[df['tipo_guasto'] == tipo_guasto].copy()

    if len(df) < min_samples:
        raise ValueError(f"Solo {len(df)} campioni per {tipo_guasto}, servono almeno {min_samples}")

    # Merge con feature
    df = merge_features_with_target(df)

    # Time-based split
    train_mask = df['data_intervento'] < TRAIN_CUTOFF
    test_mask = df['data_intervento'] >= TEST_START

    train_df = df[train_mask]
    test_df = df[test_mask]

    logger.info(f"Split {tipo_guasto}: train={len(train_df)}, test={len(test_df)}")

    # Verifica campioni positivi
    pos_train = train_df[target_col].sum()
    pos_test = test_df[target_col].sum()

    if pos_train < 10:
        raise ValueError(f"Solo {pos_train} campioni positivi nel train per {tipo_guasto}")

    logger.info(f"Campioni positivi: train={pos_train}, test={pos_test}")

    # Prepara X, y
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    use_lightgbm: bool = False,
    params: Optional[Dict] = None
) -> Any:
    """
    Addestra un singolo modello.

    Args:
        X_train, y_train: Dati di training
        X_val, y_val: Dati di validation (opzionale, per early stopping)
        use_lightgbm: Se True usa LightGBM, altrimenti XGBoost
        params: Iperparametri custom

    Returns:
        Modello addestrato
    """
    # Calcola scale_pos_weight per class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    if use_lightgbm:
        from lightgbm import LGBMClassifier

        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'min_child_samples': 20,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'verbosity': -1,
        }
        if params:
            default_params.update(params)

        model = LGBMClassifier(**default_params)
        logger.info(f"Training LightGBM (n_estimators={default_params['n_estimators']})...")

        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[],  # LightGBM handles early stopping differently
            )
        else:
            model.fit(X_train, y_train)

    else:
        from xgboost import XGBClassifier

        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'aucpr',
            'random_state': 42,
            'verbosity': 0,
        }
        if params:
            default_params.update(params)

        model = XGBClassifier(**default_params)

        logger.info(f"Training XGBoost (n_estimators={default_params['n_estimators']})...")

        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100,  # Log ogni 100 alberi
            )
        else:
            model.fit(X_train, y_train)

    return model


def walk_forward_cv(
    df: pd.DataFrame,
    tipo_guasto: str,
    horizon_days: int = 30
) -> List[float]:
    """
    Walk-forward cross-validation.

    Misura la stabilità del modello nel tempo.

    Args:
        df: Dataset completo con target e feature
        tipo_guasto: Tipo di guasto
        horizon_days: Orizzonte temporale

    Returns:
        Lista di PR-AUC scores per ogni fold
    """
    target_col = f'fail_{horizon_days}d'
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    # Filtra per tipo_guasto
    df = df[df['tipo_guasto'] == tipo_guasto].copy()

    pr_auc_scores = []

    for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(WALK_FORWARD_FOLDS):
        mask_train = (df['data_intervento'] >= train_start) & (df['data_intervento'] <= train_end)
        mask_val = (df['data_intervento'] >= val_start) & (df['data_intervento'] <= val_end)

        train_df = df[mask_train]
        val_df = df[mask_val]

        if len(train_df) < 50 or len(val_df) < 10:
            logger.warning(f"Fold {fold_idx + 1}: dati insufficienti, skip")
            continue

        X_tr, y_tr = train_df[feature_cols], train_df[target_col]
        X_vl, y_vl = val_df[feature_cols], val_df[target_col]

        try:
            model = train_model(X_tr, y_tr)
            proba = model.predict_proba(X_vl)[:, 1]
            score = average_precision_score(y_vl, proba)
            pr_auc_scores.append(score)
            logger.info(f"Fold {fold_idx + 1} ({val_start[:4]}): PR-AUC = {score:.3f}")
        except Exception as e:
            logger.warning(f"Fold {fold_idx + 1} fallito: {e}")

    if pr_auc_scores:
        mean_score = np.mean(pr_auc_scores)
        std_score = np.std(pr_auc_scores)
        logger.info(f"\nMedia PR-AUC: {mean_score:.3f} ± {std_score:.3f}")

        if std_score > 0.05:
            logger.warning("Alta variabilità tra fold — il modello potrebbe essere instabile")

    return pr_auc_scores


def optimize_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50
) -> Dict:
    """
    Ottimizza iperparametri con Optuna.

    Args:
        X_train, y_train: Dati training
        X_val, y_val: Dati validation
        n_trials: Numero di trial Optuna

    Returns:
        Best params
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna non installato, uso parametri default")
        return {}

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': scale_pos_weight,
        }

        model = train_model(X_train, y_train, params=params)
        proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, proba)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best PR-AUC: {study.best_value:.3f}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params


def train_and_save_model(
    tipo_guasto: str,
    horizon_days: int = 30,
    optimize: bool = False
) -> Dict[str, Any]:
    """
    Addestra e salva un modello per un tipo_guasto specifico.

    Args:
        tipo_guasto: Tipo di guasto
        horizon_days: Orizzonte temporale (7, 30, 90)
        optimize: Se True, ottimizza iperparametri con Optuna

    Returns:
        Dict con metriche del modello
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {tipo_guasto} @ {horizon_days}d")
    logger.info(f"{'='*60}")

    try:
        X_train, X_test, y_train, y_test = prepare_training_data(
            tipo_guasto, horizon_days
        )
    except ValueError as e:
        logger.warning(f"Skip {tipo_guasto}: {e}")
        return {'status': 'skipped', 'reason': str(e)}

    # Split train in train/val per early stopping
    val_size = int(len(X_train) * 0.15)
    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

    # Ottimizzazione (opzionale)
    params = {}
    if optimize:
        params = optimize_hyperparams(X_tr, y_tr, X_val, y_val)

    # Training finale
    model = train_model(X_train, y_train, X_val, y_val, params=params)

    # Valutazione
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"\nRisultati su test set:")
    logger.info(f"  PR-AUC: {pr_auc:.3f}")
    logger.info(f"  F1: {f1:.3f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['ok', 'fail'])}")

    # Salva modello
    model_path = MODELS_DIR / f"risk_{horizon_days}d_{tipo_guasto}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Modello salvato: {model_path}")

    # Salva metriche
    # Feature importance
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        for col, imp in zip(X_train.columns, model.feature_importances_):
            feature_importance[col] = round(float(imp), 4)
        # Ordina per importanza decrescente
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

    metrics = {
        'tipo_guasto': tipo_guasto,
        'horizon_days': horizon_days,
        'pr_auc': round(pr_auc, 4),
        'f1': round(f1, 4),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'positive_rate_train': round(y_train.mean(), 4),
        'positive_rate_test': round(y_test.mean(), 4),
        'params': params,
        'feature_cols': list(X_train.columns),
        'feature_importance': feature_importance,
    }

    metrics_path = MODELS_DIR / f"risk_{horizon_days}d_{tipo_guasto}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_all_models(
    horizons: List[int] = [7, 30, 90],
    optimize: bool = False
) -> Dict[str, Dict]:
    """
    Addestra tutti i modelli per tutti i tipi_guasto e orizzonti.

    Args:
        horizons: Lista di orizzonti temporali
        optimize: Se True, ottimizza iperparametri

    Returns:
        Dict con risultati per ogni modello
    """
    total_models = len(TIPI_GUASTO) * len(horizons)
    logger.info(f"\n{'#'*60}")
    logger.info(f"TRAINING COMPLETO: {len(TIPI_GUASTO)} tipi × {len(horizons)} orizzonti = {total_models} modelli")
    logger.info(f"{'#'*60}\n")

    results = {}
    model_idx = 0

    for horizon in horizons:
        for tipo in TIPI_GUASTO:
            model_idx += 1
            key = f"{tipo}_{horizon}d"
            logger.info(f"\n[{model_idx}/{total_models}] Training {tipo} @ {horizon}d...")
            try:
                metrics = train_and_save_model(tipo, horizon, optimize)
                results[key] = metrics
            except Exception as e:
                logger.error(f"Errore training {key}: {e}")
                results[key] = {'status': 'error', 'error': str(e)}

    # Riepilogo
    logger.info(f"\n{'='*60}")
    logger.info("RIEPILOGO TRAINING")
    logger.info(f"{'='*60}")

    for key, metrics in results.items():
        if metrics.get('status') in ('skipped', 'error'):
            logger.info(f"  {key}: {metrics.get('status')} - {metrics.get('reason', metrics.get('error', ''))}")
        else:
            logger.info(f"  {key}: PR-AUC={metrics.get('pr_auc', 0):.3f}, F1={metrics.get('f1', 0):.3f}")

    # Salva riepilogo
    summary_path = MODELS_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nRiepilogo salvato: {summary_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Training completo
    results = train_all_models(horizons=[30], optimize=False)
