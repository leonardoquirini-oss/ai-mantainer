"""
Modulo ML per scoring del rischio di guasto.

Produce un risk score 0-100 per coppia (targa, tipo_guasto)
da passare all'agente LLM.

Moduli:
- features: build_features() per training batch
- target: load_classified_interventions(), build_target()
- train: walk-forward CV, XGBoost/LightGBM, Optuna
- predict: compute_risk_score(), score_fleet()
- shap_explain: SHAP factors per spiegabilità
- drift: monitoring model drift
"""

from .features import build_features, build_features_for_targa, FEATURE_COLS
from .target import load_classified_interventions, build_target, build_training_dataset, TIPI_GUASTO
from .predict import compute_risk_score, risk_level, score_fleet, load_models, compute_all_risk_scores
from .train import train_all_models, train_and_save_model
from .shap_explain import get_shap_factors, explain_prediction
from .drift import check_model_drift, check_all_models_drift

__all__ = [
    # Features
    "build_features",
    "build_features_for_targa",
    "FEATURE_COLS",
    # Target
    "load_classified_interventions",
    "build_target",
    "build_training_dataset",
    "TIPI_GUASTO",
    # Predict
    "compute_risk_score",
    "compute_all_risk_scores",
    "risk_level",
    "score_fleet",
    "load_models",
    # Train
    "train_all_models",
    "train_and_save_model",
    # SHAP
    "get_shap_factors",
    "explain_prediction",
    # Drift
    "check_model_drift",
    "check_all_models_drift",
]
