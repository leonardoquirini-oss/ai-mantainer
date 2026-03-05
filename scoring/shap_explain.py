"""
Spiegabilità del modello con SHAP.

Fornisce i top factors che contribuiscono al risk score.
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("maintenance-agent.scoring.shap_explain")

# Cache per gli explainer
_explainer_cache: Dict[str, Any] = {}


def get_explainer(model: Any) -> Any:
    """
    Crea o recupera dalla cache uno SHAP TreeExplainer.

    Args:
        model: Modello XGBoost/LightGBM

    Returns:
        shap.TreeExplainer
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Installa shap: pip install shap")

    # Usa id del modello come chiave cache
    model_id = id(model)

    if model_id not in _explainer_cache:
        _explainer_cache[model_id] = shap.TreeExplainer(model)

    return _explainer_cache[model_id]


def get_shap_factors(
    model: Any,
    X_row: pd.DataFrame,
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Calcola i top N fattori SHAP per una predizione.

    Args:
        model: Modello addestrato
        X_row: Singola riga di feature (DataFrame 1 riga)
        top_n: Numero di fattori da restituire

    Returns:
        Lista di dict con: feature, value, shap_value, direction
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP non installato")
        return []

    try:
        explainer = get_explainer(model)
        shap_values = explainer.shap_values(X_row)

        # Per classificazione binaria, prendi i SHAP della classe positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Converti a numpy array se necessario
        shap_values = np.array(shap_values).flatten()

        # Ordina per valore assoluto
        feature_names = X_row.columns.tolist()
        feature_values = X_row.values.flatten()

        indices = np.argsort(np.abs(shap_values))[::-1][:top_n]

        factors = []
        for idx in indices:
            shap_val = float(shap_values[idx])
            factors.append({
                'feature': feature_names[idx],
                'value': float(feature_values[idx]),
                'shap_value': round(shap_val, 4),
                'direction': 'aumenta' if shap_val > 0 else 'diminuisce',
                'impact': _describe_impact(feature_names[idx], feature_values[idx], shap_val),
            })

        return factors

    except Exception as e:
        logger.error(f"Errore calcolo SHAP: {e}")
        return []


def _describe_impact(feature: str, value: float, shap_val: float) -> str:
    """
    Genera una descrizione human-readable dell'impatto.

    Args:
        feature: Nome della feature
        value: Valore della feature
        shap_val: Valore SHAP

    Returns:
        Stringa descrittiva
    """
    direction = "aumenta" if shap_val > 0 else "diminuisce"

    # Mapping feature -> descrizione
    descriptions = {
        'days_since_last': f"Giorni dall'ultimo intervento ({int(value)}d) {direction} il rischio",
        'days_ratio': f"Rapporto giorni/media ({value:.1f}x) {direction} il rischio",
        'km_dal_ultimo_intervento': f"Km percorsi ({int(value)}) {direction} il rischio",
        'km_ratio': f"Rapporto km/media ({value:.1f}x) {direction} il rischio",
        'interventions_last_90d': f"{int(value)} interventi negli ultimi 90gg {direction} il rischio",
        'recurrence_12m': f"{int(value)} interventi nell'ultimo anno {direction} il rischio",
        'cost_trend': f"Trend costi ({value:.2f}) {direction} il rischio",
        'vehicle_age_days': f"Età veicolo ({int(value/365)} anni) {direction} il rischio",
        'km_stimati_settimana': f"Km settimanali ({int(value)}) {direction} il rischio",
    }

    return descriptions.get(feature, f"{feature}={value:.2f} {direction} il rischio")


def get_shap_summary(
    model: Any,
    X: pd.DataFrame,
    plot: bool = False
) -> pd.DataFrame:
    """
    Calcola un summary delle importanze SHAP globali.

    Args:
        model: Modello addestrato
        X: Dataset di feature
        plot: Se True, mostra il plot

    Returns:
        DataFrame con feature importance SHAP
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Installa shap: pip install shap")

    explainer = get_explainer(model)
    shap_values = explainer.shap_values(X)

    # Per classificazione binaria
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Calcola importanza media
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': mean_abs_shap,
    }).sort_values('shap_importance', ascending=False)

    if plot:
        try:
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, X, show=False)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"Impossibile creare plot: {e}")

    return importance_df


def explain_prediction(
    model: Any,
    X_row: pd.DataFrame,
    plot: bool = False
) -> Dict[str, Any]:
    """
    Spiega una singola predizione con SHAP.

    Args:
        model: Modello addestrato
        X_row: Singola riga di feature
        plot: Se True, mostra il waterfall plot

    Returns:
        Dict con spiegazione completa
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Installa shap: pip install shap")

    explainer = get_explainer(model)

    # Calcola SHAP values
    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    # Top factors
    top_factors = get_shap_factors(model, X_row, top_n=5)

    result = {
        'expected_value': float(expected_value),
        'prediction_contribution': float(shap_values.sum()),
        'final_prediction': float(expected_value + shap_values.sum()),
        'top_factors': top_factors,
        'all_shap_values': dict(zip(X_row.columns, shap_values.flatten().tolist())),
    }

    if plot:
        try:
            import matplotlib.pyplot as plt

            # Crea shap.Explanation per il waterfall plot
            explanation = shap.Explanation(
                values=shap_values.flatten(),
                base_values=expected_value,
                data=X_row.values.flatten(),
                feature_names=X_row.columns.tolist(),
            )

            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"Impossibile creare waterfall plot: {e}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test con modello dummy
    print("Test SHAP explain module")

    # Crea modello di test
    try:
        from xgboost import XGBClassifier
        import numpy as np

        X = pd.DataFrame({
            'days_since_last': [30, 60, 90, 120, 150],
            'km_ratio': [0.8, 1.2, 1.5, 2.0, 2.5],
            'interventions_last_90d': [1, 2, 0, 3, 1],
        })
        y = pd.Series([0, 0, 1, 1, 1])

        model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        # Test get_shap_factors
        factors = get_shap_factors(model, X.iloc[[0]])
        print(f"\nTop factors per prima riga:")
        for f in factors:
            print(f"  {f['feature']}: {f['shap_value']:.4f} ({f['direction']})")

        # Test explain_prediction
        explanation = explain_prediction(model, X.iloc[[0]])
        print(f"\nExplanation:")
        print(f"  Expected value: {explanation['expected_value']:.4f}")
        print(f"  Prediction contribution: {explanation['prediction_contribution']:.4f}")

    except ImportError as e:
        print(f"Dipendenze mancanti: {e}")
