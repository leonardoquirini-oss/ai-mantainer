"""
Tools per l'agente LLM di gestione manutenzioni
"""

from .maintenance_tools import (
    carica_dati_csv,
    get_statistiche_dataset,
    genera_piano_manutenzione,
    analizza_weibull,
    analizza_sopravvivenza,
    analizza_hazard_ratio,
    analizza_mezzo,
    get_mezzi_critici,
    get_previsioni_guasti,
    TOOLS_SCHEMA,
)

__all__ = [
    "carica_dati_csv",
    "get_statistiche_dataset",
    "genera_piano_manutenzione",
    "analizza_weibull",
    "analizza_sopravvivenza",
    "analizza_hazard_ratio",
    "analizza_mezzo",
    "get_mezzi_critici",
    "get_previsioni_guasti",
    "TOOLS_SCHEMA",
]
