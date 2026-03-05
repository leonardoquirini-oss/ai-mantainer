"""
Tools per l'agente LLM di gestione manutenzioni
"""

from .maintenance_tools import (
    carica_dati_csv,
    carica_dati_adhoc,
    categorizza_intervento,
    get_categorie_disponibili,
    get_statistiche_dataset,
    genera_piano_manutenzione,
    analizza_weibull,
    analizza_sopravvivenza,
    analizza_hazard_ratio,
    analizza_mezzo,
    get_mezzi_critici,
    get_previsioni_guasti,
    ottimizza_manutenzioni_combinate,
    genera_calendario_manutenzioni,
    TOOLS_SCHEMA,
)

from .sqlite_tools import (
    # Manutenzioni
    get_vehicle_history,
    search_vehicles,
    get_vehicle_summary,
    get_fleet_risk_summary,
    get_vehicle_risk,
    get_high_risk_vehicles,
    # Km e viaggi
    get_vehicle_km_summary,
    get_trip_history,
    get_fleet_km_ranking,
    get_vehicle_combinations,
    SQLITE_TOOLS,
)

from .tool_definitions import (
    SQLITE_TOOL_DEFINITIONS,
    get_all_tool_definitions,
)

__all__ = [
    # Maintenance tools
    "carica_dati_csv",
    "carica_dati_adhoc",
    "categorizza_intervento",
    "get_categorie_disponibili",
    "get_statistiche_dataset",
    "genera_piano_manutenzione",
    "analizza_weibull",
    "analizza_sopravvivenza",
    "analizza_hazard_ratio",
    "analizza_mezzo",
    "get_mezzi_critici",
    "get_previsioni_guasti",
    "ottimizza_manutenzioni_combinate",
    "genera_calendario_manutenzioni",
    "TOOLS_SCHEMA",
    # SQLite tools - Manutenzioni
    "get_vehicle_history",
    "search_vehicles",
    "get_vehicle_summary",
    "get_fleet_risk_summary",
    "get_vehicle_risk",
    "get_high_risk_vehicles",
    # SQLite tools - Km e viaggi
    "get_vehicle_km_summary",
    "get_trip_history",
    "get_fleet_km_ranking",
    "get_vehicle_combinations",
    "SQLITE_TOOLS",
    # Tool definitions
    "SQLITE_TOOL_DEFINITIONS",
    "get_all_tool_definitions",
]
