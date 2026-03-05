"""
Definizioni JSON Schema per i tools dell'agente.

Usati per il function calling con Claude o altri LLM.
"""

SQLITE_TOOL_DEFINITIONS = [
    {
        "name": "get_vehicle_history",
        "description": (
            "Recupera gli ultimi interventi di manutenzione per un veicolo "
            "specifico identificato dalla targa. Mostra data, descrizione, "
            "dettaglio e costo di ogni intervento."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "targa": {
                    "type": "string",
                    "description": "Targa del veicolo o identificativo del container"
                },
                "limit": {
                    "type": "integer",
                    "description": "Numero massimo di interventi da restituire (default: 20)",
                    "default": 20
                }
            },
            "required": ["targa"]
        }
    },
    {
        "name": "search_vehicles",
        "description": (
            "Cerca veicoli per pattern nella targa. Utile quando non si conosce "
            "la targa esatta. Restituisce lista di veicoli con statistiche."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Pattern di ricerca (es. 'AA 93' per tutte le targhe che contengono 'AA 93')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Numero massimo di risultati (default: 50)",
                    "default": 50
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "get_vehicle_summary",
        "description": (
            "Restituisce un riepilogo completo per un veicolo: statistiche totali, "
            "ultimi interventi e risk score (se disponibile). "
            "Usa questo tool quando vuoi avere una visione d'insieme di un veicolo."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "targa": {
                    "type": "string",
                    "description": "Targa del veicolo"
                }
            },
            "required": ["targa"]
        }
    },
    {
        "name": "get_fleet_risk_summary",
        "description": (
            "Restituisce il risk score attuale di tutta la flotta ordinato per "
            "rischio decrescente. Filtrabile per azienda del gruppo. "
            "Utile per avere una panoramica della situazione di rischio."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "azienda": {
                    "type": "string",
                    "description": (
                        "Azienda da filtrare: G (Guido Bernardini), B (Bernardini), "
                        "C (Cosmo). Ometti per vedere tutto il gruppo."
                    ),
                    "enum": ["G", "B", "C"]
                }
            }
        }
    },
    {
        "name": "get_vehicle_risk",
        "description": (
            "Restituisce il risk score dettagliato per un singolo veicolo, "
            "inclusi i fattori che contribuiscono al rischio (SHAP values) "
            "e le feature calcolate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "targa": {
                    "type": "string",
                    "description": "Targa del veicolo"
                }
            },
            "required": ["targa"]
        }
    },
    {
        "name": "get_high_risk_vehicles",
        "description": (
            "Restituisce tutti i veicoli con risk score sopra una soglia. "
            "Usa questo tool per la prioritizzazione degli interventi urgenti. "
            "Default soglia: 70 (livello arancio + rosso)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Soglia minima di risk score (0-100). Default: 70.",
                    "default": 70
                },
                "azienda": {
                    "type": "string",
                    "description": "Filtra per azienda: G, B o C. Ometti per tutto il gruppo.",
                    "enum": ["G", "B", "C"]
                }
            }
        }
    },
    # =========================================================================
    # TOOLS KM E VIAGGI
    # =========================================================================
    {
        "name": "get_vehicle_km_summary",
        "description": (
            "Restituisce il riepilogo chilometrico di una targa (motrice o semirimorchio). "
            "Include km totali, km ultimi 30/90 giorni, km percorsi dall'ultimo intervento "
            "e il dettaglio dei ruoli (quante volte motrice vs semirimorchio)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "targa": {
                    "type": "string",
                    "description": "Targa del veicolo"
                }
            },
            "required": ["targa"]
        }
    },
    {
        "name": "get_trip_history",
        "description": (
            "Restituisce gli ultimi N viaggi di una targa, specificando il ruolo "
            "(motrice o semirimorchio) e con quale controparte viaggiava. "
            "Utile per analizzare combinazioni ricorrenti e pattern di utilizzo."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "targa": {
                    "type": "string",
                    "description": "Targa del veicolo"
                },
                "limit": {
                    "type": "integer",
                    "description": "Numero viaggi da restituire (default: 20)",
                    "default": 20
                }
            },
            "required": ["targa"]
        }
    },
    {
        "name": "get_fleet_km_ranking",
        "description": (
            "Classifica tutta la flotta per km percorsi dall'ultimo intervento di manutenzione. "
            "Usa questo tool per identificare i veicoli più usurati per utilizzo reale."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "azienda": {
                    "type": "string",
                    "description": "Filtra per azienda: G, B o C. Ometti per tutto il gruppo.",
                    "enum": ["G", "B", "C"]
                },
                "limit": {
                    "type": "integer",
                    "description": "Numero massimo di risultati (default: 50)",
                    "default": 50
                }
            }
        }
    },
    {
        "name": "get_vehicle_combinations",
        "description": (
            "Mostra con quali altri veicoli una targa ha viaggiato più spesso. "
            "Utile per correlare guasti a combinazioni specifiche "
            "(es. un semirimorchio pesante che usura di più le motrici con cui viaggia)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "targa": {
                    "type": "string",
                    "description": "Targa del veicolo"
                },
                "limit": {
                    "type": "integer",
                    "description": "Numero massimo di combinazioni (default: 20)",
                    "default": 20
                }
            },
            "required": ["targa"]
        }
    }
]


# Unisci con tool definitions esistenti (se presenti)
def get_all_tool_definitions() -> list:
    """
    Restituisce tutte le definizioni dei tools disponibili.
    """
    return SQLITE_TOOL_DEFINITIONS.copy()
