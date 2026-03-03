"""
Tools per l'agente LLM di gestione manutenzioni predittive.

Questi tools sono usati dall'agente LLM per rispondere a domande
sulla manutenzione della flotta.
"""

from datetime import date
from typing import List, Optional, Dict, Any

from ..models import (
    EventoManutenzione,
    Mezzo,
    DatasetManutenzione,
    PianoManutenzione,
)
from ..maintainer import MaintenanceOptimizer
from ..maintainer.history_learner import get_data_loader


# Cache globale dataset
_dataset_cache: Optional[DatasetManutenzione] = None


def _get_dataset() -> DatasetManutenzione:
    """Ottiene dataset dalla cache o lo carica"""
    global _dataset_cache
    if _dataset_cache is None:
        loader = get_data_loader()
        _dataset_cache = loader.carica_cache("dataset")
        if _dataset_cache is None:
            # Dataset vuoto se non c'è cache
            _dataset_cache = DatasetManutenzione()
    return _dataset_cache


def carica_dati_csv(filepath: str) -> Dict[str, Any]:
    """
    Carica dati di manutenzione da file CSV.

    Args:
        filepath: Path al file CSV

    Returns:
        Statistiche sul caricamento
    """
    global _dataset_cache
    loader = get_data_loader()
    _dataset_cache = loader.carica_da_csv(filepath)
    loader.salva_cache(_dataset_cache)

    stats = _dataset_cache.statistiche_base()
    return {
        "success": True,
        "eventi_caricati": stats["totale_eventi"],
        "mezzi_caricati": stats["totale_mezzi"],
        "tipi_mezzo": stats["tipi_mezzo"],
        "tipi_guasto": stats["tipi_guasto"],
    }


def get_statistiche_dataset() -> Dict[str, Any]:
    """
    Ottiene statistiche descrittive del dataset.

    Returns:
        Dizionario con statistiche
    """
    dataset = _get_dataset()
    return dataset.statistiche_base()


def genera_piano_manutenzione(
    affidabilita_target: float = 0.90
) -> Dict[str, Any]:
    """
    Genera piano di manutenzione ordinaria basato su analisi statistica.

    Applica:
    - Analisi Weibull per classificare guasti
    - Kaplan-Meier per curve di sopravvivenza
    - NHPP per mezzi con guasti ricorrenti

    Args:
        affidabilita_target: Affidabilità target (default 90%)

    Returns:
        Piano di manutenzione
    """
    dataset = _get_dataset()
    optimizer = MaintenanceOptimizer()

    piano = optimizer.genera_piano_manutenzione(
        dataset.eventi,
        dataset.mezzi,
        affidabilita_target
    )

    return {
        "data_generazione": piano.data_generazione.isoformat(),
        "statistiche": piano.statistiche,
        "intervalli_manutenzione": [
            {
                "tipo_mezzo": i.tipo_mezzo,
                "tipo_guasto": i.tipo_guasto,
                "intervallo_mesi": i.intervallo_mesi,
                "applicabile": i.applicabile,
                "motivazione": i.motivazione,
            }
            for i in piano.intervalli
        ],
        "mezzi_critici": piano.mezzi_critici,
        "report": piano.genera_report_testuale(),
    }


def analizza_weibull(
    tipo_mezzo: str,
    tipo_guasto: str,
    affidabilita_target: float = 0.90
) -> Optional[Dict[str, Any]]:
    """
    Esegue analisi Weibull per una combinazione tipo_mezzo x tipo_guasto.

    Classifica il pattern di guasto:
    - beta < 0.8: guasti infantili (problemi qualità)
    - 0.8 <= beta <= 1.2: guasti casuali
    - beta > 1.2: guasti da usura (manutenzione preventiva efficace)

    Args:
        tipo_mezzo: Tipo di mezzo
        tipo_guasto: Tipo di guasto
        affidabilita_target: Affidabilità target

    Returns:
        Risultato analisi Weibull
    """
    dataset = _get_dataset()
    optimizer = MaintenanceOptimizer()

    df = optimizer.prepara_dati_sopravvivenza(
        dataset.eventi,
        dataset.mezzi,
        tipo_guasto
    )

    if df.empty:
        return None

    result = optimizer.analisi_weibull(
        df, tipo_mezzo, tipo_guasto, affidabilita_target
    )

    if result:
        return result.to_dict()
    return None


def analizza_sopravvivenza(
    tipo_mezzo: Optional[str] = None,
    tipo_guasto: str = ""
) -> Optional[Dict[str, Any]]:
    """
    Esegue analisi Kaplan-Meier per stimare curva di sopravvivenza.

    S(t) = probabilità che un mezzo non abbia il guasto entro t mesi.

    Args:
        tipo_mezzo: Filtra per tipo mezzo (opzionale)
        tipo_guasto: Tipo di guasto da analizzare

    Returns:
        Risultato analisi Kaplan-Meier
    """
    if not tipo_guasto:
        return {"error": "Specificare tipo_guasto"}

    dataset = _get_dataset()
    optimizer = MaintenanceOptimizer()

    df = optimizer.prepara_dati_sopravvivenza(
        dataset.eventi,
        dataset.mezzi,
        tipo_guasto
    )

    if df.empty:
        return None

    result = optimizer.analisi_kaplan_meier(df, tipo_mezzo, tipo_guasto)

    if result:
        return result.to_dict()
    return None


def analizza_hazard_ratio(tipo_guasto: str) -> Optional[Dict[str, Any]]:
    """
    Esegue analisi Cox Proportional Hazards per confrontare
    rischio relativo tra tipi di mezzo.

    HR > 1: rischio maggiore rispetto al riferimento
    HR < 1: rischio minore

    Args:
        tipo_guasto: Tipo di guasto da analizzare

    Returns:
        Risultato analisi Cox PH
    """
    dataset = _get_dataset()
    optimizer = MaintenanceOptimizer()

    df = optimizer.prepara_dati_sopravvivenza(
        dataset.eventi,
        dataset.mezzi,
        tipo_guasto
    )

    if df.empty:
        return None

    result = optimizer.analisi_cox_ph(df, tipo_guasto)

    if result:
        return result.to_dict()
    return None


def analizza_mezzo(mezzo_id: str) -> Optional[Dict[str, Any]]:
    """
    Analizza trend guasti per un mezzo specifico usando NHPP.

    Identifica:
    - Deterioramento: guasti sempre più frequenti
    - Stabile: tasso costante
    - Miglioramento: guasti sempre meno frequenti

    Args:
        mezzo_id: ID del mezzo

    Returns:
        Risultato analisi NHPP
    """
    dataset = _get_dataset()
    optimizer = MaintenanceOptimizer()

    # Trova mezzo
    mezzo = None
    for m in dataset.mezzi:
        if m.mezzo_id == mezzo_id:
            mezzo = m
            break

    if not mezzo:
        return {"error": f"Mezzo {mezzo_id} non trovato"}

    result = optimizer.analisi_nhpp(dataset.eventi, mezzo)

    if result:
        return result.to_dict()
    return {"info": "Dati insufficienti per analisi NHPP (servono almeno 3 guasti)"}


def get_mezzi_critici() -> List[Dict[str, Any]]:
    """
    Identifica mezzi con trend di deterioramento (guasti crescenti).

    Returns:
        Lista mezzi critici con dettagli
    """
    dataset = _get_dataset()
    optimizer = MaintenanceOptimizer()

    critici = []
    for mezzo in dataset.mezzi:
        result = optimizer.analisi_nhpp(dataset.eventi, mezzo)
        if result and result.trend.value == "deterioramento":
            critici.append(result.to_dict())

    return critici


def get_previsioni_guasti(mesi: int = 12) -> List[Dict[str, Any]]:
    """
    Genera previsioni guasti per tutti i mezzi.

    Args:
        mesi: Orizzonte temporale in mesi

    Returns:
        Lista previsioni per mezzo
    """
    dataset = _get_dataset()
    optimizer = MaintenanceOptimizer()

    previsioni = []
    for mezzo in dataset.mezzi:
        result = optimizer.analisi_nhpp(dataset.eventi, mezzo)
        if result:
            previsioni.append({
                "mezzo_id": mezzo.mezzo_id,
                "tipo_mezzo": mezzo.tipo_mezzo.value if hasattr(mezzo.tipo_mezzo, 'value') else mezzo.tipo_mezzo,
                "guasti_attesi": result.guasti_attesi_12_mesi,
                "tempo_prossimo_guasto": result.tempo_prossimo_guasto_mesi,
                "trend": result.trend.value,
            })

    # Ordina per guasti attesi decrescenti
    previsioni.sort(key=lambda x: x["guasti_attesi"], reverse=True)
    return previsioni


# Mapping nome tool -> funzione
TOOL_FUNCTIONS = {
    "carica_dati_csv": carica_dati_csv,
    "get_statistiche_dataset": get_statistiche_dataset,
    "genera_piano_manutenzione": genera_piano_manutenzione,
    "analizza_weibull": analizza_weibull,
    "analizza_sopravvivenza": analizza_sopravvivenza,
    "analizza_hazard_ratio": analizza_hazard_ratio,
    "analizza_mezzo": analizza_mezzo,
    "get_mezzi_critici": get_mezzi_critici,
    "get_previsioni_guasti": get_previsioni_guasti,
}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Esegue un tool e restituisce il risultato come stringa.

    Args:
        tool_name: Nome del tool da eseguire
        arguments: Argomenti per il tool

    Returns:
        Risultato formattato come stringa
    """
    import json

    if tool_name not in TOOL_FUNCTIONS:
        return f"Errore: tool '{tool_name}' non trovato"

    try:
        func = TOOL_FUNCTIONS[tool_name]
        result = func(**arguments)

        # Formatta il risultato
        if result is None:
            return "Nessun dato disponibile"
        elif isinstance(result, dict):
            # Formatta dict come JSON leggibile
            return json.dumps(result, indent=2, ensure_ascii=False, default=str)
        elif isinstance(result, list):
            if not result:
                return "Nessun risultato"
            return json.dumps(result, indent=2, ensure_ascii=False, default=str)
        else:
            return str(result)
    except Exception as e:
        return f"Errore esecuzione tool '{tool_name}': {e}"


# Schema tools per LLM
TOOLS_SCHEMA = [
    {
        "name": "carica_dati_csv",
        "description": "Carica dati storici di manutenzione da file CSV. Colonne attese: mezzo_id, tipo_mezzo, tipo_guasto, data_evento, data_acquisto/immatricolazione",
        "parameters": {
            "type": "object",
            "required": ["filepath"],
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path al file CSV con lo storico manutenzioni"
                }
            }
        }
    },
    {
        "name": "get_statistiche_dataset",
        "description": "Ottiene statistiche descrittive del dataset: numero eventi, mezzi, distribuzione per tipo",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "genera_piano_manutenzione",
        "description": "Genera piano di manutenzione ordinaria ottimizzato basato su analisi statistica (Weibull, Kaplan-Meier, NHPP). Output azionabile per responsabile flotta.",
        "parameters": {
            "type": "object",
            "properties": {
                "affidabilita_target": {
                    "type": "number",
                    "description": "Affidabilità target (0-1). Default 0.90 = 90%",
                    "default": 0.90
                }
            }
        }
    },
    {
        "name": "analizza_weibull",
        "description": "Analisi Weibull per classificare pattern guasti: infantili (beta<0.8), casuali (0.8-1.2), usura (beta>1.2). Solo per usura la manutenzione preventiva è efficace.",
        "parameters": {
            "type": "object",
            "required": ["tipo_mezzo", "tipo_guasto"],
            "properties": {
                "tipo_mezzo": {
                    "type": "string",
                    "description": "Tipo di mezzo (semirimorchio, trattore, etc.)"
                },
                "tipo_guasto": {
                    "type": "string",
                    "description": "Tipo di guasto (pneumatici, freni, motore, etc.)"
                },
                "affidabilita_target": {
                    "type": "number",
                    "description": "Affidabilità target per calcolo intervallo manutenzione",
                    "default": 0.90
                }
            }
        }
    },
    {
        "name": "analizza_sopravvivenza",
        "description": "Analisi Kaplan-Meier: stima probabilità che un mezzo NON abbia un certo guasto entro t mesi. Gestisce dati censurati.",
        "parameters": {
            "type": "object",
            "required": ["tipo_guasto"],
            "properties": {
                "tipo_mezzo": {
                    "type": "string",
                    "description": "Filtra per tipo mezzo (opzionale)"
                },
                "tipo_guasto": {
                    "type": "string",
                    "description": "Tipo di guasto da analizzare"
                }
            }
        }
    },
    {
        "name": "analizza_hazard_ratio",
        "description": "Analisi Cox PH: confronta rischio relativo di guasto tra tipi di mezzo. HR>1 = rischio maggiore, HR<1 = rischio minore.",
        "parameters": {
            "type": "object",
            "required": ["tipo_guasto"],
            "properties": {
                "tipo_guasto": {
                    "type": "string",
                    "description": "Tipo di guasto da analizzare"
                }
            }
        }
    },
    {
        "name": "analizza_mezzo",
        "description": "Analisi NHPP per singolo mezzo: identifica trend (deterioramento/stabile/miglioramento) e prevede guasti futuri. Richiede almeno 3 guasti storici.",
        "parameters": {
            "type": "object",
            "required": ["mezzo_id"],
            "properties": {
                "mezzo_id": {
                    "type": "string",
                    "description": "ID del mezzo da analizzare"
                }
            }
        }
    },
    {
        "name": "get_mezzi_critici",
        "description": "Identifica mezzi con trend di deterioramento (guasti sempre più frequenti). Questi richiedono attenzione prioritaria o valutazione sostituzione.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_previsioni_guasti",
        "description": "Genera previsioni di guasti per tutti i mezzi nei prossimi mesi, ordinati per urgenza.",
        "parameters": {
            "type": "object",
            "properties": {
                "mesi": {
                    "type": "integer",
                    "description": "Orizzonte temporale in mesi",
                    "default": 12
                }
            }
        }
    }
]
