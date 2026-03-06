"""
Tools per l'agente LLM di gestione manutenzioni predittive.

Questi tools sono usati dall'agente LLM per rispondere a domande
sulla manutenzione della flotta.
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any

from ..models import (
    EventoManutenzione,
    Mezzo,
    DatasetManutenzione,
    PianoManutenzione,
)
from ..maintainer import MaintenanceOptimizer
from ..maintainer.history_learner import get_data_loader
from ..utils.categorizzatore import categorizza_riga, CATEGORIE
from ..connectors import get_adhoc_connector

# Import tools SQLite per km e viaggi
from .sqlite_tools import (
    get_vehicle_history,
    search_vehicles,
    get_vehicle_summary,
    get_vehicle_km_summary,
    get_trip_history,
    get_fleet_km_ranking,
    get_vehicle_combinations,
)


# Cache globale dataset
_dataset_cache: Optional[DatasetManutenzione] = None


def _get_dataset() -> DatasetManutenzione:
    """Ottiene dataset dal database SQLite"""
    global _dataset_cache
    if _dataset_cache is None:
        loader = get_data_loader()
        _dataset_cache = loader.carica_da_sqlite()
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


def carica_dati_adhoc(
    data_start: Optional[str] = None,
    data_stop: Optional[str] = None
) -> Dict[str, Any]:
    """
    Carica dati di manutenzione in tempo reale dall'API AdHoc.

    Questa funzione si connette al database AdHoc per recuperare
    lo storico completo degli interventi di manutenzione.

    Args:
        data_start: Data inizio periodo (formato YYYY-MM-DD o DD/MM/YYYY).
                   Default: 01/01/2015
        data_stop: Data fine periodo (formato YYYY-MM-DD o DD/MM/YYYY).
                  Default: oggi

    Returns:
        Statistiche sul caricamento
    """
    global _dataset_cache
    loader = get_data_loader()

    # Parse date
    start = None
    stop = None

    if data_start:
        for fmt in ["%Y-%m-%d", "%d/%m/%Y"]:
            try:
                start = datetime.strptime(data_start, fmt).date()
                break
            except ValueError:
                continue

    if data_stop:
        for fmt in ["%Y-%m-%d", "%d/%m/%Y"]:
            try:
                stop = datetime.strptime(data_stop, fmt).date()
                break
            except ValueError:
                continue

    try:
        _dataset_cache = loader.carica_da_adhoc(start, stop)
        loader.salva_cache(_dataset_cache, "dataset_adhoc")

        stats = _dataset_cache.statistiche_base()
        return {
            "success": True,
            "fonte": "AdHoc API",
            "periodo": {
                "da": (start or date(2015, 1, 1)).isoformat(),
                "a": (stop or date.today()).isoformat()
            },
            "eventi_caricati": stats["totale_eventi"],
            "mezzi_caricati": stats["totale_mezzi"],
            "tipi_mezzo": stats["tipi_mezzo"],
            "categorie": stats.get("categorie", []),
            "eventi_per_categoria": stats.get("eventi_per_categoria", {}),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "hint": "Verifica che l'API AdHoc sia raggiungibile e l'API key sia corretta"
        }


def categorizza_intervento(descrizione: str, dettaglio: str = "") -> Dict[str, Any]:
    """
    Categorizza un intervento di manutenzione in base alla descrizione.

    Usa pattern regex per classificare l'intervento in una delle 15 macro-categorie:
    01. PNEUMATICI
    02. IMPIANTO FRENANTE
    03. SOSPENSIONI E AMMORTIZZATORI
    04. CARROZZERIA CONTAINER / CASSE MOBILI
    05. TELONI E COPERTURE
    06. IMPIANTO ELETTRICO E LUCI
    07. MOTORE E MECCANICA MOTRICE
    08. MOZZI E RUOTE
    09. REVISIONE E CONTROLLI PERIODICI
    10. ATTREZZATURE SILOS / CISTERNA
    11. ROTOCELLA E TWIST LOCK
    12. SOCCORSO E INTERVENTI FUORI SEDE
    13. MATERIALI DI CONSUMO E FLUIDI
    14. STRUTTURA METALLICA E SALDATURE
    15. ALLESTIMENTO E PERSONALIZZAZIONE

    Args:
        descrizione: Descrizione dell'intervento
        dettaglio: Dettaglio aggiuntivo dell'intervento (opzionale)

    Returns:
        Categorie assegnate e mapping a TipoGuasto
    """
    from ..utils.categorizzatore import categoria_to_tipo_guasto

    categorie = categorizza_riga(descrizione, dettaglio)

    result = {
        "descrizione_input": descrizione,
        "dettaglio_input": dettaglio,
        "categorie": categorie,
        "categoria_principale": categorie[0] if categorie else "NON CLASSIFICATO",
        "tipo_guasto_mappato": categoria_to_tipo_guasto(categorie[0] if categorie else "NON CLASSIFICATO")
    }

    if len(categorie) > 1:
        result["note"] = f"Intervento multi-categoria: {len(categorie)} categorie rilevate"

    return result


def get_categorie_disponibili() -> Dict[str, Any]:
    """
    Restituisce l'elenco delle categorie disponibili per la classificazione.

    Returns:
        Lista delle 15 macro-categorie con descrizione
    """
    from ..utils.categorizzatore import CATEGORIA_TO_TIPO_GUASTO

    return {
        "totale_categorie": len(CATEGORIE),
        "categorie": [
            {
                "codice": nome,
                "tipo_guasto_mappato": CATEGORIA_TO_TIPO_GUASTO.get(nome, "altro")
            }
            for nome, _ in CATEGORIE
        ]
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
    affidabilita_target: float = 0.90,
    usa_categorie: bool = True
) -> Dict[str, Any]:
    """
    Genera piano di manutenzione ordinaria basato su analisi statistica.

    Applica:
    - Analisi Weibull per classificare guasti (con pesi proporzionali)
    - Kaplan-Meier per curve di sopravvivenza
    - NHPP per mezzi con guasti ricorrenti

    Se usa_categorie=True (default), utilizza le 15 categorie AdHoc
    con supporto per interventi multi-categoria e pesi proporzionali.

    Args:
        affidabilita_target: Affidabilità target (default 90%)
        usa_categorie: Se True, usa le 15 categorie AdHoc (default True)

    Returns:
        Piano di manutenzione
    """
    dataset = _get_dataset()
    optimizer = MaintenanceOptimizer()

    # Verifica se ci sono categorie disponibili negli eventi
    ha_categorie = any(e.categorie for e in dataset.eventi)

    if usa_categorie and ha_categorie:
        # Usa il nuovo metodo basato su categorie con pesi
        piano = optimizer.genera_piano_manutenzione_categorie(
            dataset.eventi,
            dataset.mezzi,
            affidabilita_target
        )
    else:
        # Fallback al metodo tradizionale basato su TipoGuasto
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
                "categoria": i.tipo_guasto,  # Ora contiene la categoria AdHoc
                "intervallo_mesi": i.intervallo_mesi,
                "applicabile": i.applicabile,
                "classificazione": i.classificazione.value if hasattr(i.classificazione, 'value') else str(i.classificazione),
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


def get_storico_guasti_mezzo(mezzo_id: str, limit: int = 50) -> Dict[str, Any]:
    """
    Mostra lo storico completo degli interventi di manutenzione per un mezzo specifico.

    Restituisce l'elenco dettagliato di tutti i guasti/interventi registrati
    per il mezzo, ordinati per data dal più recente al più vecchio.

    Args:
        mezzo_id: ID/targa del mezzo da analizzare
        limit: Numero massimo di interventi da restituire (default 50)

    Returns:
        Dizionario con storico interventi e statistiche
    """
    dataset = _get_dataset()

    # Ottieni eventi per questo mezzo
    eventi = dataset.get_eventi_per_mezzo(mezzo_id)

    if not eventi:
        return {
            "found": False,
            "mezzo_id": mezzo_id,
            "error": f"Nessun intervento trovato per il mezzo {mezzo_id}"
        }

    # Ordina per data decrescente (più recenti prima)
    eventi_ordinati = sorted(eventi, key=lambda e: e.data_evento, reverse=True)

    # Limita risultati
    eventi_limitati = eventi_ordinati[:limit]

    # Prepara lista interventi
    interventi = []
    for e in eventi_limitati:
        interventi.append({
            "data": e.data_evento.isoformat(),
            "tipo_guasto": e.tipo_guasto.value if hasattr(e.tipo_guasto, 'value') else str(e.tipo_guasto),
            "categorie": [c.value for c in e.categorie] if e.categorie else [],
            "descrizione": e.descrizione,
            "costo": e.costo,
            "eta_mezzo_mesi": e.eta_mezzo_mesi
        })

    # Calcola statistiche
    costo_totale = sum(e.costo for e in eventi)
    tipi_guasto = {}
    for e in eventi:
        tipo = e.tipo_guasto.value if hasattr(e.tipo_guasto, 'value') else str(e.tipo_guasto)
        tipi_guasto[tipo] = tipi_guasto.get(tipo, 0) + 1

    # Trova mezzo per info aggiuntive
    mezzo = None
    for m in dataset.mezzi:
        if m.mezzo_id == mezzo_id:
            mezzo = m
            break

    return {
        "found": True,
        "mezzo_id": mezzo_id,
        "tipo_mezzo": mezzo.tipo_mezzo.value if mezzo and hasattr(mezzo.tipo_mezzo, 'value') else None,
        "data_immatricolazione": mezzo.data_immatricolazione.isoformat() if mezzo and mezzo.data_immatricolazione else None,
        "totale_interventi": len(eventi),
        "interventi_mostrati": len(eventi_limitati),
        "costo_totale": round(costo_totale, 2),
        "primo_intervento": eventi_ordinati[-1].data_evento.isoformat() if eventi_ordinati else None,
        "ultimo_intervento": eventi_ordinati[0].data_evento.isoformat() if eventi_ordinati else None,
        "distribuzione_tipi_guasto": tipi_guasto,
        "interventi": interventi
    }


def verifica_stato_mezzo(targa: str) -> Dict[str, Any]:
    """
    Verifica se un veicolo/container è attualmente attivo nella flotta.

    Interroga il database AdHoc per verificare se il mezzo non è stato
    dismesso e risulta ancora in uso.

    Args:
        targa: Targa del veicolo o ID del container

    Returns:
        Stato del mezzo (attivo/non attivo)
    """
    # Normalizza la targa (uppercase, rimuovi spazi)
    targa_normalizzata = targa.upper().replace(" ", "")

    try:
        connector = get_adhoc_connector()
        mezzi_attivi = connector.get_mezzi_attivi()

        # Verifica se la targa è nel set dei mezzi attivi
        is_attivo = targa_normalizzata in mezzi_attivi

        return {
            "targa": targa,
            "targa_normalizzata": targa_normalizzata,
            "attivo": is_attivo,
            "stato": "ATTIVO" if is_attivo else "NON ATTIVO / DISMESSO",
            "totale_mezzi_attivi": len(mezzi_attivi)
        }
    except Exception as e:
        return {
            "targa": targa,
            "error": str(e),
            "hint": "Verifica che l'API AdHoc sia raggiungibile"
        }


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


def ottimizza_manutenzioni_combinate(
    finestra_mesi: int = 3,
    affidabilita_target: float = 0.90,
    costo_fermo_giornaliero: float = 500.0
) -> Dict[str, Any]:
    """
    Ottimizza il piano di manutenzione combinando interventi vicini nel tempo.

    Evita doppi fermi raggruppando manutenzioni che cadono entro una finestra
    temporale, anticipando quelle successive al primo intervento del gruppo.

    Args:
        finestra_mesi: Finestra temporale per raggruppare manutenzioni (default 3 mesi)
        affidabilita_target: Affidabilità target (default 0.90)
        costo_fermo_giornaliero: Costo stimato per giorno di fermo mezzo (default €500)

    Returns:
        Piano ottimizzato con manutenzioni combinate e risparmio stimato
    """
    dataset = _get_dataset()
    optimizer = MaintenanceOptimizer()

    # Genera piano base
    piano = optimizer.genera_piano_manutenzione(
        dataset.eventi,
        dataset.mezzi,
        affidabilita_target
    )

    # Raggruppa intervalli per tipo_mezzo
    intervalli_per_mezzo = {}
    for intervallo in piano.intervalli:
        if not intervallo.applicabile or not intervallo.intervallo_mesi:
            continue

        tipo_mezzo = intervallo.tipo_mezzo
        if tipo_mezzo not in intervalli_per_mezzo:
            intervalli_per_mezzo[tipo_mezzo] = []

        intervalli_per_mezzo[tipo_mezzo].append({
            "tipo_guasto": intervallo.tipo_guasto,
            "intervallo_mesi": intervallo.intervallo_mesi,
            "motivazione": intervallo.motivazione
        })

    # Ottimizza combinando manutenzioni vicine
    piani_combinati = {}
    totale_fermi_evitati = 0
    totale_risparmio = 0.0

    for tipo_mezzo, intervalli in intervalli_per_mezzo.items():
        if len(intervalli) < 2:
            # Solo un tipo di manutenzione, niente da combinare
            piani_combinati[tipo_mezzo] = {
                "manutenzioni_singole": intervalli,
                "manutenzioni_combinate": [],
                "fermi_evitati": 0
            }
            continue

        # Ordina per intervallo
        intervalli_ordinati = sorted(intervalli, key=lambda x: x["intervallo_mesi"])

        # Trova gruppi di manutenzioni combinabili
        gruppi = []
        gruppo_corrente = [intervalli_ordinati[0]]

        for i in range(1, len(intervalli_ordinati)):
            intervallo_corrente = intervalli_ordinati[i]
            intervallo_precedente = gruppo_corrente[-1]

            # Se la differenza è <= finestra_mesi, combina
            diff = intervallo_corrente["intervallo_mesi"] - intervallo_precedente["intervallo_mesi"]
            if diff <= finestra_mesi:
                gruppo_corrente.append(intervallo_corrente)
            else:
                gruppi.append(gruppo_corrente)
                gruppo_corrente = [intervallo_corrente]

        gruppi.append(gruppo_corrente)

        # Genera piano combinato
        manutenzioni_combinate = []
        fermi_evitati = 0

        for gruppo in gruppi:
            if len(gruppo) == 1:
                # Manutenzione singola
                manutenzioni_combinate.append({
                    "intervallo_mesi": gruppo[0]["intervallo_mesi"],
                    "tipi_guasto": [gruppo[0]["tipo_guasto"]],
                    "combinata": False,
                    "risparmio_fermi": 0
                })
            else:
                # Manutenzione combinata - usa l'intervallo più breve
                intervallo_combinato = min(g["intervallo_mesi"] for g in gruppo)
                tipi_guasto = [g["tipo_guasto"] for g in gruppo]
                fermi_risparmiati = len(gruppo) - 1
                fermi_evitati += fermi_risparmiati

                manutenzioni_combinate.append({
                    "intervallo_mesi": intervallo_combinato,
                    "tipi_guasto": tipi_guasto,
                    "combinata": True,
                    "risparmio_fermi": fermi_risparmiati,
                    "nota": f"Combina {len(gruppo)} interventi in uno"
                })

        piani_combinati[tipo_mezzo] = {
            "manutenzioni_combinate": manutenzioni_combinate,
            "fermi_evitati_anno": fermi_evitati,
            "risparmio_stimato_anno": fermi_evitati * costo_fermo_giornaliero
        }

        totale_fermi_evitati += fermi_evitati
        totale_risparmio += fermi_evitati * costo_fermo_giornaliero

    # Conta mezzi per tipo
    mezzi_per_tipo = {}
    for mezzo in dataset.mezzi:
        tipo = mezzo.tipo_mezzo.value if hasattr(mezzo.tipo_mezzo, 'value') else mezzo.tipo_mezzo
        mezzi_per_tipo[tipo] = mezzi_per_tipo.get(tipo, 0) + 1

    # Calcola risparmio totale flotta
    risparmio_flotta = 0.0
    for tipo_mezzo, piano_tipo in piani_combinati.items():
        n_mezzi = mezzi_per_tipo.get(tipo_mezzo, 0)
        risparmio_flotta += piano_tipo.get("risparmio_stimato_anno", 0) * n_mezzi

    return {
        "finestra_combinazione_mesi": finestra_mesi,
        "affidabilita_target": affidabilita_target,
        "costo_fermo_giornaliero": costo_fermo_giornaliero,
        "piani_per_tipo_mezzo": piani_combinati,
        "riepilogo": {
            "fermi_evitati_per_mezzo_anno": totale_fermi_evitati,
            "risparmio_per_mezzo_anno": totale_risparmio,
            "mezzi_in_flotta": mezzi_per_tipo,
            "risparmio_totale_flotta_anno": risparmio_flotta
        }
    }


def genera_calendario_manutenzioni(
    data_inizio: str = None,
    mesi_orizzonte: int = 24,
    affidabilita_target: float = 0.90
) -> Dict[str, Any]:
    """
    Genera un calendario di manutenzioni programmate per la flotta.

    Combina automaticamente manutenzioni vicine e mostra quando intervenire
    su ogni mezzo specifico.

    Args:
        data_inizio: Data inizio pianificazione (YYYY-MM-DD), default oggi
        mesi_orizzonte: Orizzonte temporale in mesi (default 24)
        affidabilita_target: Affidabilità target (default 0.90)

    Returns:
        Calendario con date e interventi per ogni mezzo
    """
    from datetime import datetime, timedelta

    if data_inizio:
        try:
            inizio = datetime.strptime(data_inizio, "%Y-%m-%d").date()
        except ValueError:
            inizio = date.today()
    else:
        inizio = date.today()

    dataset = _get_dataset()
    optimizer = MaintenanceOptimizer()

    # Genera piano ottimizzato
    piano_combinato = ottimizza_manutenzioni_combinate(
        finestra_mesi=3,
        affidabilita_target=affidabilita_target
    )

    # Genera calendario per ogni mezzo
    calendario = []

    for mezzo in dataset.mezzi:
        tipo_mezzo = mezzo.tipo_mezzo.value if hasattr(mezzo.tipo_mezzo, 'value') else mezzo.tipo_mezzo

        if tipo_mezzo not in piano_combinato["piani_per_tipo_mezzo"]:
            continue

        piano_tipo = piano_combinato["piani_per_tipo_mezzo"][tipo_mezzo]

        # Trova ultimo intervento per questo mezzo (per ogni tipo guasto)
        ultimi_interventi = {}
        for evento in dataset.eventi:
            if evento.mezzo_id != mezzo.mezzo_id:
                continue
            tipo_guasto = evento.tipo_guasto.value if hasattr(evento.tipo_guasto, 'value') else evento.tipo_guasto
            if tipo_guasto not in ultimi_interventi or evento.data_evento > ultimi_interventi[tipo_guasto]:
                ultimi_interventi[tipo_guasto] = evento.data_evento

        # Calcola prossimi interventi
        interventi_mezzo = []

        for manutenzione in piano_tipo.get("manutenzioni_combinate", []):
            intervallo = manutenzione["intervallo_mesi"]
            tipi_guasto = manutenzione["tipi_guasto"]

            # Trova la data più recente tra i tipi guasto del gruppo
            data_riferimento = inizio
            for tipo_g in tipi_guasto:
                if tipo_g in ultimi_interventi:
                    if ultimi_interventi[tipo_g] > data_riferimento:
                        data_riferimento = ultimi_interventi[tipo_g]

            # Calcola prossima data
            prossima_data = data_riferimento + timedelta(days=intervallo * 30)

            # Se la data è nel passato, programma da oggi
            if prossima_data < inizio:
                prossima_data = inizio + timedelta(days=30)  # Entro un mese

            # Verifica se rientra nell'orizzonte
            fine_orizzonte = inizio + timedelta(days=mesi_orizzonte * 30)
            if prossima_data <= fine_orizzonte:
                interventi_mezzo.append({
                    "data": prossima_data.isoformat(),
                    "tipi_guasto": tipi_guasto,
                    "combinata": manutenzione.get("combinata", False),
                    "intervallo_mesi": intervallo
                })

        if interventi_mezzo:
            # Ordina per data
            interventi_mezzo.sort(key=lambda x: x["data"])
            calendario.append({
                "mezzo_id": mezzo.mezzo_id,
                "tipo_mezzo": tipo_mezzo,
                "interventi_programmati": interventi_mezzo
            })

    # Ordina calendario per data primo intervento
    calendario.sort(key=lambda x: x["interventi_programmati"][0]["data"] if x["interventi_programmati"] else "9999")

    # Raggruppa per mese
    interventi_per_mese = {}
    for mezzo_cal in calendario:
        for intervento in mezzo_cal["interventi_programmati"]:
            mese = intervento["data"][:7]  # YYYY-MM
            if mese not in interventi_per_mese:
                interventi_per_mese[mese] = []
            interventi_per_mese[mese].append({
                "mezzo_id": mezzo_cal["mezzo_id"],
                "tipo_mezzo": mezzo_cal["tipo_mezzo"],
                "tipi_guasto": intervento["tipi_guasto"],
                "combinata": intervento["combinata"]
            })

    return {
        "data_inizio": inizio.isoformat(),
        "orizzonte_mesi": mesi_orizzonte,
        "affidabilita_target": affidabilita_target,
        "calendario_per_mezzo": calendario,
        "riepilogo_per_mese": dict(sorted(interventi_per_mese.items())),
        "totale_interventi": sum(len(m["interventi_programmati"]) for m in calendario)
    }


# Mapping nome tool -> funzione
TOOL_FUNCTIONS = {
    "carica_dati_csv": carica_dati_csv,
    "carica_dati_adhoc": carica_dati_adhoc,
    "categorizza_intervento": categorizza_intervento,
    "get_categorie_disponibili": get_categorie_disponibili,
    "get_statistiche_dataset": get_statistiche_dataset,
    "genera_piano_manutenzione": genera_piano_manutenzione,
    "analizza_weibull": analizza_weibull,
    "analizza_sopravvivenza": analizza_sopravvivenza,
    "analizza_hazard_ratio": analizza_hazard_ratio,
    "analizza_mezzo": analizza_mezzo,
    "get_mezzi_critici": get_mezzi_critici,
    "get_previsioni_guasti": get_previsioni_guasti,
    "ottimizza_manutenzioni_combinate": ottimizza_manutenzioni_combinate,
    "genera_calendario_manutenzioni": genera_calendario_manutenzioni,
    "get_storico_guasti_mezzo": get_storico_guasti_mezzo,
    "verifica_stato_mezzo": verifica_stato_mezzo,
    # Tools SQLite km e viaggi
    "get_vehicle_history": get_vehicle_history,
    "search_vehicles": search_vehicles,
    "get_vehicle_summary": get_vehicle_summary,
    "get_vehicle_km_summary": get_vehicle_km_summary,
    "get_trip_history": get_trip_history,
    "get_fleet_km_ranking": get_fleet_km_ranking,
    "get_vehicle_combinations": get_vehicle_combinations,
}


# =============================================================================
# FORMATTAZIONE OUTPUT TABELLARE
# =============================================================================

def _format_table(rows: List[Dict], columns: List[str] = None, headers: List[str] = None) -> str:
    """Formatta lista di dict come tabella markdown."""
    if not rows:
        return "_Nessun dato_"

    if columns is None:
        columns = list(rows[0].keys())
    if headers is None:
        headers = columns

    # Header + separator
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|"
    ]

    # Righe
    for row in rows:
        values = [str(row.get(c, "-") or "-") for c in columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _format_kv(data: Dict, exclude: List[str] = None) -> str:
    """Formatta dict come coppie chiave-valore."""
    exclude = exclude or []
    lines = []
    for k, v in data.items():
        if k in exclude:
            continue
        if isinstance(v, list) and v and isinstance(v[0], dict):
            continue  # Le liste di dict le gestiamo separatamente
        if isinstance(v, dict):
            continue  # Skip nested dict
        lines.append(f"**{k}:** {v}")
    return "\n".join(lines)


# Configurazione formattazione per tool specifici
TOOL_FORMAT_CONFIG = {
    "get_storico_guasti_mezzo": {
        "list_key": "interventi",
        "columns": ["data", "tipo_guasto", "descrizione", "costo"],
        "headers": ["Data", "Tipo", "Descrizione", "Costo"],
        "summary_keys": ["mezzo_id", "tipo_mezzo", "totale_interventi", "costo_totale", "primo_intervento", "ultimo_intervento"]
    },
    "get_trip_history": {
        "list_key": "viaggi",
        "columns": ["data_viaggio", "km", "ruolo_in_viaggio", "controparte"],
        "headers": ["Data", "Km", "Ruolo", "Con"],
        "summary_keys": ["targa"]
    },
    "get_vehicle_km_summary": {
        "format": "kv",
        "exclude": ["found", "dettaglio_ruoli"],
        "sections": {"dettaglio_ruoli": {"columns": ["ruolo", "n_viaggi", "km_totali"], "headers": ["Ruolo", "Viaggi", "Km"]}}
    },
    "get_vehicle_history": {
        "list_key": "interventions",
        "columns": ["data_intervento", "descrizione", "costo"],
        "headers": ["Data", "Descrizione", "Costo"],
        "summary_keys": ["targa", "total_interventions", "costo_totale"]
    },
    "get_fleet_km_ranking": {
        "list_key": "ranking",
        "columns": ["targa", "km_dal_intervento", "ultimo_intervento", "viaggi_dal_intervento"],
        "headers": ["Targa", "Km", "Ultimo Intervento", "N.Viaggi"]
    },
    "search_vehicles": {
        "list_key": "vehicles",
        "columns": ["targa", "azienda", "n_interventi", "costo_totale", "ultimo_intervento"],
        "headers": ["Targa", "Az.", "Interventi", "Costo Tot.", "Ultimo"],
        "summary_keys": ["pattern", "found"]
    },
    "get_vehicle_combinations": {
        "format": "multi_table",
        "summary_keys": ["targa"],
        "sections": {
            "come_motrice": {"title": "Come Motrice", "columns": ["controparte", "n_viaggi", "km_totali"], "headers": ["Semirimorchio", "Viaggi", "Km"]},
            "come_semirimorchio": {"title": "Come Semirimorchio", "columns": ["controparte", "n_viaggi", "km_totali"], "headers": ["Motrice", "Viaggi", "Km"]}
        }
    },
    "get_vehicle_summary": {
        "format": "custom_summary"
    }
}


def _format_tool_output(tool_name: str, result: Any) -> str:
    """Formatta output tool in markdown leggibile."""
    import json

    config = TOOL_FORMAT_CONFIG.get(tool_name)

    if config is None:
        # Fallback: JSON per tools senza configurazione specifica
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, ensure_ascii=False, default=str)
        return str(result)

    output_parts = []

    # Summary keys (es. targa, totale)
    for key in config.get("summary_keys", []):
        if key in result:
            output_parts.append(f"**{key}:** {result[key]}")

    # Tabella principale
    if "list_key" in config:
        data = result.get(config["list_key"], [])
        table = _format_table(data, config.get("columns"), config.get("headers"))
        output_parts.append(table)

    # Multi-table (es. vehicle_combinations)
    elif config.get("format") == "multi_table":
        for section_key, section_cfg in config.get("sections", {}).items():
            if section_key in result and result[section_key]:
                output_parts.append(f"\n**{section_cfg['title']}:**")
                output_parts.append(_format_table(
                    result[section_key],
                    section_cfg.get("columns"),
                    section_cfg.get("headers")
                ))

    # Key-value format
    elif config.get("format") == "kv":
        output_parts.append(_format_kv(result, config.get("exclude", [])))
        # Sezioni tabellari opzionali
        for section_key, section_cfg in config.get("sections", {}).items():
            if section_key in result and result[section_key]:
                output_parts.append(f"\n**{section_key.replace('_', ' ').title()}:**")
                output_parts.append(_format_table(
                    result[section_key],
                    section_cfg.get("columns"),
                    section_cfg.get("headers")
                ))

    # Custom summary per vehicle_summary
    elif config.get("format") == "custom_summary":
        if result.get("found"):
            output_parts.append(f"**Targa:** {result.get('targa')}")
            if result.get("azienda"):
                output_parts.append(f"**Azienda:** {result.get('azienda')}")
            if result.get("statistiche"):
                stats = result["statistiche"]
                output_parts.append(f"\n**Statistiche:**")
                output_parts.append(f"- Interventi: {stats.get('n_interventi', 0)}")
                output_parts.append(f"- Costo totale: €{stats.get('costo_totale', 0):.2f}")
                output_parts.append(f"- Primo: {stats.get('primo_intervento')}")
                output_parts.append(f"- Ultimo: {stats.get('ultimo_intervento')}")
            if result.get("ultimi_interventi"):
                output_parts.append(f"\n**Ultimi interventi:**")
                output_parts.append(_format_table(
                    result["ultimi_interventi"],
                    ["data_intervento", "descrizione", "costo"],
                    ["Data", "Descrizione", "Costo"]
                ))
        else:
            output_parts.append(f"Nessun dato trovato per targa {result.get('targa')}")

    return "\n\n".join(output_parts) if output_parts else str(result)


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Esegue un tool e restituisce il risultato formattato in markdown.

    Args:
        tool_name: Nome del tool da eseguire
        arguments: Argomenti per il tool

    Returns:
        Risultato formattato come markdown leggibile
    """
    if tool_name not in TOOL_FUNCTIONS:
        return f"Errore: tool '{tool_name}' non trovato"

    try:
        func = TOOL_FUNCTIONS[tool_name]
        result = func(**arguments)

        if result is None:
            return "Nessun dato disponibile"
        if isinstance(result, list) and not result:
            return "Nessun risultato"

        return _format_tool_output(tool_name, result)
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
        "name": "carica_dati_adhoc",
        "description": "Carica dati di manutenzione in tempo reale dall'API AdHoc. Recupera lo storico completo degli interventi dal database aziendale.",
        "parameters": {
            "type": "object",
            "properties": {
                "data_start": {
                    "type": "string",
                    "description": "Data inizio periodo (YYYY-MM-DD o DD/MM/YYYY). Default: 01/01/2015"
                },
                "data_stop": {
                    "type": "string",
                    "description": "Data fine periodo (YYYY-MM-DD o DD/MM/YYYY). Default: oggi"
                }
            }
        }
    },
    {
        "name": "categorizza_intervento",
        "description": "Categorizza un intervento di manutenzione usando pattern regex. Classifica la descrizione in una delle 15 macro-categorie (pneumatici, freni, sospensioni, etc.)",
        "parameters": {
            "type": "object",
            "required": ["descrizione"],
            "properties": {
                "descrizione": {
                    "type": "string",
                    "description": "Descrizione dell'intervento di manutenzione"
                },
                "dettaglio": {
                    "type": "string",
                    "description": "Dettaglio aggiuntivo dell'intervento (opzionale)"
                }
            }
        }
    },
    {
        "name": "get_categorie_disponibili",
        "description": "Restituisce l'elenco delle 15 macro-categorie disponibili per la classificazione degli interventi",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_statistiche_dataset",
        "description": "Ottiene statistiche descrittive del dataset: numero eventi, mezzi, distribuzione per tipo",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "genera_piano_manutenzione",
        "description": "Genera piano di manutenzione ordinaria ottimizzato basato su analisi statistica (Weibull, Kaplan-Meier, NHPP). Usa le 15 categorie AdHoc con pesi proporzionali per interventi multi-categoria. Output azionabile per responsabile flotta.",
        "parameters": {
            "type": "object",
            "properties": {
                "affidabilita_target": {
                    "type": "number",
                    "description": "Affidabilità target (0-1). Default 0.90 = 90%",
                    "default": 0.90
                },
                "usa_categorie": {
                    "type": "boolean",
                    "description": "Se True (default), usa le 15 categorie AdHoc con pesi proporzionali. Se False, usa il mapping semplificato TipoGuasto.",
                    "default": True
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
    },
    {
        "name": "ottimizza_manutenzioni_combinate",
        "description": "Ottimizza il piano combinando manutenzioni vicine nel tempo per evitare doppi fermi. Raggruppa interventi entro una finestra temporale e calcola il risparmio stimato.",
        "parameters": {
            "type": "object",
            "properties": {
                "finestra_mesi": {
                    "type": "integer",
                    "description": "Finestra temporale in mesi per raggruppare manutenzioni (default 3)",
                    "default": 3
                },
                "affidabilita_target": {
                    "type": "number",
                    "description": "Affidabilità target (0-1). Default 0.90",
                    "default": 0.90
                },
                "costo_fermo_giornaliero": {
                    "type": "number",
                    "description": "Costo stimato per giorno di fermo mezzo in euro (default 500)",
                    "default": 500.0
                }
            }
        }
    },
    {
        "name": "genera_calendario_manutenzioni",
        "description": "Genera calendario di manutenzioni programmate per ogni mezzo della flotta. Combina automaticamente interventi vicini e mostra quando intervenire.",
        "parameters": {
            "type": "object",
            "properties": {
                "data_inizio": {
                    "type": "string",
                    "description": "Data inizio pianificazione (YYYY-MM-DD). Default oggi"
                },
                "mesi_orizzonte": {
                    "type": "integer",
                    "description": "Orizzonte temporale in mesi (default 24)",
                    "default": 24
                },
                "affidabilita_target": {
                    "type": "number",
                    "description": "Affidabilità target (0-1). Default 0.90",
                    "default": 0.90
                }
            }
        }
    },
    {
        "name": "get_storico_guasti_mezzo",
        "description": "Mostra lo storico completo degli interventi di manutenzione per un mezzo specifico. Restituisce l'elenco dettagliato di tutti i guasti/interventi registrati, ordinati per data.",
        "parameters": {
            "type": "object",
            "required": ["mezzo_id"],
            "properties": {
                "mezzo_id": {
                    "type": "string",
                    "description": "ID/targa del mezzo da analizzare (es. 'AD 24573', 'GBTU 1226')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Numero massimo di interventi da restituire (default: 50)",
                    "default": 50
                }
            }
        }
    },
    {
        "name": "verifica_stato_mezzo",
        "description": "Verifica se un veicolo/container è attualmente attivo nella flotta (non dismesso). Interroga il database AdHoc in tempo reale.",
        "parameters": {
            "type": "object",
            "required": ["targa"],
            "properties": {
                "targa": {
                    "type": "string",
                    "description": "Targa del veicolo o ID del container (es. 'GBTU 1226', 'AD 24573')"
                }
            }
        }
    },
    # =========================================================================
    # TOOLS SQLITE: STORICO VEICOLI E KM
    # =========================================================================
    {
        "name": "get_vehicle_history",
        "description": "Recupera gli ultimi interventi di manutenzione per un veicolo specifico. Mostra data, descrizione, dettaglio e costo di ogni intervento.",
        "parameters": {
            "type": "object",
            "required": ["targa"],
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
            }
        }
    },
    {
        "name": "search_vehicles",
        "description": "Cerca veicoli per pattern nella targa. Utile quando non si conosce la targa esatta. Restituisce lista di veicoli con statistiche.",
        "parameters": {
            "type": "object",
            "required": ["pattern"],
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
            }
        }
    },
    {
        "name": "get_vehicle_summary",
        "description": "Restituisce un riepilogo completo per un veicolo: statistiche totali, ultimi interventi e risk score (se disponibile).",
        "parameters": {
            "type": "object",
            "required": ["targa"],
            "properties": {
                "targa": {
                    "type": "string",
                    "description": "Targa del veicolo"
                }
            }
        }
    },
    {
        "name": "get_vehicle_km_summary",
        "description": "Restituisce il riepilogo chilometrico di una targa (motrice o semirimorchio). Include km totali, km ultimi 30/90 giorni, km dall'ultimo intervento e dettaglio ruoli.",
        "parameters": {
            "type": "object",
            "required": ["targa"],
            "properties": {
                "targa": {
                    "type": "string",
                    "description": "Targa del veicolo"
                }
            }
        }
    },
    {
        "name": "get_trip_history",
        "description": "Restituisce gli ultimi N viaggi di una targa, specificando il ruolo (motrice o semirimorchio) e con quale controparte viaggiava.",
        "parameters": {
            "type": "object",
            "required": ["targa"],
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
            }
        }
    },
    {
        "name": "get_fleet_km_ranking",
        "description": "Classifica la flotta per km percorsi dall'ultimo intervento di manutenzione. Utile per identificare i veicoli più usurati.",
        "parameters": {
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
        "description": "Mostra con quali altri veicoli una targa ha viaggiato più spesso. Utile per correlare guasti a combinazioni specifiche.",
        "parameters": {
            "type": "object",
            "required": ["targa"],
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
            }
        }
    }
]
