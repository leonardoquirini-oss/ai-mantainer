"""
Tools SQLite per l'agente LLM.

Questi tools permettono all'agente di interrogare il database
SQLite con lo storico manutenzioni.

Tools disponibili:
- get_vehicle_history: storico interventi per targa
- get_fleet_risk_summary: risk score flotta
- get_vehicle_risk: risk dettagliato singolo veicolo
- get_high_risk_vehicles: veicoli ad alto rischio
- search_vehicles: cerca veicoli per pattern targa
"""

import json
import logging
from typing import Optional, Dict, Any, List

# Import database
import sys
from pathlib import Path

# Aggiungi root progetto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from db.init import get_connection

logger = logging.getLogger("maintenance-agent.sqlite_tools")


def get_vehicle_history(targa: str, limit: int = 20) -> Dict[str, Any]:
    """
    Recupera gli ultimi interventi di manutenzione per un veicolo.

    Args:
        targa: Targa del veicolo o identificativo container
        limit: Numero massimo di interventi da restituire (default: 20)

    Returns:
        Dizionario con targa, totale trovati, lista interventi
    """
    conn = get_connection()

    # Normalizza targa (case-insensitive, trimma spazi)
    targa_clean = targa.strip().upper()

    rows = conn.execute("""
        SELECT
            azienda,
            seriale_doc,
            data_intervento,
            descrizione,
            dettaglio,
            costo,
            data_imm
        FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
        ORDER BY data_intervento DESC
        LIMIT ?
    """, (targa_clean, limit)).fetchall()

    conn.close()

    # Conta totale (senza limit)
    conn = get_connection()
    total = conn.execute("""
        SELECT COUNT(*)
        FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
    """, (targa_clean,)).fetchone()[0]
    conn.close()

    # Calcola costo totale
    costo_totale = sum(r["costo"] or 0 for r in rows)

    return {
        "targa": targa_clean,
        "total_interventions": total,
        "showing": len(rows),
        "costo_totale": round(costo_totale, 2),
        "interventions": [dict(r) for r in rows]
    }


def search_vehicles(pattern: str, limit: int = 50) -> Dict[str, Any]:
    """
    Cerca veicoli per pattern targa.

    Args:
        pattern: Pattern di ricerca (es. "AA 93" per tutte le targhe che iniziano con AA 93)
        limit: Numero massimo di risultati

    Returns:
        Lista di veicoli con statistiche
    """
    conn = get_connection()

    # Usa LIKE per pattern matching
    pattern_clean = pattern.strip().upper()
    pattern_like = f"%{pattern_clean}%"

    rows = conn.execute("""
        SELECT
            targa,
            azienda,
            COUNT(*) as n_interventi,
            SUM(costo) as costo_totale,
            MIN(data_intervento) as primo_intervento,
            MAX(data_intervento) as ultimo_intervento,
            MAX(data_imm) as data_imm
        FROM maintenance_history
        WHERE UPPER(targa) LIKE ?
        GROUP BY targa, azienda
        ORDER BY n_interventi DESC
        LIMIT ?
    """, (pattern_like, limit)).fetchall()

    conn.close()

    return {
        "pattern": pattern_clean,
        "found": len(rows),
        "vehicles": [dict(r) for r in rows]
    }


def get_fleet_risk_summary(azienda: Optional[str] = None) -> Dict[str, Any]:
    """
    Restituisce il risk score attuale di tutta la flotta.

    Ordinato per rischio decrescente.
    Filtrabile per azienda (G, B, C) o su tutto il gruppo.

    Args:
        azienda: Filtra per azienda (G, B, C). None per tutto il gruppo.

    Returns:
        Dizionario con filtro, totale veicoli, lista fleet
    """
    conn = get_connection()

    query = """
        SELECT
            rs.targa,
            rs.azienda,
            rs.risk_score,
            rs.risk_level,
            rs.prob_fail_7d,
            rs.prob_fail_30d,
            rs.top_factors,
            rs.computed_at
        FROM risk_scores rs
        INNER JOIN (
            SELECT targa, azienda, MAX(computed_at) AS last_computed
            FROM risk_scores
            GROUP BY targa, azienda
        ) latest ON rs.targa = latest.targa
                 AND rs.azienda = latest.azienda
                 AND rs.computed_at = latest.last_computed
    """

    params = []
    if azienda:
        query += " WHERE rs.azienda = ?"
        params.append(azienda.upper())
    query += " ORDER BY rs.risk_score DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    # Parse top_factors JSON
    fleet = []
    for r in rows:
        row = dict(r)
        if row.get("top_factors"):
            try:
                row["top_factors"] = json.loads(row["top_factors"])
            except json.JSONDecodeError:
                pass
        fleet.append(row)

    return {
        "azienda_filter": azienda.upper() if azienda else "tutte",
        "total_vehicles": len(fleet),
        "fleet": fleet
    }


def get_vehicle_risk(targa: str) -> Dict[str, Any]:
    """
    Restituisce il risk score dettagliato per un singolo veicolo.

    Include i fattori SHAP e le feature calcolate.

    Args:
        targa: Targa del veicolo

    Returns:
        Risk score con dettagli o errore se non trovato
    """
    conn = get_connection()
    targa_clean = targa.strip().upper()

    # Ultimo risk score
    score = conn.execute("""
        SELECT * FROM risk_scores
        WHERE UPPER(TRIM(targa)) = ?
        ORDER BY computed_at DESC
        LIMIT 1
    """, (targa_clean,)).fetchone()

    # Ultime feature
    features = conn.execute("""
        SELECT * FROM vehicle_features
        WHERE UPPER(TRIM(targa)) = ?
        ORDER BY computed_at DESC
        LIMIT 1
    """, (targa_clean,)).fetchone()

    conn.close()

    if not score:
        # Nessun risk score, ma potrebbe avere storico
        history = get_vehicle_history(targa_clean, limit=5)
        if history["total_interventions"] > 0:
            return {
                "targa": targa_clean,
                "warning": "Nessun risk score calcolato per questo veicolo",
                "has_history": True,
                "total_interventions": history["total_interventions"],
                "last_interventions": history["interventions"][:3]
            }
        return {"error": f"Nessun dato trovato per targa {targa_clean}"}

    result = dict(score)

    # Parse top_factors
    if result.get("top_factors"):
        try:
            result["top_factors"] = json.loads(result["top_factors"])
        except json.JSONDecodeError:
            pass

    # Aggiungi feature
    if features:
        result["features"] = dict(features)

    return result


def get_high_risk_vehicles(
    threshold: float = 70.0,
    azienda: Optional[str] = None
) -> Dict[str, Any]:
    """
    Restituisce tutti i veicoli con risk score sopra la soglia.

    Default soglia: 70 (livello arancio+rosso).

    Args:
        threshold: Soglia minima di risk score (0-100)
        azienda: Filtra per azienda (G, B, C). None per tutto il gruppo.

    Returns:
        Lista veicoli ad alto rischio con dettagli
    """
    conn = get_connection()

    query = """
        SELECT
            rs.targa,
            rs.azienda,
            rs.risk_score,
            rs.risk_level,
            rs.prob_fail_7d,
            rs.top_factors,
            rs.computed_at
        FROM risk_scores rs
        INNER JOIN (
            SELECT targa, azienda, MAX(computed_at) AS last_computed
            FROM risk_scores
            GROUP BY targa, azienda
        ) latest ON rs.targa = latest.targa
                 AND rs.azienda = latest.azienda
                 AND rs.computed_at = latest.last_computed
        WHERE rs.risk_score >= ?
    """

    params = [threshold]
    if azienda:
        query += " AND rs.azienda = ?"
        params.append(azienda.upper())
    query += " ORDER BY rs.risk_score DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    vehicles = []
    for r in rows:
        row = dict(r)
        if row.get("top_factors"):
            try:
                row["top_factors"] = json.loads(row["top_factors"])
            except json.JSONDecodeError:
                pass
        vehicles.append(row)

    return {
        "threshold": threshold,
        "azienda_filter": azienda.upper() if azienda else "tutte",
        "count": len(vehicles),
        "vehicles": vehicles
    }


def get_vehicle_summary(targa: str) -> Dict[str, Any]:
    """
    Restituisce un riepilogo completo per un veicolo.

    Combina storico interventi e risk score (se disponibile).

    Args:
        targa: Targa del veicolo

    Returns:
        Riepilogo completo del veicolo
    """
    targa_clean = targa.strip().upper()

    # Storico interventi
    history = get_vehicle_history(targa_clean, limit=10)

    # Risk score
    risk = get_vehicle_risk(targa_clean)

    # Statistiche base
    conn = get_connection()
    stats = conn.execute("""
        SELECT
            azienda,
            COUNT(*) as n_interventi,
            SUM(costo) as costo_totale,
            AVG(costo) as costo_medio,
            MIN(data_intervento) as primo_intervento,
            MAX(data_intervento) as ultimo_intervento,
            MAX(data_imm) as data_imm
        FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
        GROUP BY azienda
    """, (targa_clean,)).fetchone()
    conn.close()

    summary = {
        "targa": targa_clean,
        "found": history["total_interventions"] > 0
    }

    if stats:
        summary["azienda"] = stats["azienda"]
        summary["data_immatricolazione"] = stats["data_imm"]
        summary["statistiche"] = {
            "n_interventi": stats["n_interventi"],
            "costo_totale": round(stats["costo_totale"] or 0, 2),
            "costo_medio": round(stats["costo_medio"] or 0, 2),
            "primo_intervento": stats["primo_intervento"],
            "ultimo_intervento": stats["ultimo_intervento"]
        }

    summary["ultimi_interventi"] = history["interventions"][:5]

    if "error" not in risk:
        summary["risk"] = risk

    return summary


# =============================================================================
# TOOLS PER KM E VIAGGI
# =============================================================================

def get_vehicle_km_summary(targa: str) -> Dict[str, Any]:
    """
    Riepilogo chilometrico di una targa (motrice o semirimorchio).

    Usa la vista vehicle_km che unifica i km di entrambi i ruoli.

    Args:
        targa: Targa del veicolo

    Returns:
        Statistiche km: totali, ultimi 30gg, dall'ultimo intervento, ecc.
    """
    conn = get_connection()
    targa_clean = targa.strip().upper()

    # Statistiche generali dalla vista vehicle_km
    summary = conn.execute("""
        SELECT
            COUNT(*)            AS totale_viaggi,
            SUM(km)             AS km_totali,
            ROUND(AVG(km), 1)   AS km_medi_viaggio,
            MAX(km)             AS km_viaggio_massimo,
            MIN(data_viaggio)   AS primo_viaggio,
            MAX(data_viaggio)   AS ultimo_viaggio
        FROM vehicle_km
        WHERE UPPER(TRIM(targa)) = ?
    """, (targa_clean,)).fetchone()

    # Km ultimi 30 giorni
    km_30d = conn.execute("""
        SELECT COALESCE(SUM(km), 0) FROM vehicle_km
        WHERE UPPER(TRIM(targa)) = ?
          AND data_viaggio >= DATE('now', '-30 days')
    """, (targa_clean,)).fetchone()[0]

    # Km ultimi 90 giorni
    km_90d = conn.execute("""
        SELECT COALESCE(SUM(km), 0) FROM vehicle_km
        WHERE UPPER(TRIM(targa)) = ?
          AND data_viaggio >= DATE('now', '-90 days')
    """, (targa_clean,)).fetchone()[0]

    # Ultimo intervento
    last_intervention = conn.execute("""
        SELECT MAX(data_intervento) FROM maintenance_history
        WHERE UPPER(TRIM(targa)) = ?
    """, (targa_clean,)).fetchone()[0]

    # Km dall'ultimo intervento
    km_dal_intervento = None
    if last_intervention:
        km_dal_intervento = conn.execute("""
            SELECT COALESCE(SUM(km), 0) FROM vehicle_km
            WHERE UPPER(TRIM(targa)) = ? AND data_viaggio >= ?
        """, (targa_clean, last_intervention)).fetchone()[0]

    # Dettaglio ruoli (quante volte motrice vs semirimorchio)
    ruoli = conn.execute("""
        SELECT ruolo, COUNT(*) AS n_viaggi, ROUND(SUM(km), 1) AS km_totali
        FROM vehicle_km
        WHERE UPPER(TRIM(targa)) = ?
        GROUP BY ruolo
    """, (targa_clean,)).fetchall()

    conn.close()

    if not summary or summary[0] == 0:
        return {
            "targa": targa_clean,
            "found": False,
            "message": "Nessun viaggio trovato per questa targa"
        }

    return {
        "targa": targa_clean,
        "found": True,
        "totale_viaggi": summary[0],
        "km_totali": round(summary[1] or 0, 1),
        "km_medi_per_viaggio": summary[2],
        "km_viaggio_massimo": summary[3],
        "primo_viaggio": summary[4],
        "ultimo_viaggio": summary[5],
        "km_ultimi_30d": round(km_30d, 1),
        "km_ultimi_90d": round(km_90d, 1),
        "ultimo_intervento": last_intervention,
        "km_dal_ultimo_intervento": round(km_dal_intervento, 1) if km_dal_intervento else None,
        "dettaglio_ruoli": [dict(r) for r in ruoli],
    }


def get_trip_history(targa: str, limit: int = 20) -> Dict[str, Any]:
    """
    Ultimi N viaggi di una targa, con ruolo e controparte.

    Mostra se la targa era motrice o semirimorchio in ogni viaggio
    e con quale altro veicolo viaggiava.

    Args:
        targa: Targa del veicolo
        limit: Numero viaggi da restituire (default: 20)

    Returns:
        Lista viaggi con dettagli
    """
    conn = get_connection()
    targa_clean = targa.strip().upper()

    rows = conn.execute("""
        SELECT
            t.bg,
            t.data_viaggio,
            t.km,
            t.targa_motrice,
            t.targa_semirimorchio,
            CASE
                WHEN UPPER(TRIM(t.targa_motrice)) = ? THEN 'motrice'
                ELSE 'semirimorchio'
            END AS ruolo_in_viaggio,
            CASE
                WHEN UPPER(TRIM(t.targa_motrice)) = ? THEN t.targa_semirimorchio
                ELSE t.targa_motrice
            END AS controparte
        FROM trips t
        WHERE UPPER(TRIM(t.targa_motrice)) = ?
           OR UPPER(TRIM(t.targa_semirimorchio)) = ?
        ORDER BY t.data_viaggio DESC
        LIMIT ?
    """, (targa_clean, targa_clean, targa_clean, targa_clean, limit)).fetchall()

    conn.close()

    return {
        "targa": targa_clean,
        "total_shown": len(rows),
        "viaggi": [dict(r) for r in rows]
    }


def get_fleet_km_ranking(azienda: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """
    Classifica la flotta per km percorsi dall'ultimo intervento.

    Prioritizza i veicoli più usurati per utilizzo reale.

    Args:
        azienda: Filtra per azienda (G, B, C). None per tutto il gruppo.
        limit: Numero massimo di risultati

    Returns:
        Ranking veicoli per km dall'ultimo intervento
    """
    conn = get_connection()

    # Query che calcola km dall'ultimo intervento per ogni veicolo
    query = """
        WITH last_interventions AS (
            SELECT targa, MAX(data_intervento) as last_date, azienda
            FROM maintenance_history
            GROUP BY targa
        )
        SELECT
            vk.targa,
            li.azienda,
            ROUND(SUM(vk.km), 0) AS km_dal_intervento,
            li.last_date AS ultimo_intervento,
            COUNT(vk.bg) AS viaggi_dal_intervento
        FROM vehicle_km vk
        LEFT JOIN last_interventions li ON UPPER(TRIM(vk.targa)) = UPPER(TRIM(li.targa))
        WHERE vk.data_viaggio >= COALESCE(li.last_date, '1900-01-01')
    """

    params = []
    if azienda:
        query += " AND li.azienda = ?"
        params.append(azienda.upper())

    query += """
        GROUP BY vk.targa
        ORDER BY km_dal_intervento DESC
        LIMIT ?
    """
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return {
        "azienda_filter": azienda.upper() if azienda else "tutte",
        "count": len(rows),
        "ranking": [dict(r) for r in rows]
    }


def get_vehicle_combinations(targa: str, limit: int = 20) -> Dict[str, Any]:
    """
    Mostra con quali altri veicoli una targa ha viaggiato più spesso.

    Utile per correlare guasti a combinazioni specifiche.

    Args:
        targa: Targa del veicolo
        limit: Numero massimo di combinazioni

    Returns:
        Lista di controparti con frequenza e km totali
    """
    conn = get_connection()
    targa_clean = targa.strip().upper()

    # Trova combinazioni come motrice
    as_motrice = conn.execute("""
        SELECT
            targa_semirimorchio AS controparte,
            'semirimorchio' AS ruolo_controparte,
            COUNT(*) AS n_viaggi,
            ROUND(SUM(km), 1) AS km_totali
        FROM trips
        WHERE UPPER(TRIM(targa_motrice)) = ?
          AND targa_semirimorchio IS NOT NULL
        GROUP BY targa_semirimorchio
        ORDER BY n_viaggi DESC
        LIMIT ?
    """, (targa_clean, limit)).fetchall()

    # Trova combinazioni come semirimorchio
    as_semirimorchio = conn.execute("""
        SELECT
            targa_motrice AS controparte,
            'motrice' AS ruolo_controparte,
            COUNT(*) AS n_viaggi,
            ROUND(SUM(km), 1) AS km_totali
        FROM trips
        WHERE UPPER(TRIM(targa_semirimorchio)) = ?
        GROUP BY targa_motrice
        ORDER BY n_viaggi DESC
        LIMIT ?
    """, (targa_clean, limit)).fetchall()

    conn.close()

    return {
        "targa": targa_clean,
        "come_motrice": [dict(r) for r in as_motrice],
        "come_semirimorchio": [dict(r) for r in as_semirimorchio]
    }


# Esporta tutti i tools
SQLITE_TOOLS = {
    # Manutenzioni
    "get_vehicle_history": get_vehicle_history,
    "search_vehicles": search_vehicles,
    "get_fleet_risk_summary": get_fleet_risk_summary,
    "get_vehicle_risk": get_vehicle_risk,
    "get_high_risk_vehicles": get_high_risk_vehicles,
    "get_vehicle_summary": get_vehicle_summary,
    # Km e viaggi
    "get_vehicle_km_summary": get_vehicle_km_summary,
    "get_trip_history": get_trip_history,
    "get_fleet_km_ranking": get_fleet_km_ranking,
    "get_vehicle_combinations": get_vehicle_combinations,
}
