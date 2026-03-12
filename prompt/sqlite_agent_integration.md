# SQLite — Integrazione nell'Agente di Manutenzione Predittiva

**Contesto:** Dati storici manutenzioni provenienti da tre aziende del gruppo (G, B, C).  
**Obiettivo:** Strutturare un database SQLite che l'agente possa interrogare in modo efficiente, pronto per l'estensione futura con feature ML e risk score.

---

## Schema dati in ingresso

| Campo | Tipo | Note |
|---|---|---|
| `AZIENDA` | TEXT | `G` = Guido Bernardini, `B` = Bernardini, `C` = Cosmo |
| `SERIALE_DOC` | TEXT | Riferimento contabile. Più righe possono condividere lo stesso seriale a parità di AZIENDA |
| `DESCRIZIONE` | TEXT | Campo descrittivo dell'intervento |
| `DETTAGLIO` | TEXT | Specifica aggiuntiva dell'intervento |
| `TARGA` | TEXT | Identificativo del veicolo o container |
| `DATA_INTERVENTO` | TEXT → DATE | Data in cui è stato effettuato l'intervento |
| `CAUSALE` | TEXT | Tipologia intervento — **ignorare per ora** |
| `COSTO` | REAL | Costo dell'intervento |
| `DATA_IMM` | TEXT → DATE | Data di immatricolazione del veicolo |

> **Nota:** `SERIALE_DOC` non è una chiave univoca. Un intervento complesso (es. sostituzione freni + olio) può generare più righe con lo stesso seriale. La chiave univoca reale è `(AZIENDA, SERIALE_DOC, riga)`.

---

## Struttura del database

Tre tabelle con responsabilità separate:

```
maintenance_history   ← storico grezzo importato (append-only, non modificare mai)
        │
        ▼
vehicle_features      ← feature calcolate dalla pipeline notturna
        │
        ▼
risk_scores           ← output del modello ML, letti dall'agente LLM
```

---

## Schema SQL

```sql
-- ============================================================
-- TABELLA 1: storico grezzo (immutabile)
-- ============================================================
CREATE TABLE IF NOT EXISTS maintenance_history (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    azienda           TEXT    NOT NULL CHECK (azienda IN ('G', 'B', 'C')),
    seriale_doc       TEXT    NOT NULL,
    descrizione       TEXT,
    dettaglio         TEXT,
    targa             TEXT    NOT NULL,
    data_intervento   DATE    NOT NULL,
    causale           TEXT,
    costo             REAL,
    data_imm          DATE,
    inserted_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indici per le query più frequenti dell'agente
CREATE INDEX IF NOT EXISTS idx_mh_targa          ON maintenance_history(targa);
CREATE INDEX IF NOT EXISTS idx_mh_data           ON maintenance_history(data_intervento);
CREATE INDEX IF NOT EXISTS idx_mh_azienda_targa  ON maintenance_history(azienda, targa);
CREATE INDEX IF NOT EXISTS idx_mh_seriale        ON maintenance_history(azienda, seriale_doc);

-- ============================================================
-- TABELLA 2: feature calcolate per veicolo (aggiornata ogni notte)
-- ============================================================
CREATE TABLE IF NOT EXISTS vehicle_features (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    targa                       TEXT    NOT NULL,
    azienda                     TEXT    NOT NULL,
    computed_at                 DATETIME NOT NULL,

    -- Dati base
    data_imm                    DATE,
    vehicle_age_days            INTEGER,

    -- Feature temporali
    days_since_last_intervention INTEGER,
    avg_days_between_interventions REAL,
    days_ratio                  REAL,   -- days_since_last / avg_days_between

    -- Feature frequenza
    interventions_last_90d      INTEGER,
    interventions_last_365d     INTEGER,

    -- Feature costo
    cost_last_intervention      REAL,
    cost_avg_12m                REAL,
    cost_trend                  REAL,   -- cost_avg_3m / cost_avg_storico

    -- Km stimati (da aggiungere quando disponibili)
    km_stimati_settimana        REAL,   -- NULL finché non disponibili
    km_dal_ultimo_intervento    REAL,   -- NULL finché non disponibili

    UNIQUE(targa, azienda)  -- una sola riga per veicolo, sovrascritta ogni notte
);

CREATE INDEX IF NOT EXISTS idx_vf_targa   ON vehicle_features(targa);
CREATE INDEX IF NOT EXISTS idx_vf_azienda ON vehicle_features(azienda);

-- ============================================================
-- TABELLA 3: risk score prodotti dal modello ML
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    targa           TEXT    NOT NULL,
    azienda         TEXT    NOT NULL,
    computed_at     DATETIME NOT NULL,

    risk_score      REAL    NOT NULL,   -- 0-100
    risk_level      TEXT    NOT NULL,   -- 'verde' | 'giallo' | 'arancio' | 'rosso'
    prob_fail_7d    REAL,               -- probabilità guasto entro 7 giorni
    prob_fail_30d   REAL,               -- probabilità guasto entro 30 giorni
    prob_fail_90d   REAL,               -- probabilità guasto entro 90 giorni
    top_factors     TEXT,               -- JSON: [{"feature": "days_ratio", "impact": 1.8}, ...]
    note_agente     TEXT                -- testo generato dall'LLM (diagnosi breve)
);

CREATE INDEX IF NOT EXISTS idx_rs_targa      ON risk_scores(targa);
CREATE INDEX IF NOT EXISTS idx_rs_computed   ON risk_scores(computed_at);
CREATE INDEX IF NOT EXISTS idx_rs_level      ON risk_scores(risk_level);
```

---

## Inizializzazione del DB

```python
# db/init.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data/maintenance.db")

def get_connection():
    """Restituisce una connessione con le impostazioni consigliate."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # accesso per nome colonna
    conn.execute("PRAGMA journal_mode=WAL") # scritture concorrenti più sicure
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    with open("db/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()
    print(f"DB inizializzato: {DB_PATH}")

if __name__ == "__main__":
    init_db()
```

---

## Import dello storico Excel

```python
# db/ingest.py
import pandas as pd
import sqlite3
from pathlib import Path
from db.init import get_connection

COLUMN_MAP = {
    'AZIENDA':           'azienda',
    'SERIALE_DOC':       'seriale_doc',
    'DESCRIZIONE':       'descrizione',
    'DETTAGLIO':         'dettaglio',
    'TARGA':             'targa',
    'DATA_INTERVENTO':   'data_intervento',
    'CAUSALE':           'causale',
    'COSTO':             'costo',
    'DATA_IMM':          'data_imm',
}

def ingest_excel(filepath: str, skip_duplicates: bool = True):
    df = pd.read_excel(filepath)

    # Rinomina colonne
    df = df.rename(columns=COLUMN_MAP)

    # Pulizia date
    df['data_intervento'] = pd.to_datetime(df['data_intervento'], errors='coerce').dt.date
    df['data_imm']        = pd.to_datetime(df['data_imm'], errors='coerce').dt.date

    # Pulizia targa e azienda
    df['targa']   = df['targa'].astype(str).str.strip().str.upper()
    df['azienda'] = df['azienda'].astype(str).str.strip().str.upper()

    # Scarta righe senza targa o data
    df = df.dropna(subset=['targa', 'data_intervento'])

    # Scarta aziende non valide
    df = df[df['azienda'].isin(['G', 'B', 'C'])]

    conn = get_connection()

    if skip_duplicates:
        # Evita di reinserire record già presenti (basato su azienda+seriale+targa+data)
        existing = pd.read_sql(
            "SELECT azienda, seriale_doc, targa, data_intervento FROM maintenance_history",
            conn
        )
        df = df.merge(
            existing,
            on=['azienda', 'seriale_doc', 'targa', 'data_intervento'],
            how='left',
            indicator=True
        )
        new_records = df[df['_merge'] == 'left_only'].drop(columns=['_merge'])
    else:
        new_records = df

    if new_records.empty:
        print("Nessun nuovo record da inserire.")
        conn.close()
        return

    new_records[list(COLUMN_MAP.values())].to_sql(
        'maintenance_history',
        conn,
        if_exists='append',
        index=False
    )
    conn.commit()
    conn.close()
    print(f"Inseriti {len(new_records)} nuovi record.")
```

---

## Query tool per l'agente LLM

Questi sono i tool che l'agente chiama tramite function calling. Ogni funzione ritorna un dizionario JSON-serializzabile.

```python
# agent/tools.py
import json
from db.init import get_connection

def get_vehicle_history(targa: str, limit: int = 20) -> dict:
    """
    Ritorna gli ultimi N interventi per una targa.
    Tool name: get_vehicle_history
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            azienda,
            seriale_doc,
            data_intervento,
            descrizione,
            dettaglio,
            costo
        FROM maintenance_history
        WHERE targa = ?
        ORDER BY data_intervento DESC
        LIMIT ?
    """, (targa.upper(), limit)).fetchall()
    conn.close()

    return {
        "targa": targa,
        "total_found": len(rows),
        "interventions": [dict(r) for r in rows]
    }


def get_fleet_risk_summary(azienda: str = None) -> dict:
    """
    Ritorna il risk score attuale di tutta la flotta, ordinato per rischio decrescente.
    Filtrabile per azienda (G, B, C) o su tutto il gruppo.
    Tool name: get_fleet_risk_summary
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

    return {
        "azienda_filter": azienda or "tutte",
        "total_vehicles": len(rows),
        "fleet": [dict(r) for r in rows]
    }


def get_vehicle_risk(targa: str) -> dict:
    """
    Ritorna il risk score più recente per una targa specifica,
    con i fattori SHAP e lo storico feature.
    Tool name: get_vehicle_risk
    """
    conn = get_connection()

    score = conn.execute("""
        SELECT * FROM risk_scores
        WHERE targa = ?
        ORDER BY computed_at DESC
        LIMIT 1
    """, (targa.upper(),)).fetchone()

    features = conn.execute("""
        SELECT * FROM vehicle_features
        WHERE targa = ?
        ORDER BY computed_at DESC
        LIMIT 1
    """, (targa.upper(),)).fetchone()

    conn.close()

    if not score:
        return {"error": f"Nessun risk score trovato per targa {targa}"}

    result = dict(score)
    if result.get('top_factors'):
        result['top_factors'] = json.loads(result['top_factors'])
    if features:
        result['features'] = dict(features)

    return result


def get_high_risk_vehicles(threshold: float = 70.0, azienda: str = None) -> dict:
    """
    Ritorna tutti i veicoli con risk score sopra la soglia.
    Default soglia: 70 (livello arancio+rosso).
    Tool name: get_high_risk_vehicles
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

    results = []
    for r in rows:
        row = dict(r)
        if row.get('top_factors'):
            row['top_factors'] = json.loads(row['top_factors'])
        results.append(row)

    return {
        "threshold": threshold,
        "azienda_filter": azienda or "tutte",
        "count": len(results),
        "vehicles": results
    }
```

---

## Definizione tool per il function calling

```python
# agent/tool_definitions.py

TOOLS = [
    {
        "name": "get_vehicle_history",
        "description": "Recupera gli ultimi interventi di manutenzione per un veicolo specifico identificato dalla targa.",
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
        "name": "get_fleet_risk_summary",
        "description": "Restituisce il risk score attuale di tutta la flotta ordinato per rischio decrescente. Filtrabile per azienda del gruppo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "azienda": {
                    "type": "string",
                    "description": "Azienda da filtrare: G (Guido Bernardini), B (Bernardini), C (Cosmo). Ometti per vedere tutto il gruppo.",
                    "enum": ["G", "B", "C"]
                }
            }
        }
    },
    {
        "name": "get_vehicle_risk",
        "description": "Restituisce il risk score dettagliato per un singolo veicolo, inclusi i fattori che contribuiscono al rischio.",
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
        "description": "Restituisce tutti i veicoli con risk score sopra una soglia. Usa questa tool per la prioritizzazione degli interventi urgenti.",
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Soglia minima di risk score (0-100). Default: 70 (arancio+rosso).",
                    "default": 70
                },
                "azienda": {
                    "type": "string",
                    "description": "Filtra per azienda: G, B o C. Ometti per tutto il gruppo.",
                    "enum": ["G", "B", "C"]
                }
            }
        }
    }
]
```

---

## Struttura directory consigliata

```
progetto/
├── data/
│   └── maintenance.db          ← database SQLite
├── db/
│   ├── schema.sql              ← DDL delle tabelle
│   ├── init.py                 ← get_connection() e init_db()
│   └── ingest.py               ← import Excel → maintenance_history
├── agent/
│   ├── tools.py                ← funzioni Python chiamate dall'agente
│   ├── tool_definitions.py     ← schema JSON per il function calling
│   └── run.py                  ← loop principale dell'agente
├── pipeline/
│   └── features.py             ← calcolo vehicle_features (notturno)
└── scoring/
    └── predict.py              ← risk_scores dal modello ML
```

---

## Note implementative

**`SERIALE_DOC` non è univoco.** Non usarlo mai come chiave primaria. Più righe con lo stesso seriale rappresentano righe contabili diverse dello stesso intervento (es. manodopera + ricambi + IVA). Trattale come un gruppo, non come record separati.

**Deduplicazione all'ingresso.** La funzione `ingest_excel` evita reinserimenti basandosi su `(azienda, seriale_doc, targa, data_intervento)`. Se lo stesso Excel viene reimportato, non crea duplicati.

**`PRAGMA journal_mode=WAL`** permette letture concorrenti mentre la pipeline scrive. Importante se l'agente gira mentre il batch notturno è ancora in corso.

**`risk_scores` è append-only.** Non sovrascrivere mai i vecchi score: tieni lo storico per monitorare il drift del modello nel tempo. La query dell'agente recupera sempre l'ultimo score tramite `MAX(computed_at)`.

**Colonne km** già presenti in `vehicle_features` ma valorizzate a `NULL`. Quando procuri i dati di percorrenza, basta fare `UPDATE vehicle_features SET km_stimati_settimana = ...` senza toccare lo schema.
