# Integrazione Dati Viaggi nel DB SQLite
## Motrici + Semirimorchi con combinazioni variabili

**Contesto:**
- Ogni viaggio ha un ID univoco (`BG`), una targa motrice, una targa semirimorchio, km e data
- Motrice e semirimorchio cambiano combinazione tra un viaggio e l'altro
- I container NON hanno km — solo tempi (gestione separata, fuori scope qui)
- I km vanno tracciati **indipendentemente** per motrice e per semirimorchio

---

## Struttura dati in ingresso

| Campo | Tipo | Note |
|---|---|---|
| `BG` | TEXT | ID univoco del viaggio — chiave primaria |
| `TargaMotrice` | TEXT | Targa della motrice |
| `TargaSemirimorchio` | TEXT | Targa del semirimorchio |
| `Km` | REAL | Km percorsi nel viaggio |
| `Data` | DATE | Data del viaggio |

---

## Schema tabelle

### Tabella `trips` — un record per viaggio

```sql
CREATE TABLE IF NOT EXISTS trips (
    bg                  TEXT    PRIMARY KEY,  -- ID univoco dalla fonte
    targa_motrice       TEXT    NOT NULL,
    targa_semirimorchio TEXT,                 -- nullable: viaggio solo motrice
    km                  REAL    NOT NULL CHECK (km > 0),
    data_viaggio        DATE    NOT NULL,
    inserted_at         DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trips_motrice       ON trips(targa_motrice);
CREATE INDEX IF NOT EXISTS idx_trips_semirimorchio ON trips(targa_semirimorchio);
CREATE INDEX IF NOT EXISTS idx_trips_data          ON trips(data_viaggio);
```

### Vista `vehicle_km` — km per targa (motrice O semirimorchio)

Questa vista unifica i km visti dalla prospettiva di ogni singola targa, indipendentemente dal ruolo che ha avuto nel viaggio.

```sql
CREATE VIEW IF NOT EXISTS vehicle_km AS

    -- Km accumulati come MOTRICE
    SELECT
        bg,
        targa_motrice       AS targa,
        'motrice'           AS ruolo,
        km,
        data_viaggio
    FROM trips
    WHERE targa_motrice IS NOT NULL

    UNION ALL

    -- Km accumulati come SEMIRIMORCHIO
    SELECT
        bg,
        targa_semirimorchio AS targa,
        'semirimorchio'     AS ruolo,
        km,
        data_viaggio
    FROM trips
    WHERE targa_semirimorchio IS NOT NULL;
```

> Con questa vista, per sapere tutti i km di una targa (sia che fosse motrice sia semirimorchio) basta fare `SELECT * FROM vehicle_km WHERE targa = 'AB123CD'`.

---

## Import viaggi

```python
# db/ingest_trips.py
import pandas as pd
from db.init import get_connection

COLUMN_MAP = {
    'BG':                   'bg',
    'TargaMotrice':         'targa_motrice',
    'TargaSemirimorchio':   'targa_semirimorchio',
    'Km':                   'km',
    'DataViaggio':          'data_viaggio',
}

def ingest_trips(filepath: str):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    df = df.rename(columns=COLUMN_MAP)

    # Pulizia
    df['bg']          = df['bg'].astype(str).str.strip()
    df['targa_motrice'] = df['targa_motrice'].astype(str).str.strip().str.upper()
    df['targa_semirimorchio'] = df['targa_semirimorchio'].astype(str).str.strip().str.upper()
    df['data_viaggio'] = pd.to_datetime(df['data_viaggio'], errors='coerce').dt.date
    df['km']          = pd.to_numeric(df['km'], errors='coerce')

    # Normalizza: semirimorchio vuoto o 'NAN' → NULL
    df['targa_semirimorchio'] = df['targa_semirimorchio'].replace({'NAN': None, '': None})

    # Scarta righe invalide
    df = df.dropna(subset=['bg', 'targa_motrice', 'data_viaggio', 'km'])
    df = df[df['km'] > 0]

    conn = get_connection()
    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT INTO trips (bg, targa_motrice, targa_semirimorchio, km, data_viaggio)
                VALUES (?, ?, ?, ?, ?)
            """, (
                row['bg'],
                row['targa_motrice'],
                row.get('targa_semirimorchio'),
                row['km'],
                row['data_viaggio']
            ))
            inserted += 1
        except Exception:
            skipped += 1  # BG duplicato, già presente

    conn.commit()
    conn.close()
    print(f"Viaggi inseriti: {inserted} | Duplicati saltati: {skipped}")
```

---

## Feature km nella pipeline notturna

```python
# pipeline/features.py — sezione km
import pandas as pd
from db.init import get_connection

def compute_km_features(targa: str, reference_date=None) -> dict:
    """
    Calcola le feature di percorrenza per una targa (motrice o semirimorchio).
    Usa la vista vehicle_km che unifica i due ruoli.
    """
    conn = get_connection()
    ref = pd.Timestamp(reference_date or pd.Timestamp.today())

    # Tutti i viaggi di questa targa (qualunque ruolo)
    trips = pd.read_sql("""
        SELECT data_viaggio, km, ruolo
        FROM vehicle_km
        WHERE targa = ?
        ORDER BY data_viaggio
    """, conn, params=(targa,))

    # Km dal'ultimo intervento di manutenzione
    last_intervention = conn.execute("""
        SELECT MAX(data_intervento)
        FROM maintenance_history
        WHERE targa = ?
    """, (targa,)).fetchone()[0]

    conn.close()

    if trips.empty:
        return {
            'km_stimati_settimana':     None,
            'km_dal_ultimo_intervento': None,
            'km_totali_storici':        None,
            'km_ultimi_30d':            None,
            'km_ultimi_90d':            None,
        }

    trips['data_viaggio'] = pd.to_datetime(trips['data_viaggio'])

    cutoff_90 = ref - pd.Timedelta(days=90)
    cutoff_30 = ref - pd.Timedelta(days=30)

    recent_90 = trips[trips['data_viaggio'] >= cutoff_90]
    recent_30 = trips[trips['data_viaggio'] >= cutoff_30]

    # Km medi settimanali (basati sugli ultimi 90 giorni)
    km_week = (recent_90['km'].sum() / 90 * 7) if not recent_90.empty else None

    # Km dal'ultimo intervento
    km_dal_intervento = None
    if last_intervention:
        km_dal_intervento = trips[
            trips['data_viaggio'] >= pd.Timestamp(last_intervention)
        ]['km'].sum()

    return {
        'km_stimati_settimana':     round(km_week, 1) if km_week else None,
        'km_dal_ultimo_intervento': round(float(km_dal_intervento), 1) if km_dal_intervento else None,
        'km_totali_storici':        round(float(trips['km'].sum()), 1),
        'km_ultimi_30d':            round(float(recent_30['km'].sum()), 1),
        'km_ultimi_90d':            round(float(recent_90['km'].sum()), 1),
    }
```

---

## Tool per l'agente LLM

```python
# agent/tools.py — aggiungi queste funzioni

def get_vehicle_km_summary(targa: str) -> dict:
    """
    Riepilogo chilometrico di una targa (motrice o semirimorchio).
    Usa la vista vehicle_km — funziona per entrambi i ruoli.
    Tool name: get_vehicle_km_summary
    """
    conn = get_connection()
    targa = targa.upper()

    summary = conn.execute("""
        SELECT
            COUNT(*)            AS totale_viaggi,
            SUM(km)             AS km_totali,
            ROUND(AVG(km), 1)   AS km_medi_viaggio,
            MAX(km)             AS km_viaggio_massimo,
            MIN(data_viaggio)   AS primo_viaggio,
            MAX(data_viaggio)   AS ultimo_viaggio
        FROM vehicle_km
        WHERE targa = ?
    """, (targa,)).fetchone()

    km_30d = conn.execute("""
        SELECT SUM(km) FROM vehicle_km
        WHERE targa = ?
          AND data_viaggio >= DATE('now', '-30 days')
    """, (targa,)).fetchone()[0]

    last_intervention = conn.execute("""
        SELECT MAX(data_intervento) FROM maintenance_history
        WHERE targa = ?
    """, (targa,)).fetchone()[0]

    km_dal_intervento = None
    if last_intervention:
        km_dal_intervento = conn.execute("""
            SELECT SUM(km) FROM vehicle_km
            WHERE targa = ? AND data_viaggio >= ?
        """, (targa, last_intervention)).fetchone()[0]

    # Dettaglio ruoli (quante volte motrice vs semirimorchio)
    ruoli = conn.execute("""
        SELECT ruolo, COUNT(*) AS n, SUM(km) AS km
        FROM vehicle_km
        WHERE targa = ?
        GROUP BY ruolo
    """, (targa,)).fetchall()

    conn.close()

    return {
        "targa": targa,
        "totale_viaggi":            summary[0],
        "km_totali":                summary[1],
        "km_medi_per_viaggio":      summary[2],
        "km_viaggio_massimo":       summary[3],
        "primo_viaggio":            summary[4],
        "ultimo_viaggio":           summary[5],
        "km_ultimi_30d":            round(km_30d, 1) if km_30d else 0,
        "km_dal_ultimo_intervento": round(km_dal_intervento, 1) if km_dal_intervento else None,
        "dettaglio_ruoli":          [dict(r) for r in ruoli],
    }


def get_trip_history(targa: str, limit: int = 20) -> dict:
    """
    Ultimi N viaggi di una targa, con il ruolo che aveva (motrice o semirimorchio)
    e la controparte con cui viaggiava.
    Tool name: get_trip_history
    """
    conn = get_connection()
    targa = targa.upper()

    rows = conn.execute("""
        SELECT
            t.bg,
            t.data_viaggio,
            t.km,
            t.targa_motrice,
            t.targa_semirimorchio,
            CASE
                WHEN t.targa_motrice = ? THEN 'motrice'
                ELSE 'semirimorchio'
            END AS ruolo_in_viaggio,
            CASE
                WHEN t.targa_motrice = ? THEN t.targa_semirimorchio
                ELSE t.targa_motrice
            END AS controparte
        FROM trips t
        WHERE t.targa_motrice = ? OR t.targa_semirimorchio = ?
        ORDER BY t.data_viaggio DESC
        LIMIT ?
    """, (targa, targa, targa, targa, limit)).fetchall()

    conn.close()

    return {
        "targa": targa,
        "viaggi": [dict(r) for r in rows]
    }


def get_fleet_km_ranking(azienda: str = None) -> dict:
    """
    Classifica la flotta per km percorsi dall'ultimo intervento.
    Prioritizza i veicoli più usurati per utilizzo reale.
    Tool name: get_fleet_km_ranking
    """
    conn = get_connection()

    query = """
        SELECT
            vk.targa,
            mh.azienda,
            ROUND(SUM(vk.km), 0)        AS km_dal_intervento,
            MAX(mh.data_intervento)      AS ultimo_intervento,
            COUNT(vk.bg)                 AS viaggi_dal_intervento
        FROM vehicle_km vk
        LEFT JOIN maintenance_history mh ON vk.targa = mh.targa
        WHERE vk.data_viaggio >= COALESCE(
            (SELECT MAX(data_intervento)
             FROM maintenance_history
             WHERE targa = vk.targa),
            '1900-01-01'
        )
    """
    params = []
    if azienda:
        query += " AND mh.azienda = ?"
        params.append(azienda.upper())

    query += " GROUP BY vk.targa ORDER BY km_dal_intervento DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return {
        "azienda_filter": azienda or "tutte",
        "ranking": [dict(r) for r in rows]
    }
```

---

## Definizioni JSON per il function calling

```python
# Da aggiungere in agent/tool_definitions.py

{
    "name": "get_vehicle_km_summary",
    "description": (
        "Restituisce il riepilogo chilometrico di una targa (motrice o semirimorchio). "
        "Include km totali, km nell'ultimo mese, km percorsi dall'ultimo intervento "
        "e il dettaglio dei ruoli (quante volte motrice vs semirimorchio)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "targa": {"type": "string", "description": "Targa del veicolo"}
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
            "targa": {"type": "string", "description": "Targa del veicolo"},
            "limit": {"type": "integer", "description": "Numero viaggi da restituire (default: 20)", "default": 20}
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
            }
        }
    }
}
```

---

## Schema DB completo

```
maintenance_history   ← storico interventi (append-only)
trips                 ← viaggi con targa_motrice + targa_semirimorchio + km  ← NUOVO
  │
  └─► vehicle_km (VIEW) ← km per singola targa, qualunque ruolo             ← NUOVO
        │
        ▼
vehicle_features      ← feature calcolate (pipeline notturna)
        │
        ▼
risk_scores           ← output ML → agente LLM
```

---

## Note implementative

**La vista `vehicle_km` è la chiave.** Non duplicare la logica motrice/semirimorchio in ogni query — usa sempre la vista. Se in futuro aggiungi un terzo ruolo (es. dollies), basta aggiungere un `UNION ALL` nella vista.

**`targa_semirimorchio` è nullable.** Alcuni viaggi potrebbero avere solo la motrice (es. viaggio a vuoto senza rimorchio). Lo schema lo gestisce già.

**Combinazioni motrice+semirimorchio.** Con `get_trip_history` l'agente può vedere con quale semirimorchio viaggiava una motrice e viceversa — utile per correlare guasti a combinazioni specifiche (es. un semirimorchio pesante che usura di più le motrici con cui viaggia).

**Colonne km in `vehicle_features`.** Le colonne `km_stimati_settimana` e `km_dal_ultimo_intervento` erano già presenti nello schema come NULL. Ora `compute_km_features()` le popola automaticamente ogni notte tramite la vista `vehicle_km`.
