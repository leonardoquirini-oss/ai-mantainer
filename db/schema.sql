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

-- ============================================================
-- TABELLA 4: viaggi con km (motrice + semirimorchio)
-- ============================================================
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

-- ============================================================
-- VISTA: km per targa (motrice O semirimorchio)
-- Unifica i km visti dalla prospettiva di ogni singola targa
-- ============================================================
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
