# Maintainer - Guida all'uso

Sistema di manutenzione predittiva per flotte di autotrasporti.

## Quick Start

```bash
cd /home/berni/maintainer
source venv/bin/activate

# 1. Training dei modelli (prima volta o retraining trimestrale)
python scripts/train_models.py --refresh-data

# 2. Scoring della flotta
python scripts/score_fleet.py --refresh-data

# 3. Verifica veicoli ad alto rischio
python scripts/score_fleet.py --targa FE065SW
```

---

## Scripts CLI

### `scripts/train_models.py` - Training modelli ML

Addestra i modelli XGBoost per predire il rischio di guasto.

```bash
# Training completo con refresh dati da AdHoc
python scripts/train_models.py --refresh-data

# Training solo orizzonte 30 giorni
python scripts/train_models.py --horizons 30

# Training con ottimizzazione Optuna (più lento)
python scripts/train_models.py --refresh-data --optimize

# Training singolo tipo guasto
python scripts/train_models.py --tipo-guasto freni
```

**Opzioni:**

| Flag | Descrizione |
|------|-------------|
| `--refresh-data`, `-r` | Importa dati da AdHoc prima del training |
| `--data-start` | Data inizio import (default: 2018-01-01) |
| `--horizons` | Orizzonti temporali (default: 7,30,90) |
| `--tipo-guasto` | Addestra solo un tipo specifico |
| `--optimize`, `-o` | Ottimizza iperparametri con Optuna |
| `--verbose`, `-v` | Output dettagliato |

**Output:**
- Modelli salvati in `models/risk_{horizon}d_{tipo_guasto}.joblib`
- Metriche in `models/risk_{horizon}d_{tipo_guasto}_metrics.json`
- Riepilogo in `models/training_summary.json`

---

### `scripts/score_fleet.py` - Scoring flotta

Calcola il risk score (0-100) per ogni coppia (targa, tipo_guasto).

```bash
# Scoring con refresh dati (uso quotidiano, solo mezzi attivi)
python scripts/score_fleet.py --refresh-data

# Refresh ultimi 7 giorni invece di 30
python scripts/score_fleet.py --refresh-data --refresh-days 7

# Scoring singola targa
python scripts/score_fleet.py --targa FE065SW

# Solo calcolo, senza salvare su DB
python scripts/score_fleet.py --no-save

# Mostra veicoli con score >= 70
python scripts/score_fleet.py --min-score 70

# Valuta TUTTI i mezzi (anche dismessi)
python scripts/score_fleet.py --all-vehicles
```

**Opzioni:**

| Flag | Descrizione |
|------|-------------|
| `--refresh-data`, `-r` | Importa dati recenti da AdHoc prima dello scoring |
| `--refresh-days` | Giorni da importare (default: 30) |
| `--targa` | Calcola solo per una targa specifica |
| `--min-score` | Soglia per mostrare veicoli ad alto rischio (default: 50) |
| `--no-save` | Non salvare su DB |
| `--all-vehicles` | Valuta tutti i mezzi, non solo quelli attivi |
| `--verbose`, `-v` | Output dettagliato |

**Output:**
- Salva in tabella `risk_scores` in SQLite
- Mostra veicoli ad alto rischio a fine esecuzione

---

### `scripts/check_drift.py` - Monitoring drift

Verifica se le performance dei modelli degradano nel tempo.

```bash
# Check standard (ultimi 90 giorni, soglia 10%)
python scripts/check_drift.py

# Check con parametri custom
python scripts/check_drift.py --lookback 60 --threshold 0.15

# Invia alert se drift rilevato
python scripts/check_drift.py --alert
```

**Opzioni:**

| Flag | Descrizione |
|------|-------------|
| `--lookback` | Giorni di dati recenti da usare (default: 90) |
| `--threshold` | Soglia di degradazione (default: 0.10 = 10%) |
| `--alert` | Invia alert se drift rilevato |
| `--verbose`, `-v` | Output dettagliato |

---

## Cron Jobs

Setup consigliato per automazione:

```bash
# Scoring notturno (ogni notte alle 3)
0 3 * * * cd /home/berni/maintainer && source venv/bin/activate && python scripts/score_fleet.py --refresh-data >> logs/scoring.log 2>&1

# Drift check mensile (1° del mese alle 9)
0 9 1 * * cd /home/berni/maintainer && source venv/bin/activate && python scripts/check_drift.py --alert >> logs/drift.log 2>&1

# Retraining trimestrale (1° gen/apr/lug/ott alle 2)
0 2 1 1,4,7,10 * cd /home/berni/maintainer && source venv/bin/activate && python scripts/train_models.py --refresh-data >> logs/training.log 2>&1
```

Crea la directory logs:
```bash
mkdir -p /home/berni/maintainer/logs
```

---

## Risk Score

Il risk score combina le probabilità dei tre orizzonti temporali:

| Orizzonte | Peso | Significato |
|-----------|------|-------------|
| 7 giorni | 50% | Urgente |
| 30 giorni | 35% | Pianificabile |
| 90 giorni | 15% | Strategico |

**Livelli di rischio:**

| Score | Livello | Azione |
|-------|---------|--------|
| >= 75 | 🔴 Rosso | Intervento urgente |
| >= 50 | 🟠 Arancio | Pianificare manutenzione |
| >= 25 | 🟡 Giallo | Monitorare |
| < 25 | 🟢 Verde | OK |

---

## Tipi di Guasto

I modelli vengono addestrati per 10 categorie:

1. `freni`
2. `pneumatici`
3. `motore`
4. `elettrico`
5. `carrozzeria`
6. `sospensioni`
7. `idraulico`
8. `revisione`
9. `tagliando`
10. `altro`

---

## Struttura Directory

```
maintainer/
├── scripts/              # CLI entry points
│   ├── train_models.py
│   ├── score_fleet.py
│   └── check_drift.py
├── scoring/              # Modulo ML
│   ├── features.py       # Feature engineering
│   ├── target.py         # Target construction
│   ├── train.py          # Training pipeline
│   ├── predict.py        # Scoring
│   ├── shap_explain.py   # Explainability
│   └── drift.py          # Drift monitoring
├── models/               # Modelli salvati (.joblib)
├── data/                 # Database SQLite
├── logs/                 # Log files
└── agent/                # Agent LLM e connettori
```

---

## Troubleshooting

### "Nessun modello caricato"
Esegui prima il training:
```bash
python scripts/train_models.py --refresh-data
```

### "0 record validi" durante import
Verifica la connessione ad AdHoc:
```bash
python -c "from agent.connectors import AdHocConnector; c = AdHocConnector(); print(c.health_check())"
```

### Training bloccato senza output
Il training mostra progresso ogni 100 alberi. Se non vedi output, verifica:
```bash
tail -f logs/training.log
```

### Eseguire in background
```bash
mkdir -p logs
nohup python scripts/train_models.py --refresh-data > logs/training.log 2>&1 &
echo "PID: $!"

# Segui il log
tail -f logs/training.log
```
