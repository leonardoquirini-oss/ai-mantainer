# Maintainer - Agente Manutenzione Predittiva

Sistema di manutenzione predittiva per flotte di autotrasporti basato su modelli statistici.

## Obiettivo

Passare da un approccio **reattivo** (intervenire dopo il guasto) a un approccio **predittivo/preventivo**, riducendo le manutenzioni straordinarie attraverso un piano di manutenzione ordinaria basato su evidenze statistiche.

## Modelli Statistici Implementati

### 1. Survival Analysis (Kaplan-Meier)
Stima la funzione di sopravvivenza S(t) = P(T > t), ovvero la probabilità che un mezzo non abbia subito un certo tipo di guasto entro t mesi.
- Segmenta per tipo_mezzo e tipo_guasto
- Gestisce dati censurati (mezzi senza guasto nel periodo di osservazione)
- Calcola mediana di sopravvivenza

### 2. Cox Proportional Hazards
Quantifica l'effetto del tipo_mezzo sul rischio relativo di guasto (hazard ratio).
- HR > 1 = rischio maggiore rispetto al riferimento
- HR < 1 = rischio minore
- Valuta significatività statistica (p-value < 0.05)

### 3. Weibull Analysis
Classifica il pattern di guasto in base al parametro beta:
- **beta < 0.8**: Guasti INFANTILI → problemi qualità, manutenzione preventiva NON efficace
- **0.8 <= beta <= 1.2**: Guasti CASUALI → tasso costante, manutenzione preventiva poco efficace
- **beta > 1.2**: Guasti da USURA → manutenzione preventiva EFFICACE

Per usura, calcola intervallo ottimale: `t = eta * (-ln(R_target))^(1/beta)`

### 4. NHPP (Non-Homogeneous Poisson Process)
Per mezzi con almeno 3 guasti, modella il tasso di guasto variabile:
- **beta > 1.15**: DETERIORAMENTO → guasti sempre più frequenti
- **0.85 <= beta <= 1.15**: STABILE
- **beta < 0.85**: MIGLIORAMENTO → guasti sempre meno frequenti

## Struttura Progetto

```
maintainer/
├── run.py                    # Entry point CLI
├── agent/
│   ├── __init__.py
│   ├── main.py               # CLI (typer)
│   ├── models/               # Modelli dati
│   │   ├── evento_manutenzione.py
│   │   └── analisi.py
│   ├── maintainer/           # Logica core
│   │   ├── optimizer.py      # Modelli statistici
│   │   ├── history_learner.py # Caricamento dati
│   │   └── maintenance_config.py
│   └── tools/                # Tools per LLM
│       └── maintenance_tools.py
├── config/
│   └── maintenance.yaml
├── data/                     # Cache dataset
└── requirements.txt
```

## Dati Richiesti

Il dataset deve contenere:
- **mezzo_id**: identificativo univoco del mezzo
- **tipo_mezzo**: categoria (semirimorchio, trattore, container, ecc.)
- **tipo_guasto**: classificazione dell'evento di manutenzione
- **data_evento**: data in cui si è verificato il guasto/intervento
- **data_acquisto/immatricolazione**: per calcolare l'età del mezzo

**Vincoli**: NON sono richiesti km percorsi né ore motore. L'unico asse temporale è l'età del mezzo (mesi).

## Comandi CLI

```bash
# Attiva virtual environment
source venv/bin/activate

# Carica dati
python run.py carica storico_manutenzioni.csv

# Genera piano manutenzione
python run.py piano

# Analisi specifiche
python run.py predizioni
python run.py statistiche

# Chat interattiva
python run.py chat
```

## Tools per LLM

| Tool | Descrizione |
|------|-------------|
| `carica_dati_csv` | Carica storico manutenzioni da CSV |
| `genera_piano_manutenzione` | Genera piano ordinaria basato su analisi |
| `analizza_weibull` | Classifica pattern guasto per tipo_mezzo x tipo_guasto |
| `analizza_sopravvivenza` | Curva Kaplan-Meier |
| `analizza_hazard_ratio` | Confronta rischio tra tipi mezzo (Cox PH) |
| `analizza_mezzo` | Analisi NHPP per singolo mezzo |
| `get_mezzi_critici` | Lista mezzi in deterioramento |
| `get_previsioni_guasti` | Previsioni guasti futuri |

## Output Piano Manutenzione

Il piano include per ogni combinazione tipo_mezzo x tipo_guasto:
- Classificazione (infantile/casuale/usura)
- Intervallo consigliato in mesi (se applicabile)
- Affidabilità target
- Motivazione statistica

**Importante**: Se Weibull indica guasti infantili o casuali, NON viene suggerita manutenzione preventiva periodica. Vengono invece suggerite azioni alternative (controllo qualità, disponibilità ricambi).

## Dipendenze

```
lifelines>=0.27.0   # Survival analysis
scipy>=1.11.0       # Distribuzioni statistiche
pandas>=2.0.0       # Data processing
numpy>=1.24.0       # Calcolo numerico
```

## Setup

```bash
cd /home/berni/maintainer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Formato CSV Storico

```csv
mezzo_id;tipo_mezzo;tipo_guasto;data_evento;data_immatricolazione
M001;semirimorchio;pneumatici;2024-01-15;2020-03-01
M001;semirimorchio;freni;2024-06-20;2020-03-01
M002;trattore;motore;2024-02-10;2019-07-15
```

---

## Note Sviluppo

- Struttura clonata da `planner` agent
- Interazione via chat come il planner
- I dati censurati (mezzi senza guasto) sono gestiti correttamente
- Se dati insufficienti (<5 record), l'analisi viene saltata con warning
