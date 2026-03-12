# ML Scoring — Guida Tecnica: Categorizzazione del Rischio

**Contesto:** Parco veicoli pesanti (autocarri + semirimorchi) · Storico manutenzioni 100k+ record  
**Aziende del gruppo:** G = Guido Bernardini, B = Bernardini, C = Cosmo  
**Colonne manutenzioni disponibili:** `azienda`, `seriale_doc`, `descrizione`, `dettaglio`, `targa`, `data_intervento`, `costo`, `data_imm`  
**Colonne viaggi disponibili:** `bg` (ID viaggio), `targa_motrice`, `targa_semirimorchio`, `km`, `data`  
**Classificazione interventi:** `categorizzatore.py` — 15 macro-categorie via regex, tag multipli per intervento, mapping verso 9 `TipoGuasto`  
**Obiettivo:** Produrre un risk score 0-100 per coppia **(targa, tipo_guasto)** da passare all'agente LLM

---

## 1. Diagnosi del problema attuale

### Situazione km — risolta

I km reali per viaggio sono ora disponibili tramite la tabella `trips` del DB SQLite. Ogni viaggio registra targa motrice, targa semirimorchio, km e data. I km vanno calcolati per singola targa tramite la vista `vehicle_km` che unifica i due ruoli (motrice e semirimorchio) in modo trasparente.

Questo sblocca le feature predittive più importanti: non si stima più il rischio solo in funzione del tempo, ma dell'**utilizzo reale** del veicolo.

### Cause comuni di scarsa accuratezza

| Causa | Sintomo | Soluzione |
|---|---|---|
| Target mal definito | Il modello non sa cosa predire esattamente | Definire target binario con orizzonte temporale |
| Class imbalance | Accuracy alta ma recall basso sui guasti | SMOTE + class_weight, ottimizzare F1 |
| Feature deboli | SHAP values bassi su tutte le feature | Feature engineering con km reali (vedi §2) |
| Granularità sbagliata | Predice per veicolo senza distinguere tipo intervento | Una riga = (targa, tipo_guasto, time_window) |
| Data leakage | Accuracy in train alta, in produzione crolla | Time-based split, mai random split |

---

## 2. Feature Engineering

Con lo storico manutenzioni + i dati viaggi reali, è possibile costruire feature predittive molto più ricche. Le feature km sono ora le più importanti.

### Feature da costruire (ordinate per importanza stimata)

| Feature | Fonte | Come calcolarla | Importanza | Note |
|---|---|---|---|---|
| `km_dal_ultimo_intervento` | `trips` + `maintenance_history` | SUM(km) da `vehicle_km` dopo la data dell'ultimo intervento | ⭐⭐⭐⭐⭐ | **Feature #1**: usura reale, non tempo |
| `km_ratio` | calcolata | `km_dal_ultimo_intervento / km_medi_per_intervento` | ⭐⭐⭐⭐⭐ | Quanto siamo oltre la soglia storica di usura |
| `days_since_last` | `maintenance_history` | oggi - data ultimo intervento per targa | ⭐⭐⭐⭐ | Proxy temporale, ora affiancato dai km |
| `days_ratio` | calcolata | `days_since_last / avg_days_between` | ⭐⭐⭐⭐ | Utile per veicoli con pochi viaggi registrati |
| `km_ultimi_30d` | `trips` | SUM(km) da `vehicle_km` ultimi 30 giorni | ⭐⭐⭐⭐ | Intensità utilizzo recente |
| `km_stimati_settimana` | `trips` | SUM(km ultimi 90gg) / 90 * 7 | ⭐⭐⭐⭐ | Ritmo di utilizzo corrente |
| `vehicle_age_days` | `maintenance_history` | oggi - `data_imm` | ⭐⭐⭐⭐ | Età correla con degrado generale |
| `interventions_last_90d` | `maintenance_history` | COUNT interventi su targa negli ultimi 90gg | ⭐⭐⭐⭐ | Alta frequenza = veicolo in degrado |
| `avg_days_between` | `maintenance_history` | MTBF storico per targa (giorni medi tra interventi) | ⭐⭐⭐ | Baseline comportamentale |
| `cost_trend` | `maintenance_history` | costo medio 3 mesi / media storica targa | ⭐⭐⭐ | Aumento costi = interventi più seri |
| `recurrence_12m` | `maintenance_history` | N interventi negli ultimi 12 mesi sulla stessa targa | ⭐⭐⭐ | Recidiva = veicolo strutturalmente a rischio |
| `component_failure_rate_fleet` | `maintenance_history` | % targhe flotta con intervento stesso tipo nell'ultimo anno | ⭐⭐ | Rischio sistemico |
| `month_sin` / `month_cos` | calcolata | `sin/cos(2π * month / 12)` | ⭐⭐ | Stagionalità (inverno = più guasti freni) |

> **Granularità del modello:** grazie a `categorizzatore.py`, ogni intervento viene classificato in una o più delle 15 macro-categorie (es. `01. PNEUMATICI`, `02. IMPIANTO FRENANTE`) e mappato in 9 `TipoGuasto` aggregati (`freni`, `pneumatici`, `motore`, ecc.). Il modello lavora a livello di **(targa, tipo_guasto)** — una riga di training per ogni combinazione. Un intervento con tag multipli contribuisce al training di più modelli contemporaneamente.

### Codice

```python
import pandas as pd
import numpy as np
import sqlite3

DB_PATH = "data/maintenance.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def build_features(reference_date=None):
    """
    Costruisce il dataset di feature per tutte le targhe attive.
    Legge da SQLite: maintenance_history e vehicle_km (vista).
    """
    conn = get_connection()
    ref = pd.Timestamp(reference_date or pd.Timestamp.today())

    # --- Storico manutenzioni ---
    maint = pd.read_sql("""
        SELECT targa, data_intervento, costo, data_imm
        FROM maintenance_history
        ORDER BY targa, data_intervento
    """, conn)
    maint['data_intervento'] = pd.to_datetime(maint['data_intervento'])
    maint['data_imm']        = pd.to_datetime(maint['data_imm'])

    # --- Km reali da vehicle_km (vista che unifica motrice + semirimorchio) ---
    trips = pd.read_sql("""
        SELECT targa, data_viaggio, km
        FROM vehicle_km
        ORDER BY targa, data_viaggio
    """, conn)
    trips['data_viaggio'] = pd.to_datetime(trips['data_viaggio'])

    conn.close()

    # Targhe attive = tutte quelle presenti nel DB
    all_targhe = pd.Series(
        pd.concat([maint['targa'], trips['targa']]).unique(),
        name='targa'
    )

    features = []
    for targa in all_targhe:
        m = maint[maint['targa'] == targa].sort_values('data_intervento')
        t = trips[trips['targa'] == targa].sort_values('data_viaggio')

        row = {'targa': targa}

        # --- Feature temporali da manutenzioni ---
        if not m.empty:
            last_date = m['data_intervento'].max()
            row['days_since_last']      = (ref - last_date).days
            row['avg_days_between']     = m['data_intervento'].diff().dt.days.mean()
            row['days_ratio']           = row['days_since_last'] / max(row['avg_days_between'] or 1, 1)
            row['vehicle_age_days']     = (ref - m['data_imm'].iloc[0]).days if pd.notna(m['data_imm'].iloc[0]) else None
            row['interventions_last_90d']  = int((m['data_intervento'] >= ref - pd.Timedelta(days=90)).sum())
            row['recurrence_12m']          = int((m['data_intervento'] >= ref - pd.Timedelta(days=365)).sum())

            # Trend costo
            cost_hist  = m['costo'].mean()
            cost_3m    = m[m['data_intervento'] >= ref - pd.Timedelta(days=90)]['costo'].mean()
            row['cost_trend'] = (cost_3m / cost_hist) if cost_hist else None

            # Km dal'ultimo intervento (dalla data dell'ultimo intervento)
            if not t.empty:
                km_post = t[t['data_viaggio'] >= last_date]['km'].sum()
                row['km_dal_ultimo_intervento'] = round(float(km_post), 1)

                # Km medi storici per intervallo tra manutenzioni
                # (stima del "budget km" storico per targa)
                if len(m) >= 2:
                    km_per_intervallo = []
                    for i in range(1, len(m)):
                        d_start = m['data_intervento'].iloc[i - 1]
                        d_end   = m['data_intervento'].iloc[i]
                        km_int  = t[(t['data_viaggio'] >= d_start) &
                                    (t['data_viaggio'] <  d_end)]['km'].sum()
                        km_per_intervallo.append(km_int)
                    avg_km_per_intervento = np.mean(km_per_intervallo) if km_per_intervallo else None
                    row['avg_km_per_intervento'] = avg_km_per_intervento
                    row['km_ratio'] = (km_post / avg_km_per_intervento) if avg_km_per_intervento else None
                else:
                    row['avg_km_per_intervento'] = None
                    row['km_ratio'] = None
            else:
                row['km_dal_ultimo_intervento'] = None
                row['avg_km_per_intervento']    = None
                row['km_ratio']                 = None
        else:
            row.update({
                'days_since_last': None, 'avg_days_between': None, 'days_ratio': None,
                'vehicle_age_days': None, 'interventions_last_90d': 0, 'recurrence_12m': 0,
                'cost_trend': None, 'km_dal_ultimo_intervento': None,
                'avg_km_per_intervento': None, 'km_ratio': None,
            })

        # --- Feature km da viaggi ---
        if not t.empty:
            cutoff_30 = ref - pd.Timedelta(days=30)
            cutoff_90 = ref - pd.Timedelta(days=90)
            recent_90 = t[t['data_viaggio'] >= cutoff_90]

            row['km_ultimi_30d']       = round(float(t[t['data_viaggio'] >= cutoff_30]['km'].sum()), 1)
            row['km_ultimi_90d']       = round(float(recent_90['km'].sum()), 1)
            row['km_stimati_settimana'] = round(float(recent_90['km'].sum()) / 90 * 7, 1) if not recent_90.empty else None
            row['km_totali_storici']   = round(float(t['km'].sum()), 1)
        else:
            row.update({
                'km_ultimi_30d': None, 'km_ultimi_90d': None,
                'km_stimati_settimana': None, 'km_totali_storici': None,
            })

        # --- Stagionalità ciclica ---
        row['month_sin'] = np.sin(2 * np.pi * ref.month / 12)
        row['month_cos'] = np.cos(2 * np.pi * ref.month / 12)

        features.append(row)

    return pd.DataFrame(features)
```

---

## 3. Definire il Target corretto

Il target deve essere **binario, specifico e con orizzonte temporale esplicito**.

**Formato:** `"questa targa avrà un intervento di tipo X entro N giorni?"`

Grazie a `categorizzatore.py`, il modello può ora lavorare a livello di **(targa, tipo_guasto)** — predice separatamente "i freni si guasteranno entro 30 giorni?" da "i pneumatici si guasteranno entro 30 giorni?". Questo è molto più utile operativamente rispetto a un target generico per targa.

### Struttura del dataset di training

Ogni riga del dataset rappresenta uno snapshot **(targa, tipo_guasto, data)** con le feature calcolate a quella data e il target che guarda N giorni in avanti.

Poiché `categorizza_riga()` ritorna tag multipli, un singolo intervento su `descrizione`/`dettaglio` che matcha sia `freni` che `pneumatici` genera **due righe** nel dataset — una per ogni tipo_guasto. Questo è corretto: entrambi i modelli devono imparare da quell'evento.

### Deduplicazione obbligatoria prima di costruire il target

`SERIALE_DOC` non è univoco: più righe con lo stesso seriale rappresentano lo stesso intervento (righe contabili distinte — manodopera, ricambi, IVA, ecc.). La deduplicazione raggruppa per `(azienda, seriale_doc, targa)`, consolida costo e poi applica il categorizzatore sul testo combinato.

### Costruzione del target

```python
import pandas as pd
import sqlite3
from categorizzatore import categorizza_riga, categoria_to_tipo_guasto

DB_PATH = "data/maintenance.db"

def load_classified_interventions():
    """
    Carica gli interventi deduplicati e classificati per tipo_guasto.
    Un intervento con tag multipli genera più righe (una per tipo_guasto).
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT
            azienda,
            seriale_doc,
            targa,
            MIN(data_intervento) AS data_intervento,
            SUM(costo)           AS costo_totale,
            -- Concatena descrizione e dettaglio per il categorizzatore
            GROUP_CONCAT(DISTINCT descrizione) AS descrizione,
            GROUP_CONCAT(DISTINCT dettaglio)   AS dettaglio
        FROM maintenance_history
        GROUP BY azienda, seriale_doc, targa
        ORDER BY targa, data_intervento
    """, conn)
    conn.close()

    df['data_intervento'] = pd.to_datetime(df['data_intervento'])

    # Applica il categorizzatore e mappa a tipo_guasto
    righe_espanse = []
    for _, row in df.iterrows():
        categorie = categorizza_riga(row['descrizione'], row['dettaglio'])
        tipi_guasto = set(categoria_to_tipo_guasto(c) for c in categorie)

        for tipo in tipi_guasto:
            righe_espanse.append({
                'azienda':          row['azienda'],
                'targa':            row['targa'],
                'data_intervento':  row['data_intervento'],
                'tipo_guasto':      tipo,
                'costo_totale':     row['costo_totale'],
            })

    return pd.DataFrame(righe_espanse)


def build_target(df, horizon_days):
    """
    Per ogni (targa, tipo_guasto) alla data X, il target è 1
    se esiste un intervento dello stesso tipo entro horizon_days giorni.
    """
    df = df.sort_values(['targa', 'tipo_guasto', 'data_intervento']).copy()

    df['next_intervention_date'] = df.groupby(
        ['targa', 'tipo_guasto']
    )['data_intervento'].shift(-1)

    df['days_to_next'] = (df['next_intervention_date'] - df['data_intervento']).dt.days
    df[f'fail_{horizon_days}d'] = (df['days_to_next'] <= horizon_days).astype(int)

    return df


# Pipeline completa
df = load_classified_interventions()
df = build_target(df, 7)    # urgente
df = build_target(df, 30)   # pianificabile
df = build_target(df, 90)   # strategico

# Escludi ultime righe per (targa, tipo_guasto) — futuro ignoto
df = df.dropna(subset=['next_intervention_date'])
```

### Esempio di output

```
targa     tipo_guasto  data_intervento  days_to_next  fail_7d  fail_30d  fail_90d
AB123CD   freni        2023-03-01       12            0        1         1
AB123CD   freni        2023-03-13       45            0        0         1
AB123CD   pneumatici   2023-01-10       80            0        0         1
AB123CD   pneumatici   2023-04-01       6             1        1         1
EF456GH   motore       2023-02-10       95            0        0         0
```

### I 9 modelli risultanti

Con 9 `TipoGuasto` × 3 orizzonti temporali hai **27 modelli** in totale. In pratica si addestrano i 9 modelli a 30 giorni come priorità, poi si estende agli altri orizzonti. I tipi con pochi dati storici (es. `idraulico`, `rotocella`) possono essere aggregati in `altro` se hanno meno di 500 esempi positivi nel training set.

### ⚠️ Time-based split — mai random split

Con dati temporali, il random split causa **data leakage**: il modello vede eventi futuri durante il training e le metriche risultanti sono falsamente ottimistiche.

#### Configurazione per storico 2003–2026

Lo storico disponibile copre 22 anni, ma i dati precedenti al 2017 rischiano di essere poco rappresentativi della flotta attuale (veicoli dismessi, pratiche di manutenzione cambiate). La finestra rilevante è quindi **2017–oggi**.

```
Storico disponibile:   2003 → mar 2026
Finestra rilevante:    gen 2017 → mar 2026  (≈ 9 anni)

Training:              gen 2017 → giu 2024  (≈ 7.5 anni)
Gap:                   30 giorni            (pari all'orizzonte target massimo)
Test finale:           ago 2024 → mar 2026  (≈ 19 mesi)
```

Walk-forward validation — 5 fold, ognuno con 6 mesi di test:

```
Fold 1:  gen 2017 → dic 2019  [gap 30gg]  feb 2020 → lug 2020
Fold 2:  gen 2017 → dic 2020  [gap 30gg]  feb 2021 → lug 2021
Fold 3:  gen 2017 → dic 2021  [gap 30gg]  feb 2022 → lug 2022
Fold 4:  gen 2017 → dic 2022  [gap 30gg]  feb 2023 → lug 2023
Fold 5:  gen 2017 → dic 2023  [gap 30gg]  feb 2024 → lug 2024
```

```python
import pandas as pd
from sklearn.metrics import average_precision_score

# Finestra rilevante: scarta dati troppo vecchi
HISTORY_START  = pd.Timestamp('2017-01-01')
TRAIN_CUTOFF   = pd.Timestamp('2024-06-01')
GAP_DAYS       = 30
TEST_START     = TRAIN_CUTOFF + pd.Timedelta(days=GAP_DAYS)

df = df[df['data_intervento'] >= HISTORY_START].sort_values('data_intervento')

# Opzione 1: split fisso — semplice e diretto
train = df[df['data_intervento'] <  TRAIN_CUTOFF]
test  = df[df['data_intervento'] >= TEST_START]

X_train, y_train = train[FEATURE_COLS], train['fail_30d']
X_test,  y_test  = test[FEATURE_COLS],  test['fail_30d']

# Opzione 2: walk-forward validation — più robusta, misura stabilità nel tempo
FOLDS = [
    ('2017-01-01', '2019-12-31', '2020-02-01', '2020-07-31'),
    ('2017-01-01', '2020-12-31', '2021-02-01', '2021-07-31'),
    ('2017-01-01', '2021-12-31', '2022-02-01', '2022-07-31'),
    ('2017-01-01', '2022-12-31', '2023-02-01', '2023-07-31'),
    ('2017-01-01', '2023-12-31', '2024-02-01', '2024-07-31'),
]

pr_auc_scores = []

for train_start, train_end, val_start, val_end in FOLDS:
    mask_train = (df['data_intervento'] >= train_start) & (df['data_intervento'] <= train_end)
    mask_val   = (df['data_intervento'] >= val_start)   & (df['data_intervento'] <= val_end)

    X_tr, y_tr = df[mask_train][FEATURE_COLS], df[mask_train]['fail_30d']
    X_vl, y_vl = df[mask_val][FEATURE_COLS],   df[mask_val]['fail_30d']

    model.fit(X_tr, y_tr)
    score = average_precision_score(y_vl, model.predict_proba(X_vl)[:, 1])
    pr_auc_scores.append(score)
    print(f"Fold {val_start[:4]}: PR-AUC = {score:.3f}")

print(f"\nMedia: {sum(pr_auc_scores)/len(pr_auc_scores):.3f}")
print(f"Stabilità (std): {pd.Series(pr_auc_scores).std():.3f}")
# Se std > 0.05 il modello è instabile nel tempo — investigare
```

> **Nota:** se i dati pre-2017 migliorano le metriche nei fold iniziali, puoi abbassare `HISTORY_START` a 2015 o 2013. Verifica empiricamente confrontando PR-AUC con e senza i dati più vecchi.

### Gestione class imbalance

Con 100k+ record i guasti reali saranno probabilmente il 5-15% del totale.

```python
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, average_precision_score

# Metodo 1: class_weight nel modello (inizia da qui)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',          # metrica corretta per dataset sbilanciati
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    early_stopping_rounds=30
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Metodo 2: SMOTE — solo sul training set, mai su validation/test
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Metriche corrette — NON usare accuracy come metrica principale
preds = model.predict(X_val)
print(classification_report(y_val, preds, target_names=['ok', 'manutenzione']))
print(f"PR-AUC: {average_precision_score(y_val, model.predict_proba(X_val)[:,1]):.3f}")
```

---

## 4. Scelta del modello

| Modello | Raccomandazione | Note |
|---|---|---|
| **XGBoost** | ✅ Consigliato | Gestisce NULL nativamente (importante: km mancanti per alcuni veicoli), feature importance integrata |
| **LightGBM** | ✅ Consigliato | Più veloce, ottimo con feature categoriali, gestisce NULL nativamente |
| Random Forest | Alternativa | Robusto ma non gestisce NULL — richiede imputation |
| Logistic Regression | Solo baseline | Non cattura interazioni non lineari |

### Strategia consigliata

Addestra **un modello separato per ogni orizzonte temporale** (7d, 30d, 90d). Combina i tre score nel risk score finale. Usa `Optuna` per ottimizzazione degli iperparametri.

```python
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'num_leaves':        trial.suggest_int('num_leaves', 20, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'scale_pos_weight':  scale_pos_weight,
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, proba)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
```

---

## 5. Risk Score 0-100

Il modello ML produce probabilità grezze (0.0–1.0). Vanno trasformate in uno score calibrato con soglie chiare.

### Soglie operative

| Score | Livello | Azione |
|---|---|---|
| 0–40 | 🟢 Verde | Nessuna azione richiesta |
| 41–70 | 🟡 Giallo | Monitoraggio rinforzato |
| 71–85 | 🟠 Arancio | Pianifica intervento |
| 86–100 | 🔴 Rosso | Intervento urgente |

### Codice

```python
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

# Calibra le probabilità per evitare score estremi non realistici
calibrated_model = CalibratedClassifierCV(xgb_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

def risk_level(score: float) -> str:
    if score <= 40:  return 'verde'
    if score <= 70:  return 'giallo'
    if score <= 85:  return 'arancio'
    return 'rosso'

def compute_risk_score(targa: str, models: dict, X_current) -> dict:
    """
    Combina predizioni a diversi orizzonti in un unico score 0-100.
    Peso maggiore all'orizzonte breve (urgenza).
    """
    prob_7d  = models['7d'].predict_proba(X_current)[0, 1]
    prob_30d = models['30d'].predict_proba(X_current)[0, 1]
    prob_90d = models['90d'].predict_proba(X_current)[0, 1]

    # Score pesato: urgenza conta di più
    weighted_prob = (
        0.50 * prob_7d  +
        0.35 * prob_30d +
        0.15 * prob_90d
    )

    # Scala a 0-100 con curva non lineare (accentua valori estremi)
    raw_score = weighted_prob * 100
    score = float(np.clip(raw_score ** 0.85 * (100 ** 0.15), 0, 100))

    return {
        'targa':          targa,
        'risk_score':     round(score, 1),
        'risk_level':     risk_level(score),
        'prob_fail_7d':   round(float(prob_7d), 3),
        'prob_fail_30d':  round(float(prob_30d), 3),
        'prob_fail_90d':  round(float(prob_90d), 3),
        'top_factors':    get_shap_factors(X_current),
    }
```

---

## 6. Spiegabilità con SHAP

Passare all'agente LLM solo il numero (es. 78/100) è poco utile. Con SHAP si include **perché** il rischio è alto — l'LLM genera diagnosi molto più precise e credibili.

```python
import shap

# Calcola explainer una sola volta (costoso), riusalo per tutti i veicoli
explainer = shap.TreeExplainer(xgb_model)

def get_shap_factors(X_row, top_n=3):
    """Ritorna le top N feature che aumentano il rischio."""
    shap_vals = explainer.shap_values(X_row)[0]
    feature_names = X_row.columns.tolist()

    factors = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    return [
        {
            'feature':   f,
            'impact':    round(float(v), 3),
            'direction': '+' if v > 0 else '-'
        }
        for f, v in factors if v > 0
    ]
```

### Esempio output JSON passato all'agente LLM

```json
{
  "targa": "AB123CD",
  "risk_score": 84.2,
  "risk_level": "arancio",
  "prob_fail_7d": 0.38,
  "prob_fail_30d": 0.71,
  "prob_fail_90d": 0.91,
  "top_factors": [
    {"feature": "km_ratio", "impact": 2.1, "direction": "+"},
    {"feature": "km_dal_ultimo_intervento", "impact": 1.4, "direction": "+"},
    {"feature": "recurrence_12m", "impact": 0.7, "direction": "+"}
  ]
}
```

Con `km_ratio: 2.1`, Claude capisce che il veicolo ha percorso il doppio dei km rispetto alla media storica tra un intervento e l'altro. Con `recurrence_12m: 5`, deduce che ha avuto 5 interventi nell'ultimo anno — frequenza anomala.

---

## 7. Pipeline di produzione

```
[Ogni notte 02:00]
    │
    ▼
db/ingest.py              → appende nuovi record da Excel manutenzioni
db/ingest_trips.py        → appende nuovi viaggi (BG, targa_motrice, targa_semirimorchio, km, data)
    │
    ▼
pipeline/features.py      → build_features() → scrive su vehicle_features in SQLite
    │
    ▼
scoring/predict.py        → compute_risk_score() per ogni targa → scrive su risk_scores in SQLite
    │
    ▼
agent/run.py              → agente LLM legge risk_scores, genera diagnosi e alert
```

### Re-training mensile con monitoraggio drift

```python
from sklearn.metrics import average_precision_score

def check_model_drift(model, X_recent, y_recent, baseline_pr_auc, threshold=0.10):
    """Alerta se le prestazioni scendono oltre la soglia."""
    current_pr_auc = average_precision_score(
        y_recent,
        model.predict_proba(X_recent)[:, 1]
    )
    drift = (baseline_pr_auc - current_pr_auc) / baseline_pr_auc

    if drift > threshold:
        send_alert(
            f"Model drift detected: PR-AUC {baseline_pr_auc:.3f} → {current_pr_auc:.3f} "
            f"(-{drift*100:.1f}%). Retraining needed."
        )
    return current_pr_auc
```

---

## 8. Dipendenze Python

```txt
# requirements.txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
imbalanced-learn>=0.11
shap>=0.44
optuna>=3.4
```

---

## 9. Checklist di miglioramento (priorità)

- [x] **[RISOLTO]** Dati km reali disponibili da tabella `trips` (BG, targa_motrice, targa_semirimorchio, km, data).
- [x] **[RISOLTO]** Vista `vehicle_km` unifica km per targa indipendentemente dal ruolo (motrice o semirimorchio).
- [x] **[RISOLTO]** Ridefinire il target con orizzonte temporale esplicito (7/30/90 giorni) — granularità (targa, tipo_guasto), deduplicazione per `SERIALE_DOC`, 27 modelli totali.
- [x] **[RISOLTO]** Implementare `km_dal_ultimo_intervento` e `km_ratio` — codice già presente in `build_features()`, dati disponibili in DB.
- [x] **[RISOLTO]** Sostituire random split con time-based split — finestra 2017→2024 training, ago 2024→oggi test, walk-forward 5 fold.
- [x] **[RISOLTO]** Classificare `descrizione`/`dettaglio` per scendere a livello di componente — `categorizzatore.py` già disponibile con 15 macro-categorie e mapping verso 9 `TipoGuasto`.
- [ ] **[ALTO]** Aggiungere `scale_pos_weight` e misurare F1/PR-AUC invece di accuracy — da applicare per ognuno dei 9 modelli per tipo_guasto.
- [ ] **[ALTO]** Implementare `days_ratio` e `recurrence_per_tipo` — `recurrence_12m` va calcolato per (targa, tipo_guasto), non solo per targa.
- [ ] **[MEDIO]** Aggiungere SHAP e passare `top_factors` all'agente LLM.
- [ ] **[MEDIO]** Calibrare le probabilità con `CalibratedClassifierCV`.
- [ ] **[MEDIO]** Verificare tipi_guasto con pochi esempi positivi (es. `idraulico`, `rotocella`) — aggregare in `altro` se < 500 esempi nel training set.
- [ ] **[MEDIO]** Pianificare re-training mensile con monitoring del drift.
- [ ] **[BASSO]** Ottimizzare iperparametri con Optuna.
