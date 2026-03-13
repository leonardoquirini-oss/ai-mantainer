#!/bin/bash
#
# Retrain modelli ML + ricalcolo risk score flotta.
# Lancia in background: nohup ./retrain.sh &
#

set -e

cd "$(dirname "$0")"
source venv/bin/activate

DATE=$(date +%Y%m%d_%H%M)
TRAIN_LOG="logs/train_${DATE}.log"
SCORE_LOG="logs/score_${DATE}.log"

echo "=== Retrain avviato: $(date) ==="
echo "Train log: $TRAIN_LOG"
echo "Score log: $SCORE_LOG"

# 1. Training
echo "[1/2] Training modelli..."
python scripts/train_models.py > "$TRAIN_LOG" 2>&1
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERRORE: Training fallito (exit $TRAIN_EXIT). Vedi $TRAIN_LOG"
    exit 1
fi
echo "[1/2] Training completato."

# 2. Scoring flotta
echo "[2/2] Scoring flotta..."
python scripts/score_fleet.py > "$SCORE_LOG" 2>&1
SCORE_EXIT=$?

if [ $SCORE_EXIT -ne 0 ]; then
    echo "ERRORE: Scoring fallito (exit $SCORE_EXIT). Vedi $SCORE_LOG"
    exit 1
fi
echo "[2/2] Scoring completato."

echo "=== Retrain terminato: $(date) ==="
