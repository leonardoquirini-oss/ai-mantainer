#!/usr/bin/env python
"""
Monitoring mensile del model drift.

Verifica se le performance dei modelli degradano nel tempo.

Usage:
    python scripts/check_drift.py [--lookback 90] [--threshold 0.10]

Tipicamente eseguito via cron mensile:
    0 9 1 * * cd /home/berni/maintainer && python scripts/check_drift.py
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description='Verifica drift dei modelli ML'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=90,
        help='Giorni di lookback per dati recenti (default: 90)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.10,
        help='Soglia di degradazione (default: 0.10 = 10%%)'
    )
    parser.add_argument(
        '--alert',
        action='store_true',
        help='Invia alert se drift rilevato'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Output verboso'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger('check_drift')
    logger.info(f"Verifica drift (lookback={args.lookback}d, threshold={args.threshold*100:.0f}%)")

    # Import dopo setup path
    from scoring.drift import check_all_models_drift, send_drift_alert, get_drift_history

    # Esegui check
    results = check_all_models_drift(
        lookback_days=args.lookback,
        threshold=args.threshold
    )

    if not results:
        logger.error("Nessun risultato. Verificare modelli e dati.")
        sys.exit(1)

    # Riepilogo
    drifted = [k for k, v in results.items() if v.get('drift_detected')]
    ok = [k for k, v in results.items() if not v.get('drift_detected') and 'current_pr_auc' in v]
    skipped = [k for k, v in results.items() if 'reason' in v]

    print(f"\n{'='*60}")
    print("RIEPILOGO DRIFT CHECK")
    print(f"{'='*60}\n")

    print(f"Modelli verificati: {len(results)}")
    print(f"  ✅ OK: {len(ok)}")
    print(f"  ⚠️  Drift: {len(drifted)}")
    print(f"  ⏭️  Skipped: {len(skipped)}")

    if drifted:
        print(f"\n{'='*60}")
        print("⚠️  MODELLI CON DRIFT")
        print(f"{'='*60}\n")

        for model_name in drifted:
            r = results[model_name]
            degradation_pct = r.get('degradation', 0) * 100
            print(f"  {model_name}:")
            print(f"    PR-AUC: {r.get('baseline_pr_auc', 'N/A'):.3f} → {r.get('current_pr_auc', 'N/A'):.3f}")
            print(f"    Degradazione: {degradation_pct:.1f}%")
            print()

        if args.alert:
            send_drift_alert(drifted, results)
            print("Alert inviato.")

        # Exit code 1 se drift rilevato
        sys.exit(1)

    else:
        print("\n✅ Nessun drift significativo rilevato.")

    # Mostra storico recente
    history = get_drift_history(last_n=5)
    if len(history) > 1:
        print(f"\n{'='*60}")
        print("STORICO DRIFT CHECK")
        print(f"{'='*60}\n")

        for entry in history[-5:]:
            timestamp = entry.get('timestamp', 'N/A')[:19]
            summary = entry.get('summary', {})
            total = summary.get('total_models', 0)
            drifted_count = summary.get('drifted', 0)

            status = "⚠️" if drifted_count > 0 else "✅"
            print(f"  {timestamp}  {status} {drifted_count}/{total} modelli con drift")


if __name__ == '__main__':
    main()
