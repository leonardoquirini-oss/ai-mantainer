#!/usr/bin/env python
"""
Scoring notturno della flotta.

Calcola risk score per tutte le targhe e salva su DB.

Usage:
    python scripts/score_fleet.py [--refresh-data] [--no-save]

Tipicamente eseguito via cron:
    0 3 * * * cd /home/berni/maintainer && python scripts/score_fleet.py --refresh-data
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description='Calcola risk score per tutta la flotta'
    )
    parser.add_argument(
        '--refresh-data', '-r',
        action='store_true',
        help='Aggiorna dati da AdHoc prima dello scoring'
    )
    parser.add_argument(
        '--refresh-days',
        type=int,
        default=30,
        help='Giorni di dati da importare con --refresh-data (default: 30)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Non salvare su DB (solo calcolo)'
    )
    parser.add_argument(
        '--targa',
        type=str,
        default=None,
        help='Calcola solo per una targa specifica'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=50,
        help='Mostra solo score >= questo valore (default: 50)'
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

    logger = logging.getLogger('score_fleet')

    # Refresh dati se richiesto
    if args.refresh_data:
        logger.info(f"Aggiornamento dati da AdHoc (ultimi {args.refresh_days} giorni)...")
        try:
            from db.ingest import ingest_from_adhoc

            data_start = date.today() - timedelta(days=args.refresh_days)
            data_end = date.today()

            logger.info(f"Periodo: {data_start} → {data_end}")
            n_rows = ingest_from_adhoc(data_start, data_end)
            logger.info(f"Importate {n_rows} nuove righe da AdHoc")
        except Exception as e:
            logger.error(f"Errore refresh dati: {e}")
            logger.warning("Continuo con dati esistenti...")

    # Import dopo setup path
    from scoring.predict import (
        score_fleet, load_models, compute_all_risk_scores,
        get_high_risk_vehicles
    )

    if args.targa:
        # Scoring singola targa
        logger.info(f"Calcolo risk score per targa: {args.targa}")

        models = load_models()
        if not models:
            logger.error("Nessun modello caricato. Eseguire prima train_models.py")
            sys.exit(1)

        scores = compute_all_risk_scores(args.targa, models)

        print(f"\nRisk scores per {args.targa}:")
        print("-" * 50)
        for s in scores:
            if s.get('risk_score') is not None:
                level_emoji = {
                    'rosso': '🔴',
                    'arancio': '🟠',
                    'giallo': '🟡',
                    'verde': '🟢',
                }.get(s['risk_level'], '⚪')

                print(f"{level_emoji} {s['tipo_guasto']:15} {s['risk_score']:5.1f} ({s['risk_level']})")

                if s.get('top_factors'):
                    for f in s['top_factors'][:2]:
                        print(f"   ↳ {f['impact']}")

    else:
        # Scoring completo flotta
        logger.info("Avvio scoring flotta completa...")

        n_scores = score_fleet(save_to_db=not args.no_save)

        if n_scores == 0:
            logger.error("Nessun score calcolato. Verificare modelli e dati.")
            sys.exit(1)

        logger.info(f"Calcolati {n_scores} score")

        # Mostra veicoli ad alto rischio
        if not args.no_save:
            high_risk = get_high_risk_vehicles(min_score=args.min_score, limit=20)

            if high_risk:
                print(f"\n{'='*60}")
                print(f"VEICOLI AD ALTO RISCHIO (score >= {args.min_score})")
                print(f"{'='*60}\n")

                for v in high_risk:
                    level_emoji = {
                        'rosso': '🔴',
                        'arancio': '🟠',
                        'giallo': '🟡',
                    }.get(v['risk_level'], '⚪')

                    print(f"{level_emoji} {v['targa']:12} {v['tipo_guasto']:15} {v['risk_score']:5.1f}")
            else:
                print(f"\nNessun veicolo con score >= {args.min_score}")


if __name__ == '__main__':
    main()
