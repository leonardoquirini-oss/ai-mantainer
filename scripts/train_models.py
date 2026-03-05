#!/usr/bin/env python
"""
Addestra tutti i modelli ML per scoring del rischio.

Usage:
    python scripts/train_models.py [--refresh-data] [--optimize] [--horizons 7,30,90]

Esempio retraining completo:
    python scripts/train_models.py --refresh-data
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description='Addestra modelli ML per scoring rischio'
    )
    parser.add_argument(
        '--refresh-data', '-r',
        action='store_true',
        help='Aggiorna dati da AdHoc prima del training'
    )
    parser.add_argument(
        '--data-start',
        type=str,
        default='2018-01-01',
        help='Data inizio per refresh dati (default: 2018-01-01)'
    )
    parser.add_argument(
        '--optimize', '-o',
        action='store_true',
        help='Ottimizza iperparametri con Optuna (più lento)'
    )
    parser.add_argument(
        '--horizons',
        type=str,
        default='7,30,90',
        help='Orizzonti temporali (default: 7,30,90)'
    )
    parser.add_argument(
        '--tipo-guasto',
        type=str,
        default=None,
        help='Addestra solo per un tipo_guasto specifico'
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

    logger = logging.getLogger('train_models')
    logger.info("Avvio training modelli ML")

    # Refresh dati se richiesto
    if args.refresh_data:
        logger.info("Aggiornamento dati da AdHoc...")
        try:
            from db.ingest import ingest_from_adhoc
            from datetime import datetime

            data_start = datetime.strptime(args.data_start, '%Y-%m-%d').date()
            data_end = date.today()

            logger.info(f"Periodo: {data_start} → {data_end}")
            n_rows = ingest_from_adhoc(data_start, data_end)
            logger.info(f"Importate {n_rows} righe da AdHoc")
        except Exception as e:
            logger.error(f"Errore refresh dati: {e}")
            logger.warning("Continuo con dati esistenti...")

    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(',')]
    logger.info(f"Orizzonti: {horizons}")

    # Import dopo setup path
    from scoring.train import train_all_models, train_and_save_model
    from scoring.target import TIPI_GUASTO

    if args.tipo_guasto:
        # Training singolo tipo_guasto
        if args.tipo_guasto not in TIPI_GUASTO:
            logger.error(f"tipo_guasto '{args.tipo_guasto}' non valido. Validi: {TIPI_GUASTO}")
            sys.exit(1)

        logger.info(f"Training solo per: {args.tipo_guasto}")
        for horizon in horizons:
            train_and_save_model(args.tipo_guasto, horizon, args.optimize)
    else:
        # Training completo
        results = train_all_models(horizons=horizons, optimize=args.optimize)

        # Statistiche finali
        success = sum(1 for r in results.values() if r.get('pr_auc'))
        failed = sum(1 for r in results.values() if r.get('status') in ('skipped', 'error'))

        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING COMPLETATO")
        logger.info(f"Modelli addestrati: {success}")
        logger.info(f"Modelli falliti/skipped: {failed}")
        logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
