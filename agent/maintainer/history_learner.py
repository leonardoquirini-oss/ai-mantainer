"""
Modulo per caricamento e gestione dati storici manutenzione.

Questo modulo gestisce il caricamento dei dati da varie fonti
(CSV, database, API) e li converte in formato utilizzabile
dall'optimizer.
"""

import csv
import json
from pathlib import Path
from datetime import date, datetime
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger("maintenance-agent.data_loader")

from ..models import (
    EventoManutenzione,
    Mezzo,
    TipoMezzo,
    TipoGuasto,
    DatasetManutenzione,
)


class MaintenanceDataLoader:
    """
    Caricatore dati per il sistema di manutenzione predittiva.

    Supporta:
    - Caricamento da CSV
    - Caricamento da JSON
    - Caricamento da database (da implementare)

    Formato dati atteso:
    - mezzo_id: identificativo univoco
    - tipo_mezzo: semirimorchio, trattore, container, ecc.
    - tipo_guasto: classificazione evento
    - data_evento: data del guasto/intervento
    - data_acquisto: data acquisto mezzo
    - data_immatricolazione: data immatricolazione (preferita)
    """

    def __init__(self):
        self.cache_path = Path(__file__).parent.parent.parent / "data"
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def carica_da_csv(
        self,
        filepath: str,
        delimiter: str = ";",
        encoding: str = "utf-8"
    ) -> DatasetManutenzione:
        """
        Carica dati da file CSV.

        Colonne attese:
        - mezzo_id
        - tipo_mezzo
        - tipo_guasto
        - data_evento (YYYY-MM-DD)
        - data_acquisto (opzionale)
        - data_immatricolazione (opzionale)
        - descrizione (opzionale)
        - costo (opzionale)

        Args:
            filepath: Path al file CSV
            delimiter: Separatore colonne
            encoding: Encoding file

        Returns:
            DatasetManutenzione popolato
        """
        eventi = []
        mezzi_dict = {}  # mezzo_id -> Mezzo

        csv_path = Path(filepath)
        if not csv_path.exists():
            raise FileNotFoundError(f"File non trovato: {filepath}")

        logger.info(f"Caricamento dati da {filepath}")

        with open(csv_path, "r", encoding=encoding) as f:
            # Auto-detect delimiter
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                delimiter = dialect.delimiter
            except csv.Error:
                pass

            reader = csv.DictReader(f, delimiter=delimiter)

            for row in reader:
                try:
                    # Parse campi base
                    mezzo_id = row.get("mezzo_id", "").strip()
                    if not mezzo_id:
                        continue

                    tipo_mezzo = self._parse_tipo_mezzo(row.get("tipo_mezzo", ""))
                    tipo_guasto = self._parse_tipo_guasto(row.get("tipo_guasto", ""))
                    data_evento = self._parse_data(row.get("data_evento", ""))

                    if not data_evento:
                        continue

                    # Parse campi opzionali
                    data_acquisto = self._parse_data(row.get("data_acquisto", ""))
                    data_immatricolazione = self._parse_data(row.get("data_immatricolazione", ""))
                    descrizione = row.get("descrizione", "")
                    costo = self._parse_float(row.get("costo", "0"))
                    straordinario = row.get("straordinario", "1").lower() in ["1", "true", "si", "yes"]

                    # Crea evento
                    evento = EventoManutenzione(
                        mezzo_id=mezzo_id,
                        tipo_mezzo=tipo_mezzo,
                        tipo_guasto=tipo_guasto,
                        data_evento=data_evento,
                        data_acquisto=data_acquisto,
                        data_immatricolazione=data_immatricolazione,
                        descrizione=descrizione,
                        costo=costo,
                        straordinario=straordinario
                    )
                    eventi.append(evento)

                    # Traccia mezzo
                    if mezzo_id not in mezzi_dict:
                        mezzi_dict[mezzo_id] = Mezzo(
                            mezzo_id=mezzo_id,
                            tipo_mezzo=tipo_mezzo,
                            data_acquisto=data_acquisto,
                            data_immatricolazione=data_immatricolazione
                        )
                    else:
                        # Aggiorna date se più recenti
                        mezzo = mezzi_dict[mezzo_id]
                        if data_immatricolazione and (not mezzo.data_immatricolazione or data_immatricolazione > mezzo.data_immatricolazione):
                            mezzo.data_immatricolazione = data_immatricolazione
                        if data_acquisto and (not mezzo.data_acquisto or data_acquisto > mezzo.data_acquisto):
                            mezzo.data_acquisto = data_acquisto

                except Exception as e:
                    logger.warning(f"Errore parsing riga: {e}")
                    continue

        logger.info(f"Caricati {len(eventi)} eventi per {len(mezzi_dict)} mezzi")

        return DatasetManutenzione(
            eventi=eventi,
            mezzi=list(mezzi_dict.values()),
            data_osservazione_fine=date.today()
        )

    def carica_da_json(self, filepath: str) -> DatasetManutenzione:
        """
        Carica dati da file JSON.

        Formato atteso:
        {
            "eventi": [...],
            "mezzi": [...],
            "data_osservazione_fine": "YYYY-MM-DD"
        }
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        eventi = []
        for e in data.get("eventi", []):
            eventi.append(EventoManutenzione(
                mezzo_id=e["mezzo_id"],
                tipo_mezzo=self._parse_tipo_mezzo(e["tipo_mezzo"]),
                tipo_guasto=self._parse_tipo_guasto(e["tipo_guasto"]),
                data_evento=self._parse_data(e["data_evento"]),
                data_acquisto=self._parse_data(e.get("data_acquisto")),
                data_immatricolazione=self._parse_data(e.get("data_immatricolazione")),
                descrizione=e.get("descrizione", ""),
                costo=e.get("costo", 0),
                straordinario=e.get("straordinario", True)
            ))

        mezzi = []
        for m in data.get("mezzi", []):
            mezzi.append(Mezzo(
                mezzo_id=m["mezzo_id"],
                tipo_mezzo=self._parse_tipo_mezzo(m["tipo_mezzo"]),
                data_acquisto=self._parse_data(m.get("data_acquisto")),
                data_immatricolazione=self._parse_data(m.get("data_immatricolazione")),
                targa=m.get("targa", ""),
                marca=m.get("marca", ""),
                modello=m.get("modello", "")
            ))

        data_fine = self._parse_data(data.get("data_osservazione_fine")) or date.today()

        return DatasetManutenzione(
            eventi=eventi,
            mezzi=mezzi,
            data_osservazione_fine=data_fine
        )

    def salva_cache(self, dataset: DatasetManutenzione, nome: str = "dataset"):
        """Salva dataset in cache locale"""
        cache_file = self.cache_path / f"{nome}.json"

        data = {
            "eventi": [e.to_dict() for e in dataset.eventi],
            "mezzi": [
                {
                    "mezzo_id": m.mezzo_id,
                    "tipo_mezzo": m.tipo_mezzo.value if hasattr(m.tipo_mezzo, 'value') else m.tipo_mezzo,
                    "data_acquisto": m.data_acquisto.isoformat() if m.data_acquisto else None,
                    "data_immatricolazione": m.data_immatricolazione.isoformat() if m.data_immatricolazione else None,
                    "targa": m.targa,
                }
                for m in dataset.mezzi
            ],
            "data_osservazione_fine": dataset.data_osservazione_fine.isoformat()
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Dataset salvato in cache: {cache_file}")

    def carica_cache(self, nome: str = "dataset") -> Optional[DatasetManutenzione]:
        """Carica dataset da cache locale"""
        cache_file = self.cache_path / f"{nome}.json"
        if cache_file.exists():
            return self.carica_da_json(str(cache_file))
        return None

    def _parse_tipo_mezzo(self, valore: str) -> TipoMezzo:
        """Converte stringa in TipoMezzo"""
        valore = (valore or "").lower().strip()
        mapping = {
            "semirimorchio": TipoMezzo.SEMIRIMORCHIO,
            "semi": TipoMezzo.SEMIRIMORCHIO,
            "trailer": TipoMezzo.SEMIRIMORCHIO,
            "trattore": TipoMezzo.TRATTORE,
            "motrice": TipoMezzo.TRATTORE,
            "truck": TipoMezzo.TRATTORE,
            "container": TipoMezzo.CONTAINER,
            "furgone": TipoMezzo.FURGONE,
            "van": TipoMezzo.FURGONE,
        }
        return mapping.get(valore, TipoMezzo.ALTRO)

    def _parse_tipo_guasto(self, valore: str) -> TipoGuasto:
        """Converte stringa in TipoGuasto"""
        valore = (valore or "").lower().strip()
        mapping = {
            "pneumatici": TipoGuasto.PNEUMATICI,
            "gomme": TipoGuasto.PNEUMATICI,
            "ruote": TipoGuasto.PNEUMATICI,
            "freni": TipoGuasto.FRENI,
            "frenata": TipoGuasto.FRENI,
            "motore": TipoGuasto.MOTORE,
            "engine": TipoGuasto.MOTORE,
            "trasmissione": TipoGuasto.TRASMISSIONE,
            "cambio": TipoGuasto.TRASMISSIONE,
            "elettrico": TipoGuasto.ELETTRICO,
            "elettronica": TipoGuasto.ELETTRICO,
            "idraulico": TipoGuasto.IDRAULICO,
            "carrozzeria": TipoGuasto.CARROZZERIA,
            "sospensioni": TipoGuasto.SOSPENSIONI,
            "revisione": TipoGuasto.REVISIONE,
            "tagliando": TipoGuasto.TAGLIANDO,
        }
        return mapping.get(valore, TipoGuasto.ALTRO)

    def _parse_data(self, valore: str) -> Optional[date]:
        """Converte stringa in date"""
        if not valore:
            return None

        formati = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]
        for fmt in formati:
            try:
                return datetime.strptime(valore.strip(), fmt).date()
            except ValueError:
                continue
        return None

    def _parse_float(self, valore: str) -> float:
        """Converte stringa in float"""
        try:
            return float(valore.replace(",", ".").strip())
        except (ValueError, AttributeError):
            return 0.0


# Singleton
_loader: Optional[MaintenanceDataLoader] = None


def get_data_loader() -> MaintenanceDataLoader:
    """Ottiene istanza singleton del loader"""
    global _loader
    if _loader is None:
        _loader = MaintenanceDataLoader()
    return _loader


# Alias per compatibilità
def get_history_learner() -> MaintenanceDataLoader:
    """Alias per compatibilità con naming del planner"""
    return get_data_loader()
