"""
Modelli per eventi di manutenzione e mezzi
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Dict
from enum import Enum


class TipoMezzo(str, Enum):
    """Tipi di mezzo nella flotta"""
    SEMIRIMORCHIO = "semirimorchio"
    TRATTORE = "trattore"
    CONTAINER = "container"
    FURGONE = "furgone"
    ALTRO = "altro"


class TipoGuasto(str, Enum):
    """Classificazione dei tipi di guasto/intervento"""
    PNEUMATICI = "pneumatici"
    FRENI = "freni"
    MOTORE = "motore"
    TRASMISSIONE = "trasmissione"
    ELETTRICO = "elettrico"
    IDRAULICO = "idraulico"
    CARROZZERIA = "carrozzeria"
    SOSPENSIONI = "sospensioni"
    REVISIONE = "revisione"
    TAGLIANDO = "tagliando"
    ALTRO = "altro"


@dataclass
class Mezzo:
    """Rappresenta un mezzo della flotta"""
    mezzo_id: str
    tipo_mezzo: TipoMezzo
    data_acquisto: Optional[date] = None
    data_immatricolazione: Optional[date] = None
    targa: str = ""
    marca: str = ""
    modello: str = ""
    note: str = ""

    @property
    def data_inizio_vita(self) -> Optional[date]:
        """
        Data di inizio vita utile effettiva.
        Preferisce data_immatricolazione, altrimenti data_acquisto.
        """
        if self.data_immatricolazione:
            return self.data_immatricolazione
        return self.data_acquisto

    @property
    def eta_mesi(self) -> Optional[int]:
        """Età del mezzo in mesi dalla data di inizio vita"""
        if not self.data_inizio_vita:
            return None
        delta = date.today() - self.data_inizio_vita
        return delta.days // 30

    def eta_mesi_a_data(self, data_riferimento: date) -> Optional[int]:
        """Età del mezzo in mesi a una data specifica"""
        if not self.data_inizio_vita:
            return None
        delta = data_riferimento - self.data_inizio_vita
        return max(0, delta.days // 30)


@dataclass
class EventoManutenzione:
    """
    Rappresenta un evento di manutenzione storicizzato.

    Dati disponibili secondo il prompt:
    - mezzo_id: identificativo univoco del mezzo
    - tipo_mezzo: categoria
    - tipo_guasto: classificazione dell'evento
    - data_evento: data in cui si è verificato
    - data_acquisto_immatricolazione: per calcolare età

    NON disponibili: km percorsi, ore motore
    """
    mezzo_id: str
    tipo_mezzo: TipoMezzo
    tipo_guasto: TipoGuasto
    data_evento: date
    data_acquisto: Optional[date] = None
    data_immatricolazione: Optional[date] = None
    descrizione: str = ""
    costo: float = 0.0
    straordinario: bool = True  # True se guasto imprevisto, False se programmato

    @property
    def data_inizio_vita(self) -> Optional[date]:
        """Data inizio vita utile (preferisce immatricolazione)"""
        if self.data_immatricolazione:
            return self.data_immatricolazione
        return self.data_acquisto

    @property
    def eta_mezzo_mesi(self) -> Optional[int]:
        """
        Età del mezzo in mesi al momento dell'evento.
        Questo è il tempo t usato nelle analisi di sopravvivenza.
        """
        if not self.data_inizio_vita:
            return None
        delta = self.data_evento - self.data_inizio_vita
        return max(0, delta.days // 30)

    def to_dict(self) -> Dict:
        """Converte in dizionario per analisi pandas"""
        return {
            "mezzo_id": self.mezzo_id,
            "tipo_mezzo": self.tipo_mezzo.value if isinstance(self.tipo_mezzo, TipoMezzo) else self.tipo_mezzo,
            "tipo_guasto": self.tipo_guasto.value if isinstance(self.tipo_guasto, TipoGuasto) else self.tipo_guasto,
            "data_evento": self.data_evento,
            "data_inizio_vita": self.data_inizio_vita,
            "eta_mesi": self.eta_mezzo_mesi,
            "costo": self.costo,
            "straordinario": self.straordinario,
        }


@dataclass
class DatasetManutenzione:
    """
    Container per il dataset completo di manutenzione.
    Gestisce anche i dati censurati (mezzi senza guasto).
    """
    eventi: List[EventoManutenzione] = field(default_factory=list)
    mezzi: List[Mezzo] = field(default_factory=list)
    data_osservazione_fine: date = field(default_factory=date.today)

    def get_eventi_per_tipo_mezzo(self, tipo_mezzo: TipoMezzo) -> List[EventoManutenzione]:
        """Filtra eventi per tipo mezzo"""
        return [e for e in self.eventi if e.tipo_mezzo == tipo_mezzo]

    def get_eventi_per_tipo_guasto(self, tipo_guasto: TipoGuasto) -> List[EventoManutenzione]:
        """Filtra eventi per tipo guasto"""
        return [e for e in self.eventi if e.tipo_guasto == tipo_guasto]

    def get_eventi_per_mezzo(self, mezzo_id: str) -> List[EventoManutenzione]:
        """Ottiene tutti gli eventi di un mezzo ordinati per data"""
        eventi_mezzo = [e for e in self.eventi if e.mezzo_id == mezzo_id]
        return sorted(eventi_mezzo, key=lambda e: e.data_evento)

    def get_mezzi_senza_guasto(self, tipo_guasto: TipoGuasto) -> List[Mezzo]:
        """
        Ottiene mezzi che non hanno avuto un certo tipo di guasto.
        Questi sono dati censurati a destra per l'analisi di sopravvivenza.
        """
        mezzi_con_guasto = {
            e.mezzo_id for e in self.eventi
            if e.tipo_guasto == tipo_guasto
        }
        return [m for m in self.mezzi if m.mezzo_id not in mezzi_con_guasto]

    def get_tipi_mezzo_presenti(self) -> List[str]:
        """Lista tipi mezzo presenti nel dataset"""
        return list(set(
            e.tipo_mezzo.value if isinstance(e.tipo_mezzo, TipoMezzo) else e.tipo_mezzo
            for e in self.eventi
        ))

    def get_tipi_guasto_presenti(self) -> List[str]:
        """Lista tipi guasto presenti nel dataset"""
        return list(set(
            e.tipo_guasto.value if isinstance(e.tipo_guasto, TipoGuasto) else e.tipo_guasto
            for e in self.eventi
        ))

    def statistiche_base(self) -> Dict:
        """Statistiche descrittive del dataset"""
        return {
            "totale_eventi": len(self.eventi),
            "totale_mezzi": len(self.mezzi),
            "tipi_mezzo": self.get_tipi_mezzo_presenti(),
            "tipi_guasto": self.get_tipi_guasto_presenti(),
            "periodo_osservazione": {
                "inizio": min((e.data_evento for e in self.eventi), default=None),
                "fine": self.data_osservazione_fine,
            },
            "eventi_per_tipo_mezzo": {
                tipo: len(self.get_eventi_per_tipo_mezzo(TipoMezzo(tipo)))
                for tipo in self.get_tipi_mezzo_presenti()
            },
            "eventi_per_tipo_guasto": {
                tipo: len(self.get_eventi_per_tipo_guasto(TipoGuasto(tipo)))
                for tipo in self.get_tipi_guasto_presenti()
            },
        }
