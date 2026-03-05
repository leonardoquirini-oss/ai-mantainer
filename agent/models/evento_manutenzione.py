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
    """
    Classificazione semplificata dei tipi di guasto/intervento.
    DEPRECATO: usare CategoriaIntervento per analisi più granulari.
    """
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


class CategoriaIntervento(str, Enum):
    """
    Categorizzazione dettagliata degli interventi di manutenzione.
    Basata sulle 15 macro-categorie AdHoc per massima granularità.
    """
    PNEUMATICI = "01. PNEUMATICI"
    IMPIANTO_FRENANTE = "02. IMPIANTO FRENANTE"
    SOSPENSIONI = "03. SOSPENSIONI E AMMORTIZZATORI"
    CARROZZERIA_CONTAINER = "04. CARROZZERIA CONTAINER / CASSE MOBILI"
    TELONI_COPERTURE = "05. TELONI E COPERTURE"
    ELETTRICO_LUCI = "06. IMPIANTO ELETTRICO E LUCI"
    MOTORE_MECCANICA = "07. MOTORE E MECCANICA MOTRICE"
    MOZZI_RUOTE = "08. MOZZI E RUOTE"
    REVISIONE = "09. REVISIONE E CONTROLLI PERIODICI"
    SILOS_CISTERNA = "10. ATTREZZATURE SILOS / CISTERNA"
    ROTOCELLA_TWIST = "11. ROTOCELLA E TWIST LOCK"
    SOCCORSO = "12. SOCCORSO E INTERVENTI FUORI SEDE"
    CONSUMO_FLUIDI = "13. MATERIALI DI CONSUMO E FLUIDI"
    STRUTTURA_SALDATURE = "14. STRUTTURA METALLICA E SALDATURE"
    ALLESTIMENTO = "15. ALLESTIMENTO E PERSONALIZZAZIONE"
    NON_CLASSIFICATO = "NON CLASSIFICATO"

    @classmethod
    def from_string(cls, valore: str) -> "CategoriaIntervento":
        """Converte stringa in CategoriaIntervento"""
        for cat in cls:
            if cat.value == valore:
                return cat
        return cls.NON_CLASSIFICATO


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
    - tipo_guasto: classificazione dell'evento (DEPRECATO)
    - categorie: lista di CategoriaIntervento (15 categorie AdHoc)
    - pesi_categorie: pesi proporzionali per interventi multi-categoria
    - data_evento: data in cui si è verificato
    - data_acquisto_immatricolazione: per calcolare età

    NON disponibili: km percorsi, ore motore
    """
    mezzo_id: str
    tipo_mezzo: TipoMezzo
    tipo_guasto: TipoGuasto  # DEPRECATO: mantenuto per retrocompatibilità
    data_evento: date
    data_acquisto: Optional[date] = None
    data_immatricolazione: Optional[date] = None
    descrizione: str = ""
    costo: float = 0.0
    straordinario: bool = True  # True se guasto imprevisto, False se programmato

    # NUOVO: Categorie multiple con pesi proporzionali
    categorie: List[CategoriaIntervento] = field(default_factory=list)
    pesi_categorie: List[float] = field(default_factory=list)  # Somma = 1.0

    def __post_init__(self):
        """Inizializza pesi di default se non specificati"""
        if self.categorie and not self.pesi_categorie:
            n = len(self.categorie)
            self.pesi_categorie = [1.0 / n] * n

    def get_peso_categoria(self, categoria: CategoriaIntervento) -> float:
        """Ottiene il peso per una specifica categoria"""
        if categoria in self.categorie:
            idx = self.categorie.index(categoria)
            return self.pesi_categorie[idx] if idx < len(self.pesi_categorie) else 0.0
        return 0.0

    def ha_categoria(self, categoria: CategoriaIntervento) -> bool:
        """Verifica se l'evento ha una specifica categoria"""
        return categoria in self.categorie

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
            "categorie": [c.value for c in self.categorie],
            "pesi_categorie": self.pesi_categorie,
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
        """Lista tipi guasto presenti nel dataset (DEPRECATO)"""
        return list(set(
            e.tipo_guasto.value if isinstance(e.tipo_guasto, TipoGuasto) else e.tipo_guasto
            for e in self.eventi
        ))

    def get_categorie_presenti(self) -> List[CategoriaIntervento]:
        """Lista categorie intervento presenti nel dataset"""
        categorie = set()
        for e in self.eventi:
            for cat in e.categorie:
                categorie.add(cat)
        return sorted(categorie, key=lambda c: c.value)

    def get_eventi_per_categoria(
        self,
        categoria: CategoriaIntervento
    ) -> List[tuple]:
        """
        Filtra eventi per categoria, ritornando tuple (evento, peso).

        Args:
            categoria: Categoria da filtrare

        Returns:
            Lista di tuple (EventoManutenzione, peso)
        """
        risultato = []
        for e in self.eventi:
            if e.ha_categoria(categoria):
                peso = e.get_peso_categoria(categoria)
                risultato.append((e, peso))
        return risultato

    def get_mezzi_senza_categoria(self, categoria: CategoriaIntervento) -> List[Mezzo]:
        """
        Ottiene mezzi che non hanno avuto una certa categoria di intervento.
        Questi sono dati censurati per l'analisi di sopravvivenza.
        """
        mezzi_con_categoria = {
            e.mezzo_id for e in self.eventi
            if e.ha_categoria(categoria)
        }
        return [m for m in self.mezzi if m.mezzo_id not in mezzi_con_categoria]

    def statistiche_base(self) -> Dict:
        """Statistiche descrittive del dataset"""
        # Conta eventi per categoria (con pesi)
        eventi_per_categoria = {}
        for cat in self.get_categorie_presenti():
            eventi_cat = self.get_eventi_per_categoria(cat)
            # Conta ponderato
            conteggio_pesato = sum(peso for _, peso in eventi_cat)
            eventi_per_categoria[cat.value] = round(conteggio_pesato, 1)

        return {
            "totale_eventi": len(self.eventi),
            "totale_mezzi": len(self.mezzi),
            "tipi_mezzo": self.get_tipi_mezzo_presenti(),
            "tipi_guasto": self.get_tipi_guasto_presenti(),  # DEPRECATO
            "categorie": [c.value for c in self.get_categorie_presenti()],
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
            "eventi_per_categoria": eventi_per_categoria,
        }
