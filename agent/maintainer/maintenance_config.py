"""
Loader per la configurazione del sistema manutenzioni da file YAML
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class IntervalloManutenzione:
    """Intervallo di manutenzione per un tipo"""
    tipo: str
    km: int = 0  # Intervallo in km
    giorni: int = 0  # Intervallo in giorni
    descrizione: str = ""


@dataclass
class CostoManutenzione:
    """Costo stimato per tipo manutenzione"""
    tipo: str
    costo_min: float = 0.0
    costo_max: float = 0.0
    costo_medio: float = 0.0


@dataclass
class TipoMezzoConfig:
    """Configurazione per un tipo di mezzo"""
    nome: str
    intervalli: Dict[str, IntervalloManutenzione] = field(default_factory=dict)
    costi: Dict[str, CostoManutenzione] = field(default_factory=dict)
    note: str = ""


@dataclass
class MaintenanceConfig:
    """Configurazione completa del sistema manutenzioni"""

    # Pesi per calcolo score
    weight_km: float = 0.35
    weight_tempo: float = 0.30
    weight_eta: float = 0.15
    weight_storico: float = 0.20

    # Soglie
    soglia_predizione: float = 0.3  # Score minimo per generare predizione
    soglia_urgenza_critica: float = 0.9
    soglia_urgenza_alta: float = 0.7
    soglia_urgenza_media: float = 0.5

    # Obiettivi
    obiettivo_rapporto_ordinarie: float = 0.8  # 80% ordinarie vs straordinarie

    # Storico
    history_days: int = 365

    # Configurazione per tipo mezzo
    tipi_mezzo: Dict[str, TipoMezzoConfig] = field(default_factory=dict)

    # Tipi di manutenzione standard
    tipi_manutenzione: List[str] = field(default_factory=lambda: [
        "tagliando",
        "revisione",
        "pneumatici",
        "freni",
        "olio",
        "filtri",
        "ordinaria",
        "straordinaria"
    ])

    # Intervalli default (se non specificati per tipo mezzo)
    intervalli_default: Dict[str, IntervalloManutenzione] = field(default_factory=dict)

    # Costi default
    costi_default: Dict[str, CostoManutenzione] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "MaintenanceConfig":
        """Carica configurazione da file YAML"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "maintenance.yaml"

        if not config_path.exists():
            # Ritorna configurazione default se file non esiste
            return cls._get_default_config()

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls._parse_config(data)

    @classmethod
    def _get_default_config(cls) -> "MaintenanceConfig":
        """Genera configurazione default"""
        config = cls()

        # Intervalli default per manutenzioni comuni
        config.intervalli_default = {
            "tagliando": IntervalloManutenzione(
                tipo="tagliando",
                km=20000,
                giorni=180,
                descrizione="Tagliando periodico"
            ),
            "revisione": IntervalloManutenzione(
                tipo="revisione",
                km=0,
                giorni=365,
                descrizione="Revisione annuale obbligatoria"
            ),
            "pneumatici": IntervalloManutenzione(
                tipo="pneumatici",
                km=80000,
                giorni=730,
                descrizione="Controllo/sostituzione pneumatici"
            ),
            "freni": IntervalloManutenzione(
                tipo="freni",
                km=60000,
                giorni=365,
                descrizione="Controllo impianto frenante"
            ),
            "olio": IntervalloManutenzione(
                tipo="olio",
                km=15000,
                giorni=180,
                descrizione="Cambio olio motore"
            ),
            "filtri": IntervalloManutenzione(
                tipo="filtri",
                km=30000,
                giorni=365,
                descrizione="Sostituzione filtri"
            ),
        }

        # Costi default
        config.costi_default = {
            "tagliando": CostoManutenzione(
                tipo="tagliando",
                costo_min=200,
                costo_max=500,
                costo_medio=350
            ),
            "revisione": CostoManutenzione(
                tipo="revisione",
                costo_min=100,
                costo_max=300,
                costo_medio=200
            ),
            "pneumatici": CostoManutenzione(
                tipo="pneumatici",
                costo_min=800,
                costo_max=2500,
                costo_medio=1500
            ),
            "freni": CostoManutenzione(
                tipo="freni",
                costo_min=300,
                costo_max=1200,
                costo_medio=700
            ),
            "olio": CostoManutenzione(
                tipo="olio",
                costo_min=100,
                costo_max=300,
                costo_medio=180
            ),
            "filtri": CostoManutenzione(
                tipo="filtri",
                costo_min=50,
                costo_max=200,
                costo_medio=120
            ),
            "ordinaria": CostoManutenzione(
                tipo="ordinaria",
                costo_min=100,
                costo_max=500,
                costo_medio=250
            ),
            "straordinaria": CostoManutenzione(
                tipo="straordinaria",
                costo_min=500,
                costo_max=5000,
                costo_medio=1500
            ),
        }

        # Configurazione per tipo mezzo
        config.tipi_mezzo = {
            "semirimorchio": TipoMezzoConfig(
                nome="semirimorchio",
                intervalli={
                    "revisione": IntervalloManutenzione("revisione", km=0, giorni=365),
                    "pneumatici": IntervalloManutenzione("pneumatici", km=100000, giorni=730),
                    "freni": IntervalloManutenzione("freni", km=80000, giorni=365),
                },
                note="Semirimorchi per trasporto merci"
            ),
            "trattore": TipoMezzoConfig(
                nome="trattore",
                intervalli={
                    "tagliando": IntervalloManutenzione("tagliando", km=25000, giorni=180),
                    "revisione": IntervalloManutenzione("revisione", km=0, giorni=365),
                    "olio": IntervalloManutenzione("olio", km=20000, giorni=180),
                    "pneumatici": IntervalloManutenzione("pneumatici", km=120000, giorni=730),
                    "freni": IntervalloManutenzione("freni", km=100000, giorni=365),
                },
                note="Trattori stradali"
            ),
            "furgone": TipoMezzoConfig(
                nome="furgone",
                intervalli={
                    "tagliando": IntervalloManutenzione("tagliando", km=15000, giorni=180),
                    "revisione": IntervalloManutenzione("revisione", km=0, giorni=365),
                    "olio": IntervalloManutenzione("olio", km=10000, giorni=180),
                },
                note="Furgoni per consegne"
            ),
        }

        return config

    @classmethod
    def _parse_config(cls, data: Dict[str, Any]) -> "MaintenanceConfig":
        """Parse configurazione da dizionario"""
        config = cls._get_default_config()

        # Pesi
        weights = data.get("weights", {})
        config.weight_km = weights.get("km", config.weight_km)
        config.weight_tempo = weights.get("tempo", config.weight_tempo)
        config.weight_eta = weights.get("eta", config.weight_eta)
        config.weight_storico = weights.get("storico", config.weight_storico)

        # Soglie
        soglie = data.get("soglie", {})
        config.soglia_predizione = soglie.get("predizione", config.soglia_predizione)
        config.soglia_urgenza_critica = soglie.get("urgenza_critica", config.soglia_urgenza_critica)
        config.soglia_urgenza_alta = soglie.get("urgenza_alta", config.soglia_urgenza_alta)
        config.soglia_urgenza_media = soglie.get("urgenza_media", config.soglia_urgenza_media)

        # Obiettivi
        config.obiettivo_rapporto_ordinarie = data.get(
            "obiettivo_rapporto_ordinarie",
            config.obiettivo_rapporto_ordinarie
        )

        # Storico
        config.history_days = data.get("history_days", config.history_days)

        # Tipi manutenzione
        if "tipi_manutenzione" in data:
            config.tipi_manutenzione = data["tipi_manutenzione"]

        # Intervalli da YAML
        for tipo, intervallo_data in data.get("intervalli", {}).items():
            config.intervalli_default[tipo] = IntervalloManutenzione(
                tipo=tipo,
                km=intervallo_data.get("km", 0),
                giorni=intervallo_data.get("giorni", 0),
                descrizione=intervallo_data.get("descrizione", "")
            )

        # Costi da YAML
        for tipo, costo_data in data.get("costi", {}).items():
            config.costi_default[tipo] = CostoManutenzione(
                tipo=tipo,
                costo_min=costo_data.get("min", 0),
                costo_max=costo_data.get("max", 0),
                costo_medio=costo_data.get("medio", 0)
            )

        # Configurazioni per tipo mezzo
        for tipo_mezzo, mezzo_data in data.get("tipi_mezzo", {}).items():
            intervalli = {}
            for tipo, int_data in mezzo_data.get("intervalli", {}).items():
                intervalli[tipo] = IntervalloManutenzione(
                    tipo=tipo,
                    km=int_data.get("km", 0),
                    giorni=int_data.get("giorni", 0),
                    descrizione=int_data.get("descrizione", "")
                )

            costi = {}
            for tipo, costo_data in mezzo_data.get("costi", {}).items():
                costi[tipo] = CostoManutenzione(
                    tipo=tipo,
                    costo_min=costo_data.get("min", 0),
                    costo_max=costo_data.get("max", 0),
                    costo_medio=costo_data.get("medio", 0)
                )

            config.tipi_mezzo[tipo_mezzo] = TipoMezzoConfig(
                nome=tipo_mezzo,
                intervalli=intervalli,
                costi=costi,
                note=mezzo_data.get("note", "")
            )

        return config

    def get_intervallo_km(self, tipo_manutenzione: str, tipo_mezzo: str = "") -> int:
        """
        Ottiene l'intervallo km per una manutenzione.

        Args:
            tipo_manutenzione: Tipo di manutenzione
            tipo_mezzo: Tipo di mezzo (opzionale)

        Returns:
            Intervallo in km
        """
        # Prima cerca nella configurazione specifica del tipo mezzo
        if tipo_mezzo and tipo_mezzo in self.tipi_mezzo:
            tm_config = self.tipi_mezzo[tipo_mezzo]
            if tipo_manutenzione in tm_config.intervalli:
                return tm_config.intervalli[tipo_manutenzione].km

        # Fallback a default
        if tipo_manutenzione in self.intervalli_default:
            return self.intervalli_default[tipo_manutenzione].km

        return 0

    def get_intervallo_giorni(self, tipo_manutenzione: str, tipo_mezzo: str = "") -> int:
        """
        Ottiene l'intervallo giorni per una manutenzione.

        Args:
            tipo_manutenzione: Tipo di manutenzione
            tipo_mezzo: Tipo di mezzo (opzionale)

        Returns:
            Intervallo in giorni
        """
        # Prima cerca nella configurazione specifica del tipo mezzo
        if tipo_mezzo and tipo_mezzo in self.tipi_mezzo:
            tm_config = self.tipi_mezzo[tipo_mezzo]
            if tipo_manutenzione in tm_config.intervalli:
                return tm_config.intervalli[tipo_manutenzione].giorni

        # Fallback a default
        if tipo_manutenzione in self.intervalli_default:
            return self.intervalli_default[tipo_manutenzione].giorni

        return 0

    def get_costo_stimato(self, tipo_manutenzione: str, tipo_mezzo: str = "") -> float:
        """
        Ottiene il costo stimato per una manutenzione.

        Args:
            tipo_manutenzione: Tipo di manutenzione
            tipo_mezzo: Tipo di mezzo (opzionale)

        Returns:
            Costo medio stimato
        """
        # Prima cerca nella configurazione specifica del tipo mezzo
        if tipo_mezzo and tipo_mezzo in self.tipi_mezzo:
            tm_config = self.tipi_mezzo[tipo_mezzo]
            if tipo_manutenzione in tm_config.costi:
                return tm_config.costi[tipo_manutenzione].costo_medio

        # Fallback a default
        if tipo_manutenzione in self.costi_default:
            return self.costi_default[tipo_manutenzione].costo_medio

        return 0.0

    def get_tipi_manutenzione(self) -> List[str]:
        """Ritorna lista tipi manutenzione configurati"""
        return self.tipi_manutenzione

    def get_urgenza(self, score: float) -> str:
        """
        Determina urgenza basata su score.

        Args:
            score: Score 0-1

        Returns:
            Stringa urgenza
        """
        if score >= self.soglia_urgenza_critica:
            return "critica"
        elif score >= self.soglia_urgenza_alta:
            return "alta"
        elif score >= self.soglia_urgenza_media:
            return "media"
        else:
            return "bassa"


# Singleton
_maintenance_config: Optional[MaintenanceConfig] = None


def get_maintenance_config() -> MaintenanceConfig:
    """Ottiene l'istanza singleton della configurazione"""
    global _maintenance_config
    if _maintenance_config is None:
        _maintenance_config = MaintenanceConfig.load()
    return _maintenance_config


def reload_maintenance_config() -> MaintenanceConfig:
    """Ricarica la configurazione da file"""
    global _maintenance_config
    _maintenance_config = MaintenanceConfig.load()
    return _maintenance_config
