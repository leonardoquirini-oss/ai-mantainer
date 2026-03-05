"""
Modelli dati per il sistema di manutenzione predittiva
"""

from .evento_manutenzione import (
    EventoManutenzione,
    Mezzo,
    TipoMezzo,
    TipoGuasto,
    CategoriaIntervento,
    DatasetManutenzione,
)
from .analisi import (
    RisultatoWeibull,
    RisultatoKaplanMeier,
    RisultatoCoxPH,
    RisultatoNHPP,
    PianoManutenzione,
    IntervalloManutenzione,
    ClassificazioneGuasto,
    TrendNHPP,
)

__all__ = [
    "EventoManutenzione",
    "Mezzo",
    "TipoMezzo",
    "TipoGuasto",
    "CategoriaIntervento",
    "DatasetManutenzione",
    "RisultatoWeibull",
    "RisultatoKaplanMeier",
    "RisultatoCoxPH",
    "RisultatoNHPP",
    "PianoManutenzione",
    "IntervalloManutenzione",
    "ClassificazioneGuasto",
    "TrendNHPP",
]
