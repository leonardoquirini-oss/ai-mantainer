"""
Modelli per i risultati delle analisi statistiche
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Dict, Tuple
from enum import Enum


class ClassificazioneGuasto(str, Enum):
    """
    Classificazione del pattern di guasto basata su beta Weibull.

    - INFANTILE (beta < 0.8): problemi di qualità iniziale
    - CASUALE (0.8 <= beta <= 1.2): tasso costante, guasti random
    - USURA (beta > 1.2): deterioramento progressivo
    """
    INFANTILE = "infantile"
    CASUALE = "casuale"
    USURA = "usura"


class TrendNHPP(str, Enum):
    """
    Trend del tasso di guasto nel modello NHPP.

    - DETERIORAMENTO (beta > 1.15): guasti sempre più frequenti
    - STABILE (0.85 <= beta <= 1.15): tasso costante
    - MIGLIORAMENTO (beta < 0.85): guasti sempre meno frequenti
    """
    DETERIORAMENTO = "deterioramento"
    STABILE = "stabile"
    MIGLIORAMENTO = "miglioramento"


@dataclass
class RisultatoWeibull:
    """
    Risultato dell'analisi Weibull per una combinazione tipo_mezzo x tipo_guasto.

    Parametri:
    - beta (shape): determina la classificazione del guasto
    - eta (scale): tempo caratteristico (tempo al 63.2% di probabilità di guasto)
    """
    tipo_mezzo: str
    tipo_guasto: str
    beta: float  # shape parameter
    eta: float   # scale parameter (mesi)
    n_campioni: int
    classificazione: ClassificazioneGuasto

    # Affidabilità R(t) = exp(-(t/eta)^beta) a vari tempi
    affidabilita_6_mesi: float = 0.0
    affidabilita_12_mesi: float = 0.0
    affidabilita_24_mesi: float = 0.0
    affidabilita_36_mesi: float = 0.0
    affidabilita_48_mesi: float = 0.0

    # Intervallo manutenzione suggerito (solo se beta > 1.2)
    intervallo_manutenzione_mesi: Optional[int] = None
    affidabilita_target: float = 0.90

    # Qualità del fit
    log_likelihood: float = 0.0
    aic: float = 0.0

    def get_raccomandazione(self) -> str:
        """Genera raccomandazione basata sulla classificazione"""
        if self.classificazione == ClassificazioneGuasto.INFANTILE:
            return (
                f"GUASTI INFANTILI (beta={self.beta:.2f}): "
                f"La manutenzione preventiva periodica NON è efficace. "
                f"Azione: migliorare il controllo qualità all'ingresso e "
                f"durante l'installazione/avviamento."
            )
        elif self.classificazione == ClassificazioneGuasto.CASUALE:
            return (
                f"GUASTI CASUALI (beta={self.beta:.2f}): "
                f"Il tasso di guasto è costante nel tempo. "
                f"La manutenzione preventiva ha poco effetto. "
                f"Azione: garantire disponibilità ricambi e tempi di risposta rapidi."
            )
        else:  # USURA
            intervallo_str = f" ogni {self.intervallo_manutenzione_mesi} mesi" if self.intervallo_manutenzione_mesi else ""
            return (
                f"GUASTI DA USURA (beta={self.beta:.2f}): "
                f"Il rischio cresce col tempo. "
                f"La manutenzione preventiva È efficace. "
                f"Azione: definire controlli periodici{intervallo_str} "
                f"per mantenere affidabilità ≥{self.affidabilita_target:.0%}."
            )

    def to_dict(self) -> Dict:
        return {
            "tipo_mezzo": self.tipo_mezzo,
            "tipo_guasto": self.tipo_guasto,
            "beta": self.beta,
            "eta": self.eta,
            "n_campioni": self.n_campioni,
            "classificazione": self.classificazione.value,
            "affidabilita": {
                "6_mesi": self.affidabilita_6_mesi,
                "12_mesi": self.affidabilita_12_mesi,
                "24_mesi": self.affidabilita_24_mesi,
                "36_mesi": self.affidabilita_36_mesi,
                "48_mesi": self.affidabilita_48_mesi,
            },
            "intervallo_manutenzione_mesi": self.intervallo_manutenzione_mesi,
            "raccomandazione": self.get_raccomandazione(),
        }


@dataclass
class RisultatoKaplanMeier:
    """
    Risultato dell'analisi Kaplan-Meier (curva di sopravvivenza).

    S(t) = P(T > t) = probabilità che il mezzo non abbia subito
    il guasto entro t mesi dall'acquisto.
    """
    tipo_mezzo: str
    tipo_guasto: str
    n_campioni: int
    n_eventi: int  # guasti osservati
    n_censurati: int  # mezzi senza guasto (censurati a destra)

    # Mediana di sopravvivenza (tempo al 50% di probabilità di guasto)
    mediana_mesi: Optional[float] = None
    mediana_ci_lower: Optional[float] = None  # CI 95%
    mediana_ci_upper: Optional[float] = None

    # Probabilità di sopravvivenza a tempi specifici
    sopravvivenza_12_mesi: float = 0.0
    sopravvivenza_24_mesi: float = 0.0
    sopravvivenza_36_mesi: float = 0.0
    sopravvivenza_48_mesi: float = 0.0

    # Dati per plotting (tempi, survival, CI)
    tempi: List[float] = field(default_factory=list)
    survival: List[float] = field(default_factory=list)
    ci_lower: List[float] = field(default_factory=list)
    ci_upper: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "tipo_mezzo": self.tipo_mezzo,
            "tipo_guasto": self.tipo_guasto,
            "n_campioni": self.n_campioni,
            "n_eventi": self.n_eventi,
            "n_censurati": self.n_censurati,
            "mediana_mesi": self.mediana_mesi,
            "mediana_ci": [self.mediana_ci_lower, self.mediana_ci_upper],
            "sopravvivenza": {
                "12_mesi": self.sopravvivenza_12_mesi,
                "24_mesi": self.sopravvivenza_24_mesi,
                "36_mesi": self.sopravvivenza_36_mesi,
                "48_mesi": self.sopravvivenza_48_mesi,
            },
        }


@dataclass
class RisultatoCoxPH:
    """
    Risultato del modello Cox Proportional Hazards.

    Quantifica l'effetto del tipo_mezzo sul rischio relativo di guasto.
    HR > 1 = rischio maggiore, HR < 1 = rischio minore rispetto al riferimento.
    """
    tipo_guasto: str
    tipo_mezzo_riferimento: str  # categoria di riferimento (baseline)

    # Hazard ratios per ogni tipo mezzo
    hazard_ratios: Dict[str, float] = field(default_factory=dict)  # tipo_mezzo -> HR
    ci_lower: Dict[str, float] = field(default_factory=dict)  # CI 95%
    ci_upper: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)

    # Qualità del modello
    concordance: float = 0.0  # C-index
    log_likelihood: float = 0.0

    def get_significativi(self, alpha: float = 0.05) -> Dict[str, float]:
        """Ritorna solo gli hazard ratio statisticamente significativi"""
        return {
            tipo: hr
            for tipo, hr in self.hazard_ratios.items()
            if self.p_values.get(tipo, 1.0) < alpha
        }

    def interpreta(self) -> List[str]:
        """Genera interpretazione testuale dei risultati"""
        interpretazioni = []
        for tipo, hr in self.hazard_ratios.items():
            p = self.p_values.get(tipo, 1.0)
            sig = "**" if p < 0.01 else "*" if p < 0.05 else ""

            if hr > 1:
                interpretazioni.append(
                    f"{tipo}: rischio {hr:.1f}x maggiore vs {self.tipo_mezzo_riferimento} "
                    f"(p={p:.3f}){sig}"
                )
            elif hr < 1:
                interpretazioni.append(
                    f"{tipo}: rischio {1/hr:.1f}x minore vs {self.tipo_mezzo_riferimento} "
                    f"(p={p:.3f}){sig}"
                )
        return interpretazioni

    def to_dict(self) -> Dict:
        return {
            "tipo_guasto": self.tipo_guasto,
            "riferimento": self.tipo_mezzo_riferimento,
            "hazard_ratios": self.hazard_ratios,
            "p_values": self.p_values,
            "concordance": self.concordance,
            "interpretazione": self.interpreta(),
        }


@dataclass
class RisultatoNHPP:
    """
    Risultato del modello NHPP (Non-Homogeneous Poisson Process) Power Law.

    Per mezzi con almeno 3 guasti, modella il tasso di guasto variabile nel tempo.
    lambda(t) = (beta/eta) * (t/eta)^(beta-1)
    """
    mezzo_id: str
    tipo_mezzo: str
    n_guasti: int
    eta_operativa_mesi: int  # età attuale del mezzo

    # Parametri Power Law
    beta: float  # shape
    eta: float   # scale

    # Trend
    trend: TrendNHPP

    # Tasso attuale di guasto (guasti/mese)
    tasso_attuale: float = 0.0

    # Previsioni
    guasti_attesi_12_mesi: float = 0.0  # E[N(T+12)] - E[N(T)]
    tempo_prossimo_guasto_mesi: Optional[float] = None  # 1/lambda(T)

    def get_interpretazione(self) -> str:
        if self.trend == TrendNHPP.DETERIORAMENTO:
            return (
                f"DETERIORAMENTO: guasti sempre più frequenti (beta={self.beta:.2f}). "
                f"Il mezzo richiede attenzione crescente. "
                f"Valutare piano di sostituzione."
            )
        elif self.trend == TrendNHPP.MIGLIORAMENTO:
            return (
                f"MIGLIORAMENTO: guasti sempre meno frequenti (beta={self.beta:.2f}). "
                f"Possibile effetto rodaggio o interventi efficaci."
            )
        else:
            return (
                f"STABILE: tasso di guasto costante (beta={self.beta:.2f}). "
                f"Comportamento prevedibile."
            )

    def to_dict(self) -> Dict:
        return {
            "mezzo_id": self.mezzo_id,
            "tipo_mezzo": self.tipo_mezzo,
            "n_guasti": self.n_guasti,
            "eta_mesi": self.eta_operativa_mesi,
            "beta": self.beta,
            "eta": self.eta,
            "trend": self.trend.value,
            "tasso_attuale_guasti_mese": self.tasso_attuale,
            "guasti_attesi_12_mesi": self.guasti_attesi_12_mesi,
            "tempo_prossimo_guasto_mesi": self.tempo_prossimo_guasto_mesi,
            "interpretazione": self.get_interpretazione(),
        }


@dataclass
class IntervalloManutenzione:
    """Intervallo di manutenzione suggerito per una combinazione"""
    tipo_mezzo: str
    tipo_guasto: str
    intervallo_mesi: int
    affidabilita_target: float
    classificazione: ClassificazioneGuasto
    motivazione: str
    priorita: int = 0  # 1 = alta, 2 = media, 3 = bassa
    applicabile: bool = True  # False se manutenzione preventiva non efficace


@dataclass
class PianoManutenzione:
    """
    Piano di manutenzione ordinaria completo.
    Output azionabile per il responsabile flotta.
    """
    data_generazione: date
    periodo_analisi_mesi: int

    # Risultati analisi
    risultati_weibull: List[RisultatoWeibull] = field(default_factory=list)
    risultati_kaplan_meier: List[RisultatoKaplanMeier] = field(default_factory=list)
    risultati_cox: List[RisultatoCoxPH] = field(default_factory=list)
    risultati_nhpp: List[RisultatoNHPP] = field(default_factory=list)

    # Piano intervalli
    intervalli: List[IntervalloManutenzione] = field(default_factory=list)

    # Mezzi critici (trend deterioramento)
    mezzi_critici: List[str] = field(default_factory=list)

    # Statistiche
    statistiche: Dict = field(default_factory=dict)

    def get_intervalli_applicabili(self) -> List[IntervalloManutenzione]:
        """Ritorna solo intervalli dove manutenzione preventiva è efficace"""
        return [i for i in self.intervalli if i.applicabile]

    def get_intervalli_per_tipo_mezzo(self, tipo_mezzo: str) -> List[IntervalloManutenzione]:
        """Filtra intervalli per tipo mezzo"""
        return [i for i in self.intervalli if i.tipo_mezzo == tipo_mezzo]

    def genera_report_testuale(self) -> str:
        """Genera report testuale del piano"""
        lines = [
            f"# PIANO MANUTENZIONE ORDINARIA",
            f"Generato: {self.data_generazione}",
            f"Periodo analisi: {self.periodo_analisi_mesi} mesi",
            "",
            "## INTERVALLI MANUTENZIONE CONSIGLIATI",
            ""
        ]

        applicabili = self.get_intervalli_applicabili()
        for i in sorted(applicabili, key=lambda x: (x.priorita, x.tipo_mezzo)):
            lines.append(
                f"- **{i.tipo_mezzo.upper()}** / {i.tipo_guasto}: "
                f"ogni {i.intervallo_mesi} mesi "
                f"(affidabilità {i.affidabilita_target:.0%}) - {i.motivazione}"
            )

        non_applicabili = [i for i in self.intervalli if not i.applicabile]
        if non_applicabili:
            lines.extend([
                "",
                "## MANUTENZIONE PREVENTIVA NON CONSIGLIATA",
                ""
            ])
            for i in non_applicabili:
                lines.append(f"- {i.tipo_mezzo}/{i.tipo_guasto}: {i.motivazione}")

        if self.mezzi_critici:
            lines.extend([
                "",
                "## MEZZI CRITICI (richiedono attenzione)",
                ""
            ])
            for mezzo in self.mezzi_critici:
                lines.append(f"- {mezzo}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "data_generazione": self.data_generazione.isoformat(),
            "periodo_analisi_mesi": self.periodo_analisi_mesi,
            "intervalli": [
                {
                    "tipo_mezzo": i.tipo_mezzo,
                    "tipo_guasto": i.tipo_guasto,
                    "intervallo_mesi": i.intervallo_mesi,
                    "affidabilita_target": i.affidabilita_target,
                    "applicabile": i.applicabile,
                    "motivazione": i.motivazione,
                }
                for i in self.intervalli
            ],
            "mezzi_critici": self.mezzi_critici,
            "statistiche": self.statistiche,
        }
