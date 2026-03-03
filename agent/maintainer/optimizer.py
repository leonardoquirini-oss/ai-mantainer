"""
Ottimizzatore per manutenzione predittiva basato su modelli statistici.

Implementa:
1. Survival Analysis (Kaplan-Meier + Cox Proportional Hazards)
2. Weibull Analysis
3. NHPP (Non-Homogeneous Poisson Process) Power Law

Configurazione: config/maintenance.yaml
"""

import math
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Tuple, Dict, Any
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger("maintenance-agent.optimizer")

# Import modelli
from ..models import (
    EventoManutenzione,
    Mezzo,
    DatasetManutenzione,
    RisultatoWeibull,
    RisultatoKaplanMeier,
    RisultatoCoxPH,
    RisultatoNHPP,
    PianoManutenzione,
    IntervalloManutenzione,
    ClassificazioneGuasto,
    TrendNHPP,
)
from .maintenance_config import get_maintenance_config


# Costanti per classificazione
BETA_INFANTILE_MAX = 0.8
BETA_CASUALE_MAX = 1.2
BETA_NHPP_DETERIORAMENTO = 1.15
BETA_NHPP_MIGLIORAMENTO = 0.85

# Minimo campioni per analisi affidabili
MIN_CAMPIONI_WEIBULL = 5
MIN_CAMPIONI_COX = 10
MIN_GUASTI_NHPP = 3


class MaintenanceOptimizer:
    """
    Ottimizzatore per manutenzione predittiva.

    Applica modelli statistici allo storico guasti per:
    - Stimare funzioni di sopravvivenza (Kaplan-Meier)
    - Quantificare rischio relativo per tipo mezzo (Cox PH)
    - Classificare pattern di guasto (Weibull)
    - Prevedere guasti futuri per singoli mezzi (NHPP)
    - Generare piano di manutenzione ordinaria basato su evidenze
    """

    def __init__(self):
        self._config = get_maintenance_config()
        self._lifelines_available = self._check_lifelines()

    def _check_lifelines(self) -> bool:
        """Verifica se lifelines è disponibile"""
        try:
            import lifelines
            return True
        except ImportError:
            logger.warning(
                "Libreria 'lifelines' non installata. "
                "Alcune analisi non saranno disponibili. "
                "Installa con: pip install lifelines"
            )
            return False

    # ==========================================================================
    # PREPARAZIONE DATI
    # ==========================================================================

    def prepara_dati_sopravvivenza(
        self,
        eventi: List[EventoManutenzione],
        mezzi: List[Mezzo],
        tipo_guasto: str,
        data_fine_osservazione: date = None
    ) -> pd.DataFrame:
        """
        Prepara DataFrame per analisi di sopravvivenza.

        Include:
        - Mezzi CON guasto: evento=1, durata=tempo al primo guasto
        - Mezzi SENZA guasto: evento=0, durata=tempo osservazione (censurati)

        Args:
            eventi: Lista eventi manutenzione
            mezzi: Lista tutti i mezzi
            tipo_guasto: Tipo guasto da analizzare
            data_fine_osservazione: Data fine periodo osservazione

        Returns:
            DataFrame con colonne: mezzo_id, tipo_mezzo, durata, evento
        """
        if data_fine_osservazione is None:
            data_fine_osservazione = date.today()

        # Filtra eventi per tipo guasto
        eventi_guasto = [e for e in eventi if self._match_tipo_guasto(e, tipo_guasto)]

        # Raggruppa per mezzo - prendi primo guasto
        primo_guasto_per_mezzo = {}
        for e in eventi_guasto:
            if e.mezzo_id not in primo_guasto_per_mezzo:
                primo_guasto_per_mezzo[e.mezzo_id] = e
            elif e.data_evento < primo_guasto_per_mezzo[e.mezzo_id].data_evento:
                primo_guasto_per_mezzo[e.mezzo_id] = e

        records = []

        # Mezzi con guasto
        for mezzo_id, evento in primo_guasto_per_mezzo.items():
            eta_mesi = evento.eta_mezzo_mesi
            if eta_mesi is not None and eta_mesi >= 0:
                tipo_mezzo = evento.tipo_mezzo
                if hasattr(tipo_mezzo, 'value'):
                    tipo_mezzo = tipo_mezzo.value
                records.append({
                    "mezzo_id": mezzo_id,
                    "tipo_mezzo": tipo_mezzo,
                    "durata": max(1, eta_mesi),  # Almeno 1 mese
                    "evento": 1  # Guasto osservato
                })

        # Mezzi senza guasto (censurati)
        mezzi_con_guasto = set(primo_guasto_per_mezzo.keys())
        for mezzo in mezzi:
            if mezzo.mezzo_id not in mezzi_con_guasto:
                eta_mesi = mezzo.eta_mesi_a_data(data_fine_osservazione)
                if eta_mesi is not None and eta_mesi > 0:
                    tipo_mezzo = mezzo.tipo_mezzo
                    if hasattr(tipo_mezzo, 'value'):
                        tipo_mezzo = tipo_mezzo.value
                    records.append({
                        "mezzo_id": mezzo.mezzo_id,
                        "tipo_mezzo": tipo_mezzo,
                        "durata": eta_mesi,
                        "evento": 0  # Censurato
                    })

        return pd.DataFrame(records)

    def prepara_dati_ricorrenza(
        self,
        eventi: List[EventoManutenzione],
        mezzo_id: str
    ) -> pd.DataFrame:
        """
        Prepara dati per analisi di ricorrenza NHPP.

        Args:
            eventi: Lista eventi manutenzione
            mezzo_id: ID del mezzo da analizzare

        Returns:
            DataFrame con tempi cumulativi dei guasti
        """
        eventi_mezzo = [e for e in eventi if e.mezzo_id == mezzo_id]
        eventi_mezzo = sorted(eventi_mezzo, key=lambda e: e.data_evento)

        if not eventi_mezzo:
            return pd.DataFrame()

        # Calcola tempi cumulativi dall'inizio vita
        records = []
        for i, e in enumerate(eventi_mezzo):
            eta_mesi = e.eta_mezzo_mesi
            if eta_mesi is not None:
                records.append({
                    "n_guasto": i + 1,
                    "tempo_cumulativo": eta_mesi,
                    "tipo_guasto": e.tipo_guasto.value if hasattr(e.tipo_guasto, 'value') else e.tipo_guasto
                })

        return pd.DataFrame(records)

    def _match_tipo_guasto(self, evento: EventoManutenzione, tipo_guasto: str) -> bool:
        """Verifica match tipo guasto"""
        evento_tipo = evento.tipo_guasto
        if hasattr(evento_tipo, 'value'):
            evento_tipo = evento_tipo.value
        return evento_tipo.lower() == tipo_guasto.lower()

    # ==========================================================================
    # ANALISI KAPLAN-MEIER
    # ==========================================================================

    def analisi_kaplan_meier(
        self,
        df: pd.DataFrame,
        tipo_mezzo: str = None,
        tipo_guasto: str = ""
    ) -> Optional[RisultatoKaplanMeier]:
        """
        Esegue analisi Kaplan-Meier per stimare funzione di sopravvivenza.

        S(t) = P(T > t) = probabilità che il mezzo non abbia guasto entro t mesi.

        Args:
            df: DataFrame con colonne durata, evento
            tipo_mezzo: Filtra per tipo mezzo (opzionale)
            tipo_guasto: Tipo guasto analizzato

        Returns:
            RisultatoKaplanMeier o None se dati insufficienti
        """
        if not self._lifelines_available:
            return self._kaplan_meier_semplificato(df, tipo_mezzo, tipo_guasto)

        from lifelines import KaplanMeierFitter

        # Filtra per tipo mezzo se specificato
        if tipo_mezzo:
            df = df[df["tipo_mezzo"] == tipo_mezzo]

        if len(df) < MIN_CAMPIONI_WEIBULL:
            logger.warning(f"Dati insufficienti per KM: {len(df)} campioni")
            return None

        kmf = KaplanMeierFitter()
        kmf.fit(df["durata"], event_observed=df["evento"])

        # Estrai risultati
        result = RisultatoKaplanMeier(
            tipo_mezzo=tipo_mezzo or "tutti",
            tipo_guasto=tipo_guasto,
            n_campioni=len(df),
            n_eventi=int(df["evento"].sum()),
            n_censurati=int((df["evento"] == 0).sum()),
        )

        # Mediana
        median = kmf.median_survival_time_
        if not np.isinf(median):
            result.mediana_mesi = float(median)
            ci = kmf.confidence_interval_median_survival_time_
            result.mediana_ci_lower = float(ci.iloc[0, 0]) if not ci.empty else None
            result.mediana_ci_upper = float(ci.iloc[0, 1]) if not ci.empty else None

        # Sopravvivenza a tempi specifici
        for mesi in [12, 24, 36, 48]:
            try:
                surv = kmf.predict(mesi)
                setattr(result, f"sopravvivenza_{mesi}_mesi", float(surv))
            except Exception:
                pass

        # Dati per plotting
        result.tempi = list(kmf.survival_function_.index)
        result.survival = list(kmf.survival_function_.iloc[:, 0])

        ci_df = kmf.confidence_interval_survival_function_
        if not ci_df.empty:
            result.ci_lower = list(ci_df.iloc[:, 0])
            result.ci_upper = list(ci_df.iloc[:, 1])

        return result

    def _kaplan_meier_semplificato(
        self,
        df: pd.DataFrame,
        tipo_mezzo: str,
        tipo_guasto: str
    ) -> Optional[RisultatoKaplanMeier]:
        """Versione semplificata senza lifelines"""
        if tipo_mezzo:
            df = df[df["tipo_mezzo"] == tipo_mezzo]

        if len(df) < MIN_CAMPIONI_WEIBULL:
            return None

        # Stima semplice: % sopravvissuti a vari tempi
        result = RisultatoKaplanMeier(
            tipo_mezzo=tipo_mezzo or "tutti",
            tipo_guasto=tipo_guasto,
            n_campioni=len(df),
            n_eventi=int(df["evento"].sum()),
            n_censurati=int((df["evento"] == 0).sum()),
        )

        # Stima grezza
        for mesi in [12, 24, 36, 48]:
            # % mezzi che hanno durato almeno 'mesi' senza guasto
            sopravvissuti = ((df["durata"] > mesi) | (df["evento"] == 0)).sum()
            setattr(result, f"sopravvivenza_{mesi}_mesi", sopravvissuti / len(df))

        return result

    # ==========================================================================
    # ANALISI COX PROPORTIONAL HAZARDS
    # ==========================================================================

    def analisi_cox_ph(
        self,
        df: pd.DataFrame,
        tipo_guasto: str
    ) -> Optional[RisultatoCoxPH]:
        """
        Esegue analisi Cox Proportional Hazards.

        Quantifica effetto del tipo_mezzo sul rischio relativo di guasto.

        Args:
            df: DataFrame con colonne durata, evento, tipo_mezzo
            tipo_guasto: Tipo guasto analizzato

        Returns:
            RisultatoCoxPH o None se dati insufficienti
        """
        if not self._lifelines_available:
            logger.warning("Cox PH richiede lifelines")
            return None

        if len(df) < MIN_CAMPIONI_COX:
            logger.warning(f"Dati insufficienti per Cox PH: {len(df)} campioni")
            return None

        from lifelines import CoxPHFitter

        # One-hot encoding tipo_mezzo
        df_encoded = pd.get_dummies(df, columns=["tipo_mezzo"], drop_first=True)

        # Identifica colonne tipo_mezzo
        tipo_cols = [c for c in df_encoded.columns if c.startswith("tipo_mezzo_")]

        if not tipo_cols:
            logger.warning("Meno di 2 tipi mezzo per Cox PH")
            return None

        # Fit modello
        cph = CoxPHFitter()
        try:
            cph.fit(
                df_encoded[["durata", "evento"] + tipo_cols],
                duration_col="durata",
                event_col="evento"
            )
        except Exception as e:
            logger.error(f"Errore fit Cox PH: {e}")
            return None

        # Estrai risultati
        # Trova tipo mezzo di riferimento (quello droppato)
        tipi_presenti = df["tipo_mezzo"].unique()
        tipo_riferimento = [t for t in tipi_presenti if f"tipo_mezzo_{t}" not in tipo_cols][0]

        result = RisultatoCoxPH(
            tipo_guasto=tipo_guasto,
            tipo_mezzo_riferimento=tipo_riferimento,
            concordance=cph.concordance_index_,
            log_likelihood=cph.log_likelihood_,
        )

        # Hazard ratios
        summary = cph.summary
        for col in tipo_cols:
            tipo = col.replace("tipo_mezzo_", "")
            result.hazard_ratios[tipo] = float(np.exp(summary.loc[col, "coef"]))
            result.ci_lower[tipo] = float(np.exp(summary.loc[col, "coef lower 95%"]))
            result.ci_upper[tipo] = float(np.exp(summary.loc[col, "coef upper 95%"]))
            result.p_values[tipo] = float(summary.loc[col, "p"])

        return result

    # ==========================================================================
    # ANALISI WEIBULL
    # ==========================================================================

    def analisi_weibull(
        self,
        df: pd.DataFrame,
        tipo_mezzo: str,
        tipo_guasto: str,
        affidabilita_target: float = 0.90
    ) -> Optional[RisultatoWeibull]:
        """
        Fitta distribuzione Weibull sui tempi al guasto.

        Il parametro beta determina la classificazione:
        - beta < 0.8: guasti infantili
        - 0.8 <= beta <= 1.2: guasti casuali
        - beta > 1.2: guasti da usura

        Args:
            df: DataFrame con colonne durata, evento
            tipo_mezzo: Tipo mezzo analizzato
            tipo_guasto: Tipo guasto analizzato
            affidabilita_target: Affidabilità target per intervallo manutenzione

        Returns:
            RisultatoWeibull o None se dati insufficienti
        """
        # Filtra per tipo mezzo
        df_filtrato = df[df["tipo_mezzo"] == tipo_mezzo]

        # Prendi solo eventi (guasti effettivi) per fit Weibull
        tempi = df_filtrato[df_filtrato["evento"] == 1]["durata"].values

        if len(tempi) < MIN_CAMPIONI_WEIBULL:
            logger.warning(
                f"Dati insufficienti per Weibull {tipo_mezzo}/{tipo_guasto}: "
                f"{len(tempi)} eventi"
            )
            return None

        # Fit Weibull usando scipy o lifelines
        beta, eta, log_lik = self._fit_weibull(tempi)

        if beta is None:
            return None

        # Classifica
        if beta < BETA_INFANTILE_MAX:
            classificazione = ClassificazioneGuasto.INFANTILE
        elif beta <= BETA_CASUALE_MAX:
            classificazione = ClassificazioneGuasto.CASUALE
        else:
            classificazione = ClassificazioneGuasto.USURA

        result = RisultatoWeibull(
            tipo_mezzo=tipo_mezzo,
            tipo_guasto=tipo_guasto,
            beta=beta,
            eta=eta,
            n_campioni=len(tempi),
            classificazione=classificazione,
            affidabilita_target=affidabilita_target,
            log_likelihood=log_lik,
        )

        # Calcola affidabilità R(t) = exp(-(t/eta)^beta)
        for mesi in [6, 12, 24, 36, 48]:
            R = math.exp(-((mesi / eta) ** beta))
            setattr(result, f"affidabilita_{mesi}_mesi", R)

        # Calcola AIC
        result.aic = 2 * 2 - 2 * log_lik  # 2 parametri (beta, eta)

        # Intervallo manutenzione (solo se usura)
        if classificazione == ClassificazioneGuasto.USURA:
            # t = eta * (-ln(R_target))^(1/beta)
            t_manutenzione = eta * ((-math.log(affidabilita_target)) ** (1 / beta))
            result.intervallo_manutenzione_mesi = int(t_manutenzione)

        return result

    def _fit_weibull(self, tempi: np.ndarray) -> Tuple[Optional[float], Optional[float], float]:
        """
        Fitta distribuzione Weibull.

        Returns:
            Tupla (beta, eta, log_likelihood) o (None, None, 0) se fallisce
        """
        try:
            from scipy.stats import weibull_min
            from scipy.optimize import minimize

            # Fit MLE
            # weibull_min ha parametri (c=shape, loc, scale)
            c, loc, scale = weibull_min.fit(tempi, floc=0)

            # Log likelihood
            log_lik = np.sum(weibull_min.logpdf(tempi, c, loc=0, scale=scale))

            return (c, scale, log_lik)

        except Exception as e:
            logger.error(f"Errore fit Weibull: {e}")

            # Fallback: stima dei momenti
            try:
                mean_t = np.mean(tempi)
                std_t = np.std(tempi)
                cv = std_t / mean_t if mean_t > 0 else 1

                # Stima approssimativa
                # Per Weibull: CV ≈ Gamma(1+1/beta) / sqrt(Gamma(1+2/beta) - Gamma(1+1/beta)^2)
                # Approssimazione: beta ≈ 1.2 / CV
                beta = 1.2 / cv if cv > 0 else 1.0
                beta = max(0.1, min(10, beta))

                # eta ≈ mean / Gamma(1 + 1/beta)
                from math import gamma
                eta = mean_t / gamma(1 + 1/beta)

                return (beta, eta, 0.0)
            except Exception:
                return (None, None, 0.0)

    # ==========================================================================
    # ANALISI NHPP (Non-Homogeneous Poisson Process)
    # ==========================================================================

    def analisi_nhpp(
        self,
        eventi: List[EventoManutenzione],
        mezzo: Mezzo
    ) -> Optional[RisultatoNHPP]:
        """
        Applica modello NHPP Power Law per mezzi con guasti ricorrenti.

        lambda(t) = (beta/eta) * (t/eta)^(beta-1)

        Args:
            eventi: Lista eventi per questo mezzo
            mezzo: Mezzo da analizzare

        Returns:
            RisultatoNHPP o None se dati insufficienti
        """
        eventi_mezzo = [e for e in eventi if e.mezzo_id == mezzo.mezzo_id]
        eventi_mezzo = sorted(eventi_mezzo, key=lambda e: e.data_evento)

        if len(eventi_mezzo) < MIN_GUASTI_NHPP:
            return None

        # Tempi cumulativi dei guasti
        tempi = []
        for e in eventi_mezzo:
            t = e.eta_mezzo_mesi
            if t is not None and t > 0:
                tempi.append(t)

        if len(tempi) < MIN_GUASTI_NHPP:
            return None

        tempi = np.array(sorted(tempi))
        n = len(tempi)
        T = tempi[-1]  # Tempo ultimo guasto

        # Stima MLE Power Law
        # beta_hat = n / sum(ln(T/t_i))
        # eta_hat = T / n^(1/beta)
        try:
            sum_log = np.sum(np.log(T / tempi[:-1]))  # Escludi ultimo
            if sum_log <= 0:
                sum_log = 0.01

            beta = (n - 1) / sum_log
            beta = max(0.1, min(10, beta))

            eta = T / (n ** (1/beta))

        except Exception as e:
            logger.error(f"Errore stima NHPP: {e}")
            return None

        # Classifica trend
        if beta > BETA_NHPP_DETERIORAMENTO:
            trend = TrendNHPP.DETERIORAMENTO
        elif beta < BETA_NHPP_MIGLIORAMENTO:
            trend = TrendNHPP.MIGLIORAMENTO
        else:
            trend = TrendNHPP.STABILE

        # Età attuale
        eta_attuale = mezzo.eta_mesi or T

        # Tasso attuale di guasto
        # lambda(t) = (beta/eta) * (t/eta)^(beta-1)
        if eta > 0:
            tasso_attuale = (beta / eta) * ((eta_attuale / eta) ** (beta - 1))
        else:
            tasso_attuale = 0

        # Guasti attesi nei prossimi 12 mesi
        # E[N(t)] = (t/eta)^beta
        # E[N(T+12)] - E[N(T)]
        guasti_attesi = ((eta_attuale + 12) / eta) ** beta - (eta_attuale / eta) ** beta

        # Tempo al prossimo guasto (approssimazione: 1/lambda)
        tempo_prossimo = 1 / tasso_attuale if tasso_attuale > 0 else None

        tipo_mezzo = mezzo.tipo_mezzo
        if hasattr(tipo_mezzo, 'value'):
            tipo_mezzo = tipo_mezzo.value

        return RisultatoNHPP(
            mezzo_id=mezzo.mezzo_id,
            tipo_mezzo=tipo_mezzo,
            n_guasti=n,
            eta_operativa_mesi=eta_attuale,
            beta=beta,
            eta=eta,
            trend=trend,
            tasso_attuale=tasso_attuale,
            guasti_attesi_12_mesi=guasti_attesi,
            tempo_prossimo_guasto_mesi=tempo_prossimo
        )

    # ==========================================================================
    # GENERAZIONE PIANO MANUTENZIONE
    # ==========================================================================

    def genera_piano_manutenzione(
        self,
        eventi: List[EventoManutenzione],
        mezzi: List[Mezzo],
        affidabilita_target: float = 0.90
    ) -> PianoManutenzione:
        """
        Genera piano di manutenzione ordinaria completo.

        Esegue tutte le analisi e produce output azionabile.

        Args:
            eventi: Lista storico eventi manutenzione
            mezzi: Lista mezzi della flotta
            affidabilita_target: Affidabilità target (default 90%)

        Returns:
            PianoManutenzione completo
        """
        piano = PianoManutenzione(
            data_generazione=date.today(),
            periodo_analisi_mesi=self._config.history_days // 30
        )

        # Identifica combinazioni tipo_mezzo x tipo_guasto
        tipi_mezzo = set()
        tipi_guasto = set()
        for e in eventi:
            tm = e.tipo_mezzo.value if hasattr(e.tipo_mezzo, 'value') else e.tipo_mezzo
            tg = e.tipo_guasto.value if hasattr(e.tipo_guasto, 'value') else e.tipo_guasto
            tipi_mezzo.add(tm)
            tipi_guasto.add(tg)

        # Per ogni tipo guasto
        for tipo_guasto in tipi_guasto:
            # Prepara dati sopravvivenza
            df = self.prepara_dati_sopravvivenza(eventi, mezzi, tipo_guasto)

            if df.empty:
                continue

            # Analisi Kaplan-Meier per tipo mezzo
            for tipo_mezzo in tipi_mezzo:
                km_result = self.analisi_kaplan_meier(df, tipo_mezzo, tipo_guasto)
                if km_result:
                    piano.risultati_kaplan_meier.append(km_result)

                # Analisi Weibull
                weibull_result = self.analisi_weibull(
                    df, tipo_mezzo, tipo_guasto, affidabilita_target
                )
                if weibull_result:
                    piano.risultati_weibull.append(weibull_result)

                    # Crea intervallo manutenzione
                    intervallo = self._crea_intervallo_da_weibull(weibull_result)
                    piano.intervalli.append(intervallo)

            # Analisi Cox PH (confronto tra tipi mezzo)
            if len(df["tipo_mezzo"].unique()) > 1:
                cox_result = self.analisi_cox_ph(df, tipo_guasto)
                if cox_result:
                    piano.risultati_cox.append(cox_result)

        # Analisi NHPP per mezzi con storico sufficiente
        for mezzo in mezzi:
            nhpp_result = self.analisi_nhpp(eventi, mezzo)
            if nhpp_result:
                piano.risultati_nhpp.append(nhpp_result)

                # Identifica mezzi critici
                if nhpp_result.trend == TrendNHPP.DETERIORAMENTO:
                    piano.mezzi_critici.append(mezzo.mezzo_id)

        # Calcola statistiche
        piano.statistiche = self._calcola_statistiche_piano(eventi, piano)

        return piano

    def _crea_intervallo_da_weibull(
        self,
        weibull: RisultatoWeibull
    ) -> IntervalloManutenzione:
        """Crea intervallo manutenzione da risultato Weibull"""

        if weibull.classificazione == ClassificazioneGuasto.USURA:
            return IntervalloManutenzione(
                tipo_mezzo=weibull.tipo_mezzo,
                tipo_guasto=weibull.tipo_guasto,
                intervallo_mesi=weibull.intervallo_manutenzione_mesi or 12,
                affidabilita_target=weibull.affidabilita_target,
                classificazione=weibull.classificazione,
                motivazione=(
                    f"Guasto da usura (beta={weibull.beta:.2f}). "
                    f"Manutenzione preventiva efficace."
                ),
                priorita=1,
                applicabile=True
            )

        elif weibull.classificazione == ClassificazioneGuasto.INFANTILE:
            return IntervalloManutenzione(
                tipo_mezzo=weibull.tipo_mezzo,
                tipo_guasto=weibull.tipo_guasto,
                intervallo_mesi=0,
                affidabilita_target=weibull.affidabilita_target,
                classificazione=weibull.classificazione,
                motivazione=(
                    f"Guasto infantile (beta={weibull.beta:.2f}). "
                    f"Manutenzione preventiva NON efficace. "
                    f"Migliorare controllo qualità all'ingresso."
                ),
                priorita=3,
                applicabile=False
            )

        else:  # CASUALE
            return IntervalloManutenzione(
                tipo_mezzo=weibull.tipo_mezzo,
                tipo_guasto=weibull.tipo_guasto,
                intervallo_mesi=0,
                affidabilita_target=weibull.affidabilita_target,
                classificazione=weibull.classificazione,
                motivazione=(
                    f"Guasto casuale (beta={weibull.beta:.2f}). "
                    f"Manutenzione preventiva poco efficace. "
                    f"Garantire disponibilità ricambi."
                ),
                priorita=3,
                applicabile=False
            )

    def _calcola_statistiche_piano(
        self,
        eventi: List[EventoManutenzione],
        piano: PianoManutenzione
    ) -> Dict:
        """Calcola statistiche del piano"""
        n_eventi = len(eventi)
        n_straordinari = sum(1 for e in eventi if e.straordinario)

        return {
            "totale_eventi_analizzati": n_eventi,
            "eventi_straordinari": n_straordinari,
            "rapporto_ordinarie": 1 - (n_straordinari / n_eventi) if n_eventi > 0 else 0,
            "analisi_weibull_completate": len(piano.risultati_weibull),
            "analisi_kaplan_meier_completate": len(piano.risultati_kaplan_meier),
            "analisi_cox_completate": len(piano.risultati_cox),
            "analisi_nhpp_completate": len(piano.risultati_nhpp),
            "intervalli_manutenzione_suggeriti": len([i for i in piano.intervalli if i.applicabile]),
            "mezzi_in_deterioramento": len(piano.mezzi_critici),
        }

    # ==========================================================================
    # METODI DI SUPPORTO PER CHAT
    # ==========================================================================

    def rispondi_domanda(
        self,
        domanda: str,
        eventi: List[EventoManutenzione],
        mezzi: List[Mezzo]
    ) -> str:
        """
        Risponde a domande in linguaggio naturale.

        Questo metodo sarà chiamato dall'agente LLM.

        Args:
            domanda: Domanda dell'utente
            eventi: Dataset eventi
            mezzi: Lista mezzi

        Returns:
            Risposta testuale
        """
        domanda_lower = domanda.lower()

        # Pattern matching semplice per demo
        # In produzione, questo sarà gestito dall'LLM

        if "piano" in domanda_lower or "programma" in domanda_lower:
            piano = self.genera_piano_manutenzione(eventi, mezzi)
            return piano.genera_report_testuale()

        if "critico" in domanda_lower or "deterioramento" in domanda_lower:
            risultati = []
            for mezzo in mezzi:
                nhpp = self.analisi_nhpp(eventi, mezzo)
                if nhpp and nhpp.trend == TrendNHPP.DETERIORAMENTO:
                    risultati.append(
                        f"- {mezzo.mezzo_id}: {nhpp.guasti_attesi_12_mesi:.1f} guasti attesi nei prossimi 12 mesi"
                    )
            if risultati:
                return "**Mezzi in deterioramento:**\n" + "\n".join(risultati)
            return "Nessun mezzo mostra trend di deterioramento significativo."

        if "statistiche" in domanda_lower or "riepilogo" in domanda_lower:
            piano = self.genera_piano_manutenzione(eventi, mezzi)
            stats = piano.statistiche
            return (
                f"**Statistiche manutenzione:**\n"
                f"- Eventi analizzati: {stats['totale_eventi_analizzati']}\n"
                f"- Rapporto ordinarie: {stats['rapporto_ordinarie']:.0%}\n"
                f"- Mezzi critici: {stats['mezzi_in_deterioramento']}\n"
                f"- Intervalli suggeriti: {stats['intervalli_manutenzione_suggeriti']}"
            )

        return "Non ho capito la domanda. Prova a chiedere: piano manutenzione, mezzi critici, statistiche."
