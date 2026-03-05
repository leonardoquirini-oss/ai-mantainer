"""
Connettore API AdHoc per recuperare storico manutenzioni.

Identico a TIRConnector ma con URL diverso (http://192.168.0.12:9100/).
"""

import csv
import io
import httpx
import inspect
import logging
from datetime import date, datetime
from typing import List, Optional, Any, Dict

from ..config import APIConfig, get_settings

logger = logging.getLogger("maintenance-agent.adhoc")


def _get_caller_name() -> str:
    """Ritorna il nome del metodo chiamante (2 livelli sopra)"""
    stack = inspect.stack()
    if len(stack) >= 3:
        return stack[2].function
    return "unknown"


class AdHocConnector:
    """
    Connettore per l'API AdHoc (gestione manutenzioni).

    Fornisce accesso alla query 'elenco_manutenzioni' che restituisce
    lo storico degli interventi di manutenzione sui mezzi della flotta.

    Colonne restituite:
    - AZIENDA: B, G o C (Bernardini, Guido Bernardini, Cosmo)
    - SERIALE_DOC: riferimento contabilità
    - DESCRIZIONE: descrizione intervento
    - DETTAGLIO: specifica dell'intervento
    - TARGA: identificativo veicolo/container
    - DATA_INTERVENTO: data dell'intervento
    - CAUSALE: tipologia intervento
    - COSTO: costo dell'intervento
    - DATA_IMM_CTR: data immatricolazione container (se container)
    - DATA_IMM_MEZZO: data immatricolazione mezzo (se mezzo)
    """

    def __init__(self, config: Optional[APIConfig] = None):
        if config is None:
            config = get_settings().adhoc_api
        self.config = config
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy initialization del client HTTP"""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.base_url,
                headers={
                    "X-API-Key": self.config.api_key,
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout,
            )
        return self._client

    def close(self):
        """Chiude il client HTTP"""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def execute_template(
        self,
        template_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        output_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Esegue un template di query predefinito.

        Args:
            template_name: Nome del template (es. elenco_manutenzioni)
            parameters: Parametri del template
            output_format: Formato output ("json" o "csv")

        Returns:
            Risposta API con data, rowCount, columns
        """
        payload = {
            "templateName": template_name,
            "parameters": parameters or {}
        }

        if output_format:
            payload["outputFormat"] = output_format

        caller = _get_caller_name()
        logger.info(f"AdHoc API: POST /api/Query/templates/execute | caller: {caller}")
        logger.info(f"AdHoc TEMPLATE: {template_name} | format: {output_format or 'json'}")

        response = self.client.post(
            "/api/Query/templates/execute",
            json=payload
        )
        response.raise_for_status()

        # Se CSV, parsa il contenuto
        if output_format == "csv":
            return self._parse_csv_response(response.text)

        result = response.json()
        logger.info(f"AdHoc RESULT: {result.get('rowCount', 0)} righe")
        return result

    def _parse_csv_response(self, csv_text: str) -> Dict[str, Any]:
        """
        Parsa una risposta CSV in formato dizionario.

        Args:
            csv_text: Contenuto CSV come stringa

        Returns:
            Dict con data (lista di dict) e rowCount
        """
        if not csv_text or not csv_text.strip():
            return {"data": [], "rowCount": 0}

        # Rimuovi BOM se presente
        if csv_text.startswith('\ufeff'):
            csv_text = csv_text[1:]

        # Auto-detect delimiter
        first_line = csv_text.split('\n')[0]
        if ';' in first_line and ',' not in first_line:
            delimiter = ';'
        elif ',' in first_line:
            delimiter = ','
        else:
            delimiter = ';'

        # Usa csv.DictReader per parsing
        reader = csv.DictReader(io.StringIO(csv_text), delimiter=delimiter)
        data = list(reader)

        logger.info(f"AdHoc CSV RESULT: {len(data)} righe (delimiter: '{delimiter}')")
        return {
            "data": data,
            "rowCount": len(data),
            "columns": reader.fieldnames or []
        }

    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Esegue una query SQL custom.

        Args:
            query: Query SQL da eseguire

        Returns:
            Risposta API con risultati
        """
        caller = _get_caller_name()
        logger.info(f"AdHoc API: POST /api/Query/execute | caller: {caller}")
        query_preview = query[:200] + "..." if len(query) > 200 else query
        logger.info(f"AdHoc QUERY: {query_preview}")

        response = self.client.post(
            "/api/Query/execute",
            json={"query": query}
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"AdHoc RESULT: {result.get('rowCount', 0)} righe")
        return result

    def get_manutenzioni(
        self,
        data_start: Optional[date] = None,
        data_stop: Optional[date] = None,
        output_format: str = "csv"
    ) -> List[Dict[str, Any]]:
        """
        Recupera l'elenco delle manutenzioni dal database AdHoc.

        Args:
            data_start: Data inizio periodo (default: 01/01/2015)
            data_stop: Data fine periodo (default: oggi)
            output_format: Formato output ("json" o "csv", default: csv)

        Returns:
            Lista di dizionari con i dati delle manutenzioni
        """
        if data_start is None:
            data_start = date(2015, 1, 1)
        if data_stop is None:
            data_stop = date.today()

        # Formato date gg/mm/aaaa come richiesto dall'API
        data_start_str = data_start.strftime("%d/%m/%Y")
        data_stop_str = data_stop.strftime("%d/%m/%Y")

        logger.info(f"AdHoc get_manutenzioni | periodo: {data_start_str} - {data_stop_str}")

        result = self.execute_template(
            "elenco_manutenzioni",
            {
                "dateStart": data_start_str,
                "dateStop": data_stop_str
            },
            output_format=output_format
        )

        data = result.get("data", [])
        logger.info(f"AdHoc get_manutenzioni RESULT: {len(data)} interventi")
        return data

    def health_check(self) -> bool:
        """Verifica la connessione all'API"""
        try:
            # Prova a recuperare un piccolo set di dati (JSON per health check, più veloce)
            result = self.execute_template(
                "elenco_manutenzioni",
                {
                    "dateStart": "01/01/2024",
                    "dateStop": "31/01/2024"
                },
                output_format="json"
            )
            return "data" in result or "rowCount" in result
        except Exception as e:
            logger.error(f"AdHoc health_check FAILED: {e}")
            return False


# Singleton
_connector: Optional[AdHocConnector] = None


def get_adhoc_connector() -> AdHocConnector:
    """Ottiene istanza singleton del connettore AdHoc"""
    global _connector
    if _connector is None:
        _connector = AdHocConnector()
    return _connector
