"""
Connettore API TIR per recuperare viaggi e km.

Endpoint: http://192.168.0.12:9090
"""

import csv
import io
import httpx
import inspect
import logging
from datetime import date, datetime
from typing import List, Optional, Any, Dict

from ..config import APIConfig, get_settings

logger = logging.getLogger("maintenance-agent.tir")


def _get_caller_name() -> str:
    """Ritorna il nome del metodo chiamante (2 livelli sopra)"""
    stack = inspect.stack()
    if len(stack) >= 3:
        return stack[2].function
    return "unknown"


class TIRConnector:
    """
    Connettore per l'API TIR (gestione viaggi e km).

    Fornisce accesso alla query 'viaggi' che restituisce
    lo storico dei viaggi con km per motrici e semirimorchi.

    Colonne restituite:
    - BG: ID univoco viaggio
    - TargaMotrice: targa della motrice
    - TargaSemirimorchio: targa del semirimorchio
    - Km: km percorsi
    - Data: data del viaggio
    """

    def __init__(self, config: Optional[APIConfig] = None):
        if config is None:
            config = get_settings().tir_api
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
            template_name: Nome del template (es. viaggi)
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
        logger.info(f"TIR API: POST /api/Query/templates/execute | caller: {caller}")
        logger.info(f"TIR TEMPLATE: {template_name} | format: {output_format or 'json'}")

        response = self.client.post(
            "/api/Query/templates/execute",
            json=payload
        )
        response.raise_for_status()

        # Se CSV, parsa il contenuto
        if output_format == "csv":
            return self._parse_csv_response(response.text)

        result = response.json()
        logger.info(f"TIR RESULT: {result.get('rowCount', 0)} righe")
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

        logger.info(f"TIR CSV RESULT: {len(data)} righe (delimiter: '{delimiter}')")
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
        logger.info(f"TIR API: POST /api/Query/execute | caller: {caller}")
        query_preview = query[:200] + "..." if len(query) > 200 else query
        logger.info(f"TIR QUERY: {query_preview}")

        response = self.client.post(
            "/api/Query/execute",
            json={"query": query}
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"TIR RESULT: {result.get('rowCount', 0)} righe")
        return result

    def get_viaggi(
        self,
        data_start: Optional[date] = None,
        data_stop: Optional[date] = None,
        output_format: str = "csv"
    ) -> List[Dict[str, Any]]:
        """
        Recupera l'elenco dei viaggi dal database TIR.

        Args:
            data_start: Data inizio periodo (default: 01/01/2015)
            data_stop: Data fine periodo (default: oggi)
            output_format: Formato output ("json" o "csv", default: csv)

        Returns:
            Lista di dizionari con i dati dei viaggi
        """
        if data_start is None:
            data_start = date(2015, 1, 1)
        if data_stop is None:
            data_stop = date.today()

        # Formato date gg/mm/aaaa come richiesto dall'API
        data_start_str = data_start.strftime("%d/%m/%Y")
        data_stop_str = data_stop.strftime("%d/%m/%Y")

        logger.info(f"TIR get_viaggi | periodo: {data_start_str} - {data_stop_str}")

        result = self.execute_template(
            "viaggi",
            {
                "dateStart": data_start_str,
                "dateStop": data_stop_str
            },
            output_format=output_format
        )

        data = result.get("data", [])
        logger.info(f"TIR get_viaggi RESULT: {len(data)} viaggi")
        return data

    def health_check(self) -> bool:
        """Verifica la connessione all'API"""
        try:
            # Prova a recuperare un piccolo set di dati (ultimo mese)
            today = date.today()
            start = date(today.year, today.month, 1)
            result = self.execute_template(
                "viaggi",
                {
                    "dateStart": start.strftime("%d/%m/%Y"),
                    "dateStop": today.strftime("%d/%m/%Y")
                },
                output_format="json"  # JSON per health check (più veloce)
            )
            return "data" in result or "rowCount" in result
        except Exception as e:
            logger.error(f"TIR health_check FAILED: {e}")
            return False


# Singleton
_connector: Optional[TIRConnector] = None


def get_tir_connector() -> TIRConnector:
    """Ottiene istanza singleton del connettore TIR"""
    global _connector
    if _connector is None:
        _connector = TIRConnector()
    return _connector
