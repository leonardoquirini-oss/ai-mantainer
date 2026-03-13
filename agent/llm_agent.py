"""
Agente LLM per la manutenzione predittiva
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional, Generator

from openai import OpenAI

from .config import get_settings, LLMConfig
from .tools.maintenance_tools import TOOLS_SCHEMA, execute_tool

# Configura logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("maintenance-agent")


SYSTEM_PROMPT = """Sei un assistente per la manutenzione predittiva di una flotta di autotrasporti.

IMPORTANTE: Usa SEMPRE i tool disponibili per rispondere alle domande. Non inventare dati.

MODELLI STATISTICI DISPONIBILI:
- Weibull Analysis: classifica guasti in infantili (beta<0.8), casuali (0.8-1.2), usura (beta>1.2)
- Kaplan-Meier: curve di sopravvivenza S(t) = probabilità no guasto entro t mesi
- Cox Proportional Hazards: confronta rischio relativo tra tipi di mezzo (HR>1 = rischio maggiore)
- NHPP Power Law: trend guasti per singolo mezzo (deterioramento/stabile/miglioramento)

TOOL DISPONIBILI:
- carica_dati_csv: carica storico manutenzioni da CSV
- get_statistiche_dataset: statistiche descrittive del dataset
- genera_piano_manutenzione: genera piano manutenzione basato su analisi statistica
- analizza_weibull: analisi Weibull per tipo_mezzo x tipo_guasto
- analizza_sopravvivenza: curva Kaplan-Meier
- analizza_hazard_ratio: confronta rischio tra tipi mezzo (Cox PH)
- analizza_mezzo: analisi NHPP per singolo mezzo
- get_mezzi_critici: lista mezzi in deterioramento
- get_previsioni_guasti: previsioni guasti futuri
- get_fleet_risk_summary: risk score ML di tutta la flotta (filtrabile per azienda)
- get_vehicle_risk: risk score ML dettagliato per singolo veicolo
- get_high_risk_vehicles: veicoli con risk score sopra soglia (default 70)

QUANDO L'UTENTE CHIEDE:
- "carica dati" / "importa" → USA carica_dati_csv
- "statistiche" / "quanti mezzi" → USA get_statistiche_dataset
- "piano manutenzione" / "intervalli" → USA genera_piano_manutenzione
- "analisi weibull" / "tipo guasto" → USA analizza_weibull
- "sopravvivenza" / "curva" → USA analizza_sopravvivenza
- "hazard ratio" / "confronto rischio" → USA analizza_hazard_ratio
- "analizza mezzo X" → USA analizza_mezzo
- "mezzi critici" / "deterioramento" → USA get_mezzi_critici
- "previsioni" / "guasti futuri" → USA get_previsioni_guasti
- "rischio" / "risk score" / "alto rischio" → USA get_fleet_risk_summary o get_high_risk_vehicles
- "rischio di [targa]" / "risk score [targa]" → USA get_vehicle_risk

INTERPRETAZIONE RISULTATI:

Weibull (parametro beta):
- beta < 0.8: Guasti INFANTILI → problemi qualità iniziale, manutenzione preventiva NON efficace
- 0.8 <= beta <= 1.2: Guasti CASUALI → tasso costante, imprevedibili
- beta > 1.2: Guasti da USURA → manutenzione preventiva EFFICACE, calcola intervallo ottimale

NHPP (trend):
- DETERIORAMENTO: guasti sempre più frequenti → attenzione prioritaria, valutare sostituzione
- STABILE: tasso costante → monitoraggio normale
- MIGLIORAMENTO: guasti sempre meno frequenti → situazione positiva

FORMATO RISPOSTE:
- Sii conciso e operativo
- Riporta sempre i parametri statistici (beta, HR, p-value)
- Evidenzia azioni consigliate
- Se l'output contiene tabelle markdown, includile nella risposta

REGOLA CRITICA - OUTPUT DEI TOOL:
Quando un tool restituisce output formattato:
1. COPIA E INCOLLA l'output ESATTAMENTE nella risposta
2. NON riassumere le tabelle - mostra TUTTI i dati
3. PRIMA mostra i dati, POI aggiungi commenti
"""


class MaintenanceAgent:
    """
    Agente LLM per la manutenzione predittiva interattiva
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_settings().llm
        self.client = self._create_client()
        self.messages: List[Dict[str, Any]] = []
        self.reset_conversation()
        logger.info(f"Agent inizializzato - Model: {self.config.model}")

    def _create_client(self) -> OpenAI:
        """Crea il client OpenAI/OpenRouter"""
        api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "API key non trovata. Imposta OPENROUTER_API_KEY "
                "o configura api_key in settings.yaml"
            )

        return OpenAI(
            api_key=api_key,
            base_url=self.config.base_url
        )

    def reset_conversation(self):
        """Resetta la conversazione"""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        logger.info("Conversazione resettata")

    def _process_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """Elabora le chiamate ai tools e restituisce i risultati"""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            args_str = tool_call.function.arguments
            if args_str and args_str.strip():
                arguments = json.loads(args_str)
            else:
                arguments = {}

            # Esegui il tool
            result = execute_tool(tool_name, arguments)

            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": result
            })

        return results

    def chat(self, user_message: str) -> str:
        """
        Invia un messaggio e ottiene una risposta

        Args:
            user_message: Messaggio dell'utente

        Returns:
            Risposta dell'agente
        """
        logger.info(f"USER: {user_message}")

        # Aggiungi messaggio utente
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # Converti TOOLS_SCHEMA nel formato OpenAI
        tools = [
            {"type": "function", "function": tool}
            for tool in TOOLS_SCHEMA
        ]

        # Prima chiamata API
        logger.info(f"LLM call: {self.config.model}")
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=self.messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        assistant_message = response.choices[0].message

        # Loop per gestire chiamate tools multiple
        max_iterations = 10
        iteration = 0
        last_tool_output = ""

        while assistant_message.tool_calls and iteration < max_iterations:
            iteration += 1
            logger.info(f"Tool iteration {iteration}/{max_iterations}")

            # Log delle tool calls
            for tc in assistant_message.tool_calls:
                logger.info(f"TOOL CALL: {tc.function.name}({tc.function.arguments})")

            # Aggiungi risposta assistente con tool calls
            self.messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            # Esegui tools e aggiungi risultati
            tool_results = self._process_tool_calls(assistant_message.tool_calls)
            self.messages.extend(tool_results)

            # Log e salva l'output dei tool
            for tr in tool_results:
                content = tr.get("content", "")
                content_preview = content[:300] + "..." if len(content) > 300 else content
                logger.info(f"TOOL RESULT: {content_preview}")
                if "|" in content and "---" in content:
                    last_tool_output = content

            # Nuova chiamata API con risultati tools
            logger.info("LLM call with tool results")
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=self.messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            assistant_message = response.choices[0].message

        # Aggiungi risposta finale
        final_content = assistant_message.content or ""

        # FALLBACK: Se l'LLM non ha incluso la tabella, aggiungila
        if last_tool_output and "|" not in final_content:
            logger.info("FALLBACK: Aggiunta tabella mancante alla risposta")
            final_content = last_tool_output + "\n\n---\n\n" + final_content

        self.messages.append({
            "role": "assistant",
            "content": final_content
        })

        # Log risposta
        response_preview = final_content[:500] + "..." if len(final_content) > 500 else final_content
        logger.info(f"RESPONSE: {response_preview}")

        return final_content

    def get_riepilogo(self) -> str:
        """Ottiene un riepilogo dello stato del dataset"""
        return self.chat("Dammi un riepilogo del dataset: quanti mezzi, quanti eventi, distribuzione per tipo")


def create_agent() -> MaintenanceAgent:
    """Factory function per creare un agente"""
    return MaintenanceAgent()
