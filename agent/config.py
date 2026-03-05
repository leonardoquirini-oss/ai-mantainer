"""
Configurazione dell'agente di manutenzione predittiva
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import yaml

# Carica variabili d'ambiente da .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


@dataclass
class LLMConfig:
    """Configurazione LLM"""
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-2.5-flash-lite"
    temperature: float = 0.1
    max_tokens: int = 4000
    api_key: Optional[str] = None


@dataclass
class APIConfig:
    """Configurazione per API esterne"""
    base_url: str
    api_key: str
    timeout: float = 60.0


@dataclass
class Settings:
    """Configurazione completa dell'agente"""
    llm: LLMConfig
    adhoc_api: APIConfig
    tir_api: APIConfig

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Settings":
        """Carica configurazione da file YAML"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

        # Default se file non esiste
        llm_data = {}
        adhoc_data = {}
        tir_data = {}
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
                llm_data = data.get("llm", {})
                adhoc_data = data.get("api", {}).get("adhoc", {})
                tir_data = data.get("api", {}).get("tir", {})

        llm_config = LLMConfig(
            provider=llm_data.get("provider", "openrouter"),
            base_url=llm_data.get("base_url", "https://openrouter.ai/api/v1"),
            model=llm_data.get("model", "google/gemini-2.5-flash-lite"),
            temperature=llm_data.get("temperature", 0.1),
            max_tokens=llm_data.get("max_tokens", 4000),
            api_key=os.environ.get("OPENROUTER_API_KEY") or llm_data.get("api_key")
        )

        adhoc_config = APIConfig(
            base_url=adhoc_data.get("base_url", "http://192.168.0.12:9100"),
            api_key=os.environ.get("ADHOC_API_KEY") or adhoc_data.get("api_key", "12909db2-ffaa-4e55-8ef7-20d08bdfffa0"),
            timeout=adhoc_data.get("timeout", 60.0)
        )

        tir_config = APIConfig(
            base_url=tir_data.get("base_url", "http://192.168.0.12:9090"),
            api_key=os.environ.get("TIR_API_KEY") or tir_data.get("api_key", "afd633be-262b-4d62-9089-bfe300a58ffa"),
            timeout=tir_data.get("timeout", 120.0)  # Timeout più lungo per export CSV
        )

        return cls(llm=llm_config, adhoc_api=adhoc_config, tir_api=tir_config)


# Singleton settings
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Ottiene l'istanza singleton delle impostazioni"""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings
