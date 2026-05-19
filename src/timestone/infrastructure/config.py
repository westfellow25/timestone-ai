"""Configuration loaded from environment / .env"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    default_iterations: int = 1000
    default_horizon_years: int = 5
    default_discount_rate: float = 0.12
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            default_iterations=int(os.environ.get("TIMESTONE_ITERATIONS", "1000")),
            default_horizon_years=int(os.environ.get("TIMESTONE_HORIZON_YEARS", "5")),
            default_discount_rate=float(os.environ.get("TIMESTONE_DISCOUNT_RATE", "0.12")),
            log_level=os.environ.get("TIMESTONE_LOG_LEVEL", "INFO"),
        )
