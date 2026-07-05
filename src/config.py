"""Shared config, data loading, and optional LLM helper."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# src/config.py -> parents[0]=src, parents[1]=project root
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "generated"


def env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes"}


def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def load_world() -> dict:
    """Load the generated synthetic world. Raises a clear error if not generated yet."""
    if not (DATA_DIR / "suppliers.csv").exists():
        raise FileNotFoundError(
            f"No generated data at {DATA_DIR}. Run: python data/generate_data.py"
        )
    import json

    return {
        "suppliers": pd.read_csv(DATA_DIR / "suppliers.csv"),
        "skus": pd.read_csv(DATA_DIR / "skus.csv"),
        "demand": pd.read_csv(DATA_DIR / "demand_history.csv"),
        "news": pd.read_csv(DATA_DIR / "supplier_news.csv"),
        "scenario": json.loads((DATA_DIR / "scenario.json").read_text()),
    }


def get_llm():
    """Return a LangChain OpenAI chat model, or None if LLM is disabled.

    Runs fully offline (rule-based agents) when USE_LLM=false so results are reproducible
    without a key. Set USE_LLM=true to route agent reasoning through OpenAI.
    """
    if not env_bool("USE_LLM", False):
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,  # determinism for reproducibility
    )
