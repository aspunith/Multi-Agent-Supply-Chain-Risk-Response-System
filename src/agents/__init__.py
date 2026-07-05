"""The four agents. Rule-based by default; each owns one sub-problem and one tool.
LLM reasoning plugs in via `src.config.get_llm()` without changing the message contracts.
"""
from .signal_agent import signal_agent  # noqa: F401
from .signal_agent import assess_suppliers  # noqa: F401
from .forecast_agent import forecast_agent  # noqa: F401
from .planner_agent import planner_agent  # noqa: F401
from .supervisor import supervisor_agent, human_checkpoint  # noqa: F401
