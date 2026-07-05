"""Structured JSON trace recorder with per-step token/cost/latency logging.

Records every agent step and inter-agent message, then exports to
traces/<correlation_id>.json for replay and debugging.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

TRACE_DIR = Path(__file__).resolve().parents[2] / "traces"


@dataclass
class TraceRecorder:
    correlation_id: str
    events: list[dict] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    def log_step(self, agent: str, action: str, detail: dict | None = None,
                 tokens: int = 0, cost_usd: float = 0.0) -> None:
        self.events.append(
            {
                "ts_offset_s": round(time.perf_counter() - self._t0, 4),
                "type": "step",
                "agent": agent,
                "action": action,
                "tokens": tokens,
                "cost_usd": round(cost_usd, 6),
                "detail": detail or {},
            }
        )

    def log_message(self, message: dict) -> None:
        self.events.append({"type": "message", **message})

    def summary(self) -> dict:
        steps = [e for e in self.events if e["type"] == "step"]
        return {
            "correlation_id": self.correlation_id,
            "total_steps": len(steps),
            "total_messages": sum(1 for e in self.events if e["type"] == "message"),
            "total_tokens": sum(e.get("tokens", 0) for e in steps),
            "total_cost_usd": round(sum(e.get("cost_usd", 0.0) for e in steps), 6),
            "wall_time_s": round(time.perf_counter() - self._t0, 4),
        }

    def export(self) -> Path:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / f"{self.correlation_id}.json"
        path.write_text(json.dumps({"summary": self.summary(), "events": self.events}, indent=2))
        return path
