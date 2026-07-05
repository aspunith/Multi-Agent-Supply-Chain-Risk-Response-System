"""Signal agent: sense supplier disruption.

Fuses a numeric lead-time anomaly with a news-retrieval signal and emits a RiskAlert.
"""
from __future__ import annotations

import numpy as np

from ..schema import AgentMessage, AgentName, MessageType, RiskAlert, RiskLevel, RiskSignal
from ..tools import NewsRetriever, lead_time_anomaly


def _synth_recent_lead_times(supplier: dict, disrupted: bool, rng: np.random.Generator) -> list[float]:
    base = supplier["base_lead_time_days"]
    mult = 3.0 if disrupted else 1.0
    return list(np.maximum(1, rng.normal(base * mult, supplier["lead_time_std"], size=7)))


def signal_agent(world: dict, recorder=None, rng=None) -> AgentMessage:
    rng = rng or np.random.default_rng(7)
    suppliers = world["suppliers"].to_dict("records")
    skus = world["skus"].to_dict("records")
    news = world["news"].to_dict("records")
    scenario = world["scenario"]
    retriever = NewsRetriever(news)

    best = None  # (risk_score, supplier, signals, affected)
    for s in suppliers:
        disrupted = s["supplier_id"] == scenario["disrupted_supplier"]
        recent = _synth_recent_lead_times(s, disrupted, rng)
        anomaly = lead_time_anomaly(recent, s["base_lead_time_days"], max(s["lead_time_std"], 0.5))
        news_flag = retriever.disruption_flag(s["supplier_id"])

        # Fuse: numeric anomaly (structured) + keyword hit (unstructured).
        score = 0.0
        signals = []
        if anomaly["is_anomaly"]:
            score += min(0.6, 0.15 * anomaly["z_score"])
            signals.append(RiskSignal(source="lead_time_monitor",
                                      description=f"Lead time {anomaly['recent_mean_lead_time']}d vs "
                                                  f"{anomaly['baseline_mean']}d baseline (z={anomaly['z_score']})",
                                      metric=anomaly["recent_mean_lead_time"],
                                      threshold=anomaly["baseline_mean"] + 2 * max(s["lead_time_std"], 0.5)))
        if news_flag["flagged"]:
            score += 0.35
            signals.append(RiskSignal(source="supplier_news",
                                      description=f"News keywords: {', '.join(news_flag['keyword_hits'])}"))
        if best is None or score > best[0]:
            affected = [k["sku_id"] for k in skus if k["primary_supplier"] == s["supplier_id"]]
            best = (score, s, signals, affected)

    score, sup, signals, affected = best
    score = float(min(1.0, score))
    level = (RiskLevel.CRITICAL if score >= 0.85 else RiskLevel.HIGH if score >= 0.6
             else RiskLevel.MEDIUM if score >= 0.3 else RiskLevel.LOW)

    alert = RiskAlert(
        supplier_id=sup["supplier_id"],
        affected_skus=affected,
        risk_level=level,
        risk_score=round(score, 3),
        signals=signals,
        rationale="Corroborated structured lead-time anomaly with unstructured news signal."
        if len(signals) > 1 else "Single-source signal; monitor for corroboration.",
    )
    msg = AgentMessage(
        correlation_id=world["correlation_id"],
        sender=AgentName.SIGNAL,
        recipient=AgentName.FORECAST,
        message_type=MessageType.RISK_ALERT,
        payload=alert,
    )
    if recorder:
        recorder.log_step("signal_agent", "assess_risk",
                          {"supplier": sup["supplier_id"], "risk_score": round(score, 3), "level": level.value})
        recorder.log_message(msg.to_record())
    return msg
