"""Supervisor agent + human-in-the-loop checkpoint.

The Supervisor reconciles agent outputs and applies trust checks before execution:
    - infeasible plan
    - emergency spend over threshold
    - forecast confidence below threshold
    - unmet demand
If any check trips it escalates to the human checkpoint instead of auto-approving.
"""
from __future__ import annotations

from ..config import env_float
from ..schema import (
    AgentMessage,
    AgentName,
    DecisionType,
    EscalationRequest,
    ForecastResult,
    HumanDecision,
    MessageType,
    ReplenishmentProposal,
    RiskLevel,
)


def supervisor_agent(
    world: dict,
    proposal_msg: AgentMessage,
    forecast_msgs: list[AgentMessage],
    recorder=None,
) -> tuple[str, AgentMessage | None]:
    """Return (route, escalation_message|None) where route in {'auto_approve','escalate'}."""
    proposal: ReplenishmentProposal = proposal_msg.payload
    forecasts: list[ForecastResult] = [m.payload for m in forecast_msgs]

    cost_threshold = env_float("EMERGENCY_ORDER_COST_THRESHOLD", 25_000)
    min_conf = env_float("MIN_FORECAST_CONFIDENCE", 0.60)
    min_forecast_conf = min((f.confidence for f in forecasts), default=1.0)

    triggers = []
    if not proposal.feasible:
        triggers.append("infeasible_plan")
    if proposal.constraint_report.unmet_demand_units > 0:
        triggers.append("unmet_demand")
    if proposal.emergency_order_cost > cost_threshold:
        triggers.append("cost_threshold_exceeded")
    if min_forecast_conf < min_conf:
        triggers.append("low_forecast_confidence")

    if not triggers:
        if recorder:
            recorder.log_step("supervisor", "reconcile",
                              {"route": "auto_approve", "min_forecast_conf": round(min_forecast_conf, 3)})
        return "auto_approve", None

    severity = RiskLevel.CRITICAL if "infeasible_plan" in triggers else RiskLevel.HIGH
    reason = {
        "infeasible_plan": "Optimizer returned no feasible plan under current constraints.",
        "unmet_demand": "Plan leaves demand unmet; potential stockout.",
        "cost_threshold_exceeded": f"Emergency (supplier-substitution) spend ${proposal.emergency_order_cost:,.0f} "
                                   f"exceeds ${cost_threshold:,.0f} auto-approval limit.",
        "low_forecast_confidence": f"Forecast confidence {min_forecast_conf:.2f} below {min_conf:.2f} threshold.",
    }[triggers[0]]

    esc = EscalationRequest(reason=reason, severity=severity, triggers=triggers)
    msg = AgentMessage(
        correlation_id=world["correlation_id"],
        sender=AgentName.SUPERVISOR,
        recipient=AgentName.HUMAN,
        message_type=MessageType.ESCALATION_REQUEST,
        payload=esc,
    )
    if recorder:
        recorder.log_step("supervisor", "reconcile", {"route": "escalate", "triggers": triggers})
        recorder.log_message(msg.to_record())
    return "escalate", msg


def human_checkpoint(
    world: dict,
    escalation_msg: AgentMessage,
    decision_provider=None,
    recorder=None,
) -> AgentMessage:
    """HITL: obtain a human decision. `decision_provider(escalation)->HumanDecision` lets us run
    non-interactively for reproducible traces; default auto-approves confirmed disruptions but
    REJECTS on infeasible plans (a human would not execute a plan the solver couldn't build)."""
    esc: EscalationRequest = escalation_msg.payload
    if decision_provider is not None:
        decision = decision_provider(esc)
    elif "infeasible_plan" in esc.triggers:
        decision = HumanDecision(decision=DecisionType.REJECT, decided_by="supply_ops_lead",
                                 notes="Plan infeasible under tightened constraints; hold and renegotiate capacity.")
    else:
        decision = HumanDecision(decision=DecisionType.APPROVE, decided_by="supply_ops_lead",
                                 notes="Disruption confirmed; approve emergency replenishment.")
    msg = AgentMessage(
        correlation_id=world["correlation_id"],
        sender=AgentName.HUMAN,
        recipient=AgentName.SUPERVISOR,
        message_type=MessageType.HUMAN_DECISION,
        payload=decision,
    )
    if recorder:
        recorder.log_step("human", "decision", {"decision": decision.decision.value, "notes": decision.notes})
        recorder.log_message(msg.to_record())
    return msg
