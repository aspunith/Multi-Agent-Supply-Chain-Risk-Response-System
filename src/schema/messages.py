"""Inter-agent message schema.

Every hand-off between agents is an ``AgentMessage`` whose ``payload`` is one of five typed
models. Messages are serialized into traces/*.json so a run can be replayed and audited.

Flow: Signal (RiskAlert) -> Forecast (ForecastResult) -> Planner (ReplenishmentProposal)
      -> Supervisor (EscalationRequest) -> Human (HumanDecision) -> execute | abort
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


def _uid() -> str:
    return uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------
class MessageType(str, Enum):
    RISK_ALERT = "risk_alert"
    FORECAST_RESULT = "forecast_result"
    REPLENISHMENT_PROPOSAL = "replenishment_proposal"
    ESCALATION_REQUEST = "escalation_request"
    HUMAN_DECISION = "human_decision"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentName(str, Enum):
    SIGNAL = "signal_agent"
    FORECAST = "forecast_agent"
    PLANNER = "planner_agent"
    SUPERVISOR = "supervisor"
    HUMAN = "human"


class DecisionType(str, Enum):
    APPROVE = "approve"
    OVERRIDE = "override"
    REJECT = "reject"


# --------------------------------------------------------------------------------------
# Payloads — one per message type
# --------------------------------------------------------------------------------------
class RiskSignal(BaseModel):
    """A single observable that contributed to a risk assessment."""

    source: str = Field(description="e.g. 'lead_time_monitor', 'supplier_news'")
    description: str
    metric: Optional[float] = Field(default=None, description="observed value, if numeric")
    threshold: Optional[float] = Field(default=None, description="alerting threshold, if any")


class RiskAlert(BaseModel):
    """Emitted by the Signal agent when a supplier disruption is suspected."""

    supplier_id: str
    affected_skus: list[str]
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0, description="0=no risk, 1=certain disruption")
    signals: list[RiskSignal]
    rationale: str


class ForecastResult(BaseModel):
    """Emitted by the Forecast agent for one at-risk SKU."""

    sku_id: str
    horizon_days: int
    point_forecast: list[float] = Field(description="per-period expected demand")
    lower_80: list[float] = Field(description="10th percentile band")
    upper_80: list[float] = Field(description="90th percentile band")
    method: str = Field(description="e.g. 'seasonal_trend', 'ets'")
    backtest_mape: float = Field(description="rolling one-step MAPE on held-out tail")
    confidence: float = Field(ge=0.0, le=1.0, description="derived from backtest error")


class OrderAction(BaseModel):
    """A single line item in a replenishment proposal."""

    sku_id: str
    supplier_id: str
    quantity: float = Field(ge=0.0)
    unit_cost: float
    is_emergency: bool = False


class ConstraintReport(BaseModel):
    """Solver diagnostics — feeds the Supervisor's trust decision."""

    all_hard_constraints_met: bool
    capacity_ok: bool
    budget_ok: bool
    unmet_demand_units: float = 0.0
    notes: list[str] = Field(default_factory=list)


class ReplenishmentProposal(BaseModel):
    """Emitted by the Planner agent — the prescriptive action set."""

    proposal_id: str = Field(default_factory=_uid)
    feasible: bool
    actions: list[OrderAction]
    total_cost: float
    baseline_cost: float = Field(description="naive fixed-reorder policy cost")
    cost_delta: float = Field(description="baseline_cost - total_cost (positive = savings)")
    constraint_report: ConstraintReport
    max_single_order_cost: float = Field(description="largest single order line cost")
    emergency_order_cost: float = Field(
        default=0.0, description="total spend on emergency (non-primary supplier) orders; drives HITL"
    )


class EscalationRequest(BaseModel):
    """Emitted by the Supervisor when confidence is too low to auto-execute."""

    reason: str
    severity: RiskLevel
    triggers: list[str] = Field(description="which checks tripped, e.g. ['infeasible_plan']")
    requires_human: bool = True


class HumanDecision(BaseModel):
    """The human-in-the-loop response. Closes the loop before any high-cost action executes."""

    decision: DecisionType
    decided_by: str = "operator"
    overrides: dict = Field(default_factory=dict, description="field-level overrides if any")
    notes: str = ""


Payload = Union[
    RiskAlert,
    ForecastResult,
    ReplenishmentProposal,
    EscalationRequest,
    HumanDecision,
]


# --------------------------------------------------------------------------------------
# Envelope
# --------------------------------------------------------------------------------------
class AgentMessage(BaseModel):
    """The single envelope type exchanged between agents. Serialized into traces/*.json."""

    message_id: str = Field(default_factory=_uid)
    correlation_id: str = Field(description="ties every message in one run together")
    timestamp: datetime = Field(default_factory=_now)
    sender: AgentName
    recipient: AgentName
    message_type: MessageType
    payload: Payload

    def to_record(self) -> dict:
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------------------
# Example payloads
# --------------------------------------------------------------------------------------
def example_messages() -> list[AgentMessage]:
    cid = "run-demo-0001"
    risk = AgentMessage(
        correlation_id=cid,
        sender=AgentName.SIGNAL,
        recipient=AgentName.FORECAST,
        message_type=MessageType.RISK_ALERT,
        payload=RiskAlert(
            supplier_id="SUP-03",
            affected_skus=["SKU-014", "SKU-027"],
            risk_level=RiskLevel.HIGH,
            risk_score=0.82,
            signals=[
                RiskSignal(
                    source="lead_time_monitor",
                    description="Lead time spiked to 21d vs 7d baseline (z=3.4)",
                    metric=21.0,
                    threshold=13.0,
                ),
                RiskSignal(
                    source="supplier_news",
                    description="Report mentions port congestion at supplier origin",
                ),
            ],
            rationale="Lead-time anomaly corroborated by unstructured news signal.",
        ),
    )
    forecast = AgentMessage(
        correlation_id=cid,
        sender=AgentName.FORECAST,
        recipient=AgentName.PLANNER,
        message_type=MessageType.FORECAST_RESULT,
        payload=ForecastResult(
            sku_id="SKU-014",
            horizon_days=14,
            point_forecast=[42.0, 44.0],
            lower_80=[35.0, 36.0],
            upper_80=[49.0, 52.0],
            method="seasonal_trend",
            backtest_mape=0.11,
            confidence=0.78,
        ),
    )
    escalation = AgentMessage(
        correlation_id=cid,
        sender=AgentName.SUPERVISOR,
        recipient=AgentName.HUMAN,
        message_type=MessageType.ESCALATION_REQUEST,
        payload=EscalationRequest(
            reason="Emergency order exceeds $25k auto-approval threshold.",
            severity=RiskLevel.HIGH,
            triggers=["cost_threshold_exceeded"],
        ),
    )
    decision = AgentMessage(
        correlation_id=cid,
        sender=AgentName.HUMAN,
        recipient=AgentName.SUPERVISOR,
        message_type=MessageType.HUMAN_DECISION,
        payload=HumanDecision(
            decision=DecisionType.APPROVE,
            decided_by="supply_ops_lead",
            notes="Approved; disruption confirmed with supplier by phone.",
        ),
    )
    return [risk, forecast, escalation, decision]


if __name__ == "__main__":
    import json

    for msg in example_messages():
        print(json.dumps(msg.to_record(), indent=2))
        print("-" * 60)
