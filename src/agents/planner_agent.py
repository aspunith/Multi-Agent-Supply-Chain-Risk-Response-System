"""Planner agent: prescribe replenishment.

Turns forecasts + supplier constraints into an order plan via the MILP allocator, and compares
it against a greedy baseline to quantify savings. Emits a ReplenishmentProposal.
"""
from __future__ import annotations

from ..schema import (
    AgentMessage,
    AgentName,
    ConstraintReport,
    ForecastResult,
    MessageType,
    OrderAction,
    ReplenishmentProposal,
)
from ..tools import solve_replenishment
from ..tools.allocator import greedy_baseline


def planner_agent(
    world: dict,
    forecast_msgs: list[AgentMessage],
    allow_stockout: bool = True,
    capacity_override: dict | None = None,
    budget: float | None = None,
    recorder=None,
) -> AgentMessage:
    forecasts: list[ForecastResult] = [m.payload for m in forecast_msgs]
    at_risk_ids = {f.sku_id for f in forecasts}
    demand = {f.sku_id: float(sum(f.point_forecast)) for f in forecasts}

    skus = [r for r in world["skus"].to_dict("records") if r["sku_id"] in at_risk_ids]
    suppliers = world["suppliers"].to_dict("records")

    sol = solve_replenishment(
        skus=skus,
        suppliers=suppliers,
        demand=demand,
        allow_stockout=allow_stockout,
        budget=budget,
        capacity_override=capacity_override,
    )
    base = greedy_baseline(skus, demand)

    actions = [OrderAction(**a) for a in sol["actions"]]
    emergency_cost = sum(a.quantity * a.unit_cost for a in actions if a.is_emergency)
    report = ConstraintReport(
        all_hard_constraints_met=sol["feasible"],
        capacity_ok=sol["feasible"],
        budget_ok=sol["feasible"] if budget is not None else True,
        unmet_demand_units=sol["unmet_demand_units"],
        notes=[f"solver_status={sol['status']}"],
    )
    total_cost = sol["objective"] if sol["feasible"] else float("inf")
    proposal = ReplenishmentProposal(
        feasible=sol["feasible"],
        actions=actions,
        total_cost=total_cost if total_cost != float("inf") else 0.0,
        baseline_cost=base["baseline_cost"],
        cost_delta=round(base["baseline_cost"] - total_cost, 2) if sol["feasible"] else 0.0,
        constraint_report=report,
        max_single_order_cost=sol["max_single_order_cost"],
        emergency_order_cost=round(emergency_cost, 2),
    )
    msg = AgentMessage(
        correlation_id=world["correlation_id"],
        sender=AgentName.PLANNER,
        recipient=AgentName.SUPERVISOR,
        message_type=MessageType.REPLENISHMENT_PROPOSAL,
        payload=proposal,
    )
    if recorder:
        recorder.log_step("planner_agent", "optimize_replenishment",
                          {"feasible": sol["feasible"], "status": sol["status"],
                           "total_cost": proposal.total_cost, "baseline_cost": base["baseline_cost"],
                           "cost_delta": proposal.cost_delta,
                           "emergency_order_cost": proposal.emergency_order_cost,
                           "n_emergency_actions": sum(1 for a in actions if a.is_emergency)})
        recorder.log_message(msg.to_record())
    return msg
