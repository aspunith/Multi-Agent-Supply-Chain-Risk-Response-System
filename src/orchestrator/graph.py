"""LangGraph orchestrator: Signal -> Forecast -> Planner -> Supervisor -> {Human | Execute}.

A graph keeps every transition explicit and gives a clean human-in-the-loop branch, which
matters for auditable supply decisions. Node functions are kept pure so they can be tested in
isolation. Runs offline (rule-based agents) by default; set USE_LLM=true to route agent
reasoning through OpenAI without touching the graph or the message contracts.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from ..agents import forecast_agent, human_checkpoint, planner_agent, signal_agent, supervisor_agent
from ..config import load_world
from ..observability import TraceRecorder


class SCState(TypedDict, total=False):
    correlation_id: str
    world: dict
    recorder: Any
    scenario_mode: str  # "success" | "failure"
    risk_msg: Any
    forecast_msgs: list
    proposal_msg: Any
    route: str
    escalation_msg: Optional[Any]
    human_msg: Optional[Any]
    status: str


# --------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------
def node_signal(state: SCState) -> SCState:
    rng = np.random.default_rng(7)
    msg = signal_agent(state["world"], recorder=state["recorder"], rng=rng)
    return {"risk_msg": msg}


def node_forecast(state: SCState) -> SCState:
    msgs = forecast_agent(state["world"], state["risk_msg"], recorder=state["recorder"])
    return {"forecast_msgs": msgs}


def node_planner(state: SCState) -> SCState:
    world = state["world"]
    suppliers = world["suppliers"].to_dict("records")
    disrupted = world["scenario"]["disrupted_supplier"]
    if state.get("scenario_mode") == "failure":
        # Broad supply shock + no-stockout policy -> deliberately infeasible (edge/failure path).
        capacity_override = {s["supplier_id"]: max(1, int(s["capacity_per_period"] * 0.1)) for s in suppliers}
        msg = planner_agent(world, state["forecast_msgs"], allow_stockout=False,
                            capacity_override=capacity_override, recorder=state["recorder"])
    else:
        # Success path: the DETECTED disruption reduces the disrupted supplier's effective
        # capacity (3x lead time ~ ~30% throughput), forcing the optimizer to reroute to
        # alternate suppliers = emergency spend = a human-approval checkpoint.
        cap = next(s["capacity_per_period"] for s in suppliers if s["supplier_id"] == disrupted)
        capacity_override = {disrupted: max(1, int(cap * 0.3))}
        msg = planner_agent(world, state["forecast_msgs"], allow_stockout=True,
                            capacity_override=capacity_override, recorder=state["recorder"])
    return {"proposal_msg": msg}


def node_supervisor(state: SCState) -> SCState:
    route, esc = supervisor_agent(state["world"], state["proposal_msg"], state["forecast_msgs"],
                                  recorder=state["recorder"])
    return {"route": route, "escalation_msg": esc}


def node_human(state: SCState) -> SCState:
    msg = human_checkpoint(state["world"], state["escalation_msg"], recorder=state["recorder"])
    decision = msg.payload.decision.value
    status = "executed" if decision == "approve" else ("executed_with_overrides"
                                                        if decision == "override" else "aborted")
    return {"human_msg": msg, "status": status}


def node_execute(state: SCState) -> SCState:
    if state["recorder"]:
        state["recorder"].log_step("supervisor", "auto_execute", {"status": "executed_no_hitl"})
    return {"status": "executed_no_hitl"}


def _route_after_supervisor(state: SCState) -> str:
    return "human" if state["route"] == "escalate" else "execute"


# --------------------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------------------
def build_graph():
    g = StateGraph(SCState)
    g.add_node("signal", node_signal)
    g.add_node("forecast", node_forecast)
    g.add_node("planner", node_planner)
    g.add_node("supervisor", node_supervisor)
    g.add_node("human", node_human)
    g.add_node("execute", node_execute)

    g.add_edge(START, "signal")
    g.add_edge("signal", "forecast")
    g.add_edge("forecast", "planner")
    g.add_edge("planner", "supervisor")
    g.add_conditional_edges("supervisor", _route_after_supervisor,
                            {"human": "human", "execute": "execute"})
    g.add_edge("human", END)
    g.add_edge("execute", END)
    return g.compile()


def run_scenario(mode: str = "success") -> dict:
    """Run one end-to-end scenario and export its trace. Returns the final state summary."""
    world = load_world()
    correlation_id = f"run-{mode}"
    world["correlation_id"] = correlation_id
    recorder = TraceRecorder(correlation_id=correlation_id)

    graph = build_graph()
    final = graph.invoke(
        {"correlation_id": correlation_id, "world": world, "recorder": recorder, "scenario_mode": mode}
    )
    path = recorder.export()

    result = {
        "scenario": mode,
        "status": final.get("status"),
        "route": final.get("route"),
        "trace_path": str(path),
        "trace_summary": recorder.summary(),
    }
    if final.get("proposal_msg") is not None:
        p = final["proposal_msg"].payload
        result["plan"] = {
            "feasible": p.feasible,
            "total_cost": p.total_cost,
            "baseline_cost": p.baseline_cost,
            "cost_delta": p.cost_delta,
            "n_actions": len(p.actions),
        }
    return result
