"""Single-period constrained replenishment allocator (MILP).

Decision variables
    x[i,s] >= 0        units of SKU i ordered from supplier s   (continuous)
    y[i,s] in {0,1}    order-placed flag for MOQ / fixed cost
    short[i] >= 0      unmet demand for SKU i                   (soft, penalized)
Objective (minimize USD): purchase cost + holding cost + stockout penalty
Constraints
    balance    end_inv[i] = init_inv[i] + sum_s x[i,s] - demand[i] + short[i]
    capacity   sum_i x[i,s] <= capacity[s]
    MOQ        x[i,s] >= moq[s]*y[i,s] ; x[i,s] <= BIGM*y[i,s]
    budget     sum purchase cost <= budget   (optional)

Solved with PuLP/CBC. `short` is soft so the model stays feasible under disruption and reports
unmet demand; the failure scenario hardens it (allow_stockout=False) to force infeasibility.
"""
from __future__ import annotations

import pulp

BIGM = 10_000


def solve_replenishment(
    skus: list[dict],
    suppliers: list[dict],
    demand: dict[str, float],
    allow_stockout: bool = True,
    budget: float | None = None,
    capacity_override: dict[str, float] | None = None,
) -> dict:
    """Solve one replenishment period.

    Args:
        skus: rows with sku_id, unit_cost, holding_cost_per_unit_period,
              stockout_penalty_per_unit, initial_inventory, primary_supplier.
        suppliers: rows with supplier_id, unit_cost_multiplier, capacity_per_period, moq.
        demand: {sku_id: forecast_demand_units} over the horizon.
        allow_stockout: if False, demand must be fully met (can make the model infeasible).
        budget: optional hard cap on purchase cost.
        capacity_override: optional {supplier_id: capacity} to model a disrupted supplier.
    """
    sup_by_id = {s["supplier_id"]: s for s in suppliers}
    prob = pulp.LpProblem("replenishment", pulp.LpMinimize)

    x, y, short = {}, {}, {}
    for i in skus:
        sid = i["sku_id"]
        short[sid] = pulp.LpVariable(f"short_{sid}", lowBound=0)
        for s in suppliers:
            key = (sid, s["supplier_id"])
            x[key] = pulp.LpVariable(f"x_{sid}_{s['supplier_id']}", lowBound=0)
            y[key] = pulp.LpVariable(f"y_{sid}_{s['supplier_id']}", cat="Binary")

    # Objective
    purchase = pulp.lpSum(
        i["unit_cost"] * sup_by_id[s["supplier_id"]]["unit_cost_multiplier"] * x[(i["sku_id"], s["supplier_id"])]
        for i in skus
        for s in suppliers
    )
    holding = pulp.lpSum(
        i["holding_cost_per_unit_period"]
        * (i["initial_inventory"] + pulp.lpSum(x[(i["sku_id"], s["supplier_id"])] for s in suppliers) - demand.get(i["sku_id"], 0) + short[i["sku_id"]])
        for i in skus
    )
    stockout = pulp.lpSum(i["stockout_penalty_per_unit"] * short[i["sku_id"]] for i in skus)
    prob += purchase + holding + stockout

    # Constraints
    for i in skus:
        sid = i["sku_id"]
        received = pulp.lpSum(x[(sid, s["supplier_id"])] for s in suppliers)
        # ending inventory must be >= 0: init + received - demand + short >= 0
        prob += i["initial_inventory"] + received - demand.get(sid, 0) + short[sid] >= 0
        if not allow_stockout:
            prob += short[sid] == 0  # hardened -> can be infeasible under tight capacity

    for s in suppliers:
        sid = s["supplier_id"]
        cap = (capacity_override or {}).get(sid, s["capacity_per_period"])
        prob += pulp.lpSum(x[(i["sku_id"], sid)] for i in skus) <= cap
        for i in skus:
            key = (i["sku_id"], sid)
            moq = s.get("moq", 0)
            if moq > 0:
                prob += x[key] >= moq * y[key]
            prob += x[key] <= BIGM * y[key]

    if budget is not None:
        prob += purchase <= budget

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status_str = pulp.LpStatus[status]
    feasible = status_str == "Optimal"

    actions, max_order_cost, unmet = [], 0.0, 0.0
    if feasible:
        for i in skus:
            for s in suppliers:
                key = (i["sku_id"], s["supplier_id"])
                q = x[key].value() or 0.0
                if q > 1e-6:
                    unit = i["unit_cost"] * sup_by_id[s["supplier_id"]]["unit_cost_multiplier"]
                    line_cost = unit * q
                    max_order_cost = max(max_order_cost, line_cost)
                    actions.append(
                        {
                            "sku_id": i["sku_id"],
                            "supplier_id": s["supplier_id"],
                            "quantity": round(q, 2),
                            "unit_cost": round(unit, 2),
                            "is_emergency": s["supplier_id"] != i["primary_supplier"],
                        }
                    )
            unmet += short[i["sku_id"]].value() or 0.0

    return {
        "feasible": feasible,
        "status": status_str,
        "objective": round(pulp.value(prob.objective), 2) if feasible else None,
        "actions": actions,
        "max_single_order_cost": round(max_order_cost, 2),
        "unmet_demand_units": round(unmet, 2),
    }


def greedy_baseline(skus: list[dict], demand: dict[str, float], reorder_qty: float = 100.0) -> dict:
    """Naive 'always reorder a fixed quantity from the primary supplier' policy."""
    total = 0.0
    for i in skus:
        total += i["unit_cost"] * reorder_qty
        # holding on whatever is left over
        ending = i["initial_inventory"] + reorder_qty - demand.get(i["sku_id"], 0)
        if ending > 0:
            total += i["holding_cost_per_unit_period"] * ending
        else:
            total += i["stockout_penalty_per_unit"] * (-ending)
    return {"baseline_cost": round(total, 2), "policy": f"fixed_reorder_{int(reorder_qty)}"}
