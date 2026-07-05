"""Generate a synthetic supply-chain world.

Real supplier terms (MOQ, lead-time variance, reliability) aren't available in public datasets,
so we synthesize them with seeded distributions for reproducibility. Outputs (data/generated/):
    suppliers.csv       cost, lead time, reliability, capacity, MOQ
    skus.csv            costs, initial inventory, primary supplier
    demand_history.csv  date, sku_id, demand (trend + weekly seasonality + noise)
    supplier_news.csv   short text corpus for the Signal agent's retrieval
    scenario.json       the injected disruption (supplier, timing, magnitude)

Assumptions: demand = max(0, trend + weekly seasonality + Normal noise); reliability ~ Beta(8,2);
one supplier's lead time is tripled to simulate a disruption. Real demand also has promotions,
holidays and cross-SKU effects we don't model, so treat the numbers as illustrative.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "generated"

N_SUPPLIERS = 5
N_SKUS = 30
HISTORY_DAYS = 365
DISRUPTED_SUPPLIER = "SUP-03"
DISRUPTION_START_DAY = HISTORY_DAYS - 21  # recent event, still visible in the tail
DISRUPTION_LEAD_TIME_MULT = 3.0


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def make_suppliers(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(N_SUPPLIERS):
        sid = f"SUP-{i:02d}"
        reliability = float(np.clip(rng.beta(8, 2), 0.5, 0.99))
        rows.append(
            {
                "supplier_id": sid,
                "name": f"Supplier {i}",
                "reliability": round(reliability, 3),
                "base_lead_time_days": int(rng.integers(5, 12)),
                "lead_time_std": round(float(rng.uniform(0.5, 2.5)), 2),
                "unit_cost_multiplier": round(float(rng.uniform(0.9, 1.4)), 3),
                "capacity_per_period": int(rng.integers(400, 900)),
                "moq": int(rng.choice([0, 25, 50, 100])),
            }
        )
    return pd.DataFrame(rows)


def make_skus(rng: np.random.Generator, suppliers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(N_SKUS):
        base_cost = float(rng.uniform(5, 60))
        primary = suppliers.sample(1, random_state=int(rng.integers(0, 1e6))).iloc[0]["supplier_id"]
        rows.append(
            {
                "sku_id": f"SKU-{i:03d}",
                "name": f"Part {i}",
                "unit_cost": round(base_cost, 2),
                "holding_cost_per_unit_period": round(base_cost * 0.02, 3),
                # penalty set above unit cost (lost margin + goodwill) so ordering is worthwhile
                "stockout_penalty_per_unit": round(base_cost * float(rng.uniform(2.5, 4.0)), 2),
                "initial_inventory": int(rng.integers(20, 200)),
                "primary_supplier": primary,
            }
        )
    return pd.DataFrame(rows)


def make_demand_history(rng: np.random.Generator, skus: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp("2026-06-30"), periods=HISTORY_DAYS, freq="D")
    t = np.arange(HISTORY_DAYS)
    records = []
    for _, sku in skus.iterrows():
        level = rng.uniform(20, 60)
        trend = rng.uniform(-0.02, 0.05)
        weekly = rng.uniform(2, 10)
        sigma = rng.uniform(2, 6)
        seasonal = weekly * np.sin(2 * np.pi * (t % 7) / 7.0)
        mean = level + trend * t + seasonal
        demand = np.maximum(0, rng.normal(mean, sigma)).round().astype(int)
        for d, q in zip(dates, demand):
            records.append({"date": d.date().isoformat(), "sku_id": sku["sku_id"], "demand": int(q)})
    return pd.DataFrame(records)


def make_supplier_news(rng: np.random.Generator, suppliers: pd.DataFrame) -> pd.DataFrame:
    """Small unstructured corpus. One doc encodes the injected disruption for retrieval."""
    templates_ok = [
        "{name} reports on-time delivery performance holding steady this quarter.",
        "{name} completed a capacity expansion, improving throughput.",
        "No material operational changes reported for {name}.",
        "{name} maintains stable pricing and standard lead times.",
    ]
    rows = []
    for _, s in suppliers.iterrows():
        if s["supplier_id"] == DISRUPTED_SUPPLIER:
            text = (
                f"{s['name']} is experiencing shipment delays due to port congestion and a labor "
                f"shortage at its primary origin. Customers should expect lead times roughly "
                f"three times normal for the next several weeks."
            )
        else:
            text = rng.choice(templates_ok).format(name=s["name"])
        rows.append({"supplier_id": s["supplier_id"], "text": text})
    return pd.DataFrame(rows)


def apply_disruption(demand: pd.DataFrame, skus: pd.DataFrame) -> dict:
    """Record the ground-truth disruption so the Signal agent can be evaluated against it."""
    affected = skus.loc[skus["primary_supplier"] == DISRUPTED_SUPPLIER, "sku_id"].tolist()
    return {
        "disrupted_supplier": DISRUPTED_SUPPLIER,
        "start_day_index": DISRUPTION_START_DAY,
        "lead_time_multiplier": DISRUPTION_LEAD_TIME_MULT,
        "affected_skus": affected,
        "description": "Port congestion + labor shortage tripled lead time on SUP-03.",
    }


def main() -> None:
    rng = _rng()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    suppliers = make_suppliers(rng)
    skus = make_skus(rng, suppliers)
    demand = make_demand_history(rng, skus)
    news = make_supplier_news(rng, suppliers)
    scenario = apply_disruption(demand, skus)

    suppliers.to_csv(OUT_DIR / "suppliers.csv", index=False)
    skus.to_csv(OUT_DIR / "skus.csv", index=False)
    demand.to_csv(OUT_DIR / "demand_history.csv", index=False)
    news.to_csv(OUT_DIR / "supplier_news.csv", index=False)
    (OUT_DIR / "scenario.json").write_text(json.dumps(scenario, indent=2))

    print(f"Wrote synthetic data to {OUT_DIR}")
    print(f"  suppliers={len(suppliers)}  skus={len(skus)}  demand_rows={len(demand)}")
    print(f"  disrupted supplier={scenario['disrupted_supplier']} "
          f"affecting {len(scenario['affected_skus'])} SKUs")


if __name__ == "__main__":
    main()
