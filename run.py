"""Entry point for the supply-chain risk & response system.

Usage:
    python run.py --scenario both      run success + failure scenarios (default)
    python run.py --scenario success   run one scenario
    python run.py --show-schema        print example message payloads
    python run.py --evaluate           run the evaluation harness -> results/
"""
from __future__ import annotations

import argparse
import json

from src.orchestrator.graph import run_scenario
from src.schema import example_messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Supply-chain risk & response system.")
    parser.add_argument("--scenario", choices=["success", "failure", "both"], default="both")
    parser.add_argument("--show-schema", action="store_true", help="print example message payloads and exit")
    parser.add_argument("--evaluate", action="store_true", help="run the evaluation harness and exit")
    args = parser.parse_args()

    if args.show_schema:
        for msg in example_messages():
            print(json.dumps(msg.to_record(), indent=2))
            print("-" * 60)
        return

    if args.evaluate:
        from src.evaluation import run_all

        out = run_all()
        r = out["results"]
        print("Signal   P/R/F1:", r["signal"]["precision"], r["signal"]["recall"], r["signal"]["f1"])
        print("Forecast MAPE mean:", r["forecast"]["aggregate"]["mape_mean"],
              "| 80% coverage:", r["forecast"]["aggregate"]["coverage_80_mean"])
        print("Optim    gap%:", r["optimization"]["optimality_gap_pct"],
              "| vs greedy%:", r["optimization"]["objective_improvement_vs_greedy_pct"])
        print("Wrote:", out["json"], "and", out["markdown"])
        return

    modes = ["success", "failure"] if args.scenario == "both" else [args.scenario]
    for mode in modes:
        res = run_scenario(mode)
        print(f"\n=== Scenario: {mode} ===")
        for k, v in res.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
