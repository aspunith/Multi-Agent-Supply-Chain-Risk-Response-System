"""Entry point for the supply-chain risk & response system.

Usage:
    python run.py --scenario both      run success + failure scenarios (default)
    python run.py --scenario success   run one scenario
    python run.py --show-schema        print example message payloads
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
    args = parser.parse_args()

    if args.show_schema:
        for msg in example_messages():
            print(json.dumps(msg.to_record(), indent=2))
            print("-" * 60)
        return

    modes = ["success", "failure"] if args.scenario == "both" else [args.scenario]
    for mode in modes:
        res = run_scenario(mode)
        print(f"\n=== Scenario: {mode} ===")
        for k, v in res.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
