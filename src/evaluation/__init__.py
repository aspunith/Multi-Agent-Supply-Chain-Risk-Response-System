"""Evaluation harness.

Produces held-out, out-of-sample metrics for each component that feeds the agent system:

  1. Signal detection  precision / recall / F1 of disruption flagging vs ground truth.
  2. Forecast          held-out MAPE / MAE / RMSE (last 14 days per SKU).
  3. Optimization      LP-relaxation optimality gap, constraint satisfaction, savings vs greedy.
  4. Sensitivity       how the optimal plan cost moves under forecast error (+/-20%).

Run:  python -m src.evaluation.evaluate
Writes results/evaluation.json and results/metrics.md.
"""
from .evaluate import run_all  # noqa: F401
