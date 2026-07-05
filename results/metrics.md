# Evaluation Results

## 1. Signal detection (vs ground-truth disruption)
- Precision **1.0**, Recall **1.0**, F1 **1.0** (threshold 0.3)
- Confusion: {'tp': 1, 'fp': 0, 'fn': 0, 'tn': 4}

| supplier | risk_score | predicted | actual |
|---|---|---|---|
| SUP-00 | 0.0 | False | False |
| SUP-01 | 0.0 | False | False |
| SUP-02 | 0.0 | False | False |
| SUP-03 | 0.95 | True | True |
| SUP-04 | 0.0 | False | False |

> Note: the synthetic disruption signal is clean, so detection is near-perfect. Real signals are noisier; see the write-up for the honest caveat.

## 2. Forecast (held-out last 14 days, no leakage)
- MAPE mean **0.0746**, median 0.0728 over 30 SKUs
- MAE mean 2.8, RMSE mean 3.54
- 80% interval coverage **0.824** (target 0.80)
- At-risk SKU MAPE mean: 0.0779

## 3. Optimization
- MILP objective **109767.68** vs LP relaxation 109722.97 -> optimality gap **0.041%**
- Improvement over greedy baseline: **56.64%** (greedy cost 253168.71)
- Constraint satisfaction: MILP **1.0** vs greedy **0.8** (greedy exceeds capacity on 1 supplier(s))

## 4. Plan sensitivity to forecast error
- Cost elasticity to demand: **2.211** (% cost change per % demand change)

| demand factor | feasible | total cost | orders | unmet |
|---|---|---|---|---|
| 0.8 | True | 72971.31 | 13 | 0.0 |
| 0.9 | True | 88720.1 | 12 | 0.0 |
| 1.0 | True | 109767.68 | 13 | 405.44 |
| 1.1 | True | 137253.33 | 11 | 817.08 |
| 1.2 | True | 169791.48 | 10 | 1228.73 |
