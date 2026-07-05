# Multi-Agent Supply Chain Risk & Response System

**Avathon AI Intelligence Challenge — Track A (Agentic / Multi-Agent AI) × Scenario S1 (Supply Chain)**

Autonomous agents that **sense** supplier disruption, **re-forecast** affected SKUs,
**prescribe** a constrained replenishment plan, and **escalate** high-impact actions to a
human before execution.

> **One track, not two.** The MILP optimizer and the forecaster are *tools invoked by agents*,
> not the deliverable. The deliverable is the orchestration.

---

## Architecture

```
 Signal ──RiskAlert──▶ Forecast ──ForecastResult(s)──▶ Planner ──ReplenishmentProposal──▶ Supervisor
                                                                                             │
                                                              auto-approve ◀── (all checks pass)
                                                                                             │
                                                              escalate ──▶ Human ──HumanDecision──▶ execute | abort
```

| Agent | Role | Tool | Emits |
|---|---|---|---|
| **Signal** | sense disruption | TF-IDF retrieval + lead-time anomaly | `RiskAlert` |
| **Forecast** | re-forecast at-risk SKUs | seasonal-trend forecaster | `ForecastResult` |
| **Planner** | prescribe replenishment | MILP allocator (PuLP/CBC) | `ReplenishmentProposal` |
| **Supervisor** | reconcile & escalate | threshold/feasibility checks | `EscalationRequest` |
| **Human** | approve / override / reject | HITL checkpoint | `HumanDecision` |

Message schema: [src/schema/messages.py](src/schema/messages.py) (typed Pydantic, with example payloads).

---

## Setup

Requires **Python 3.11+** (validated on 3.13.5).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # optional; runs offline without a key
```

## Reproduce all results (from a clean clone)

```powershell
# 1) Generate the seeded synthetic world (data/generated/)
python data/generate_data.py

# 2) Print the example message payloads
python run.py --show-schema

# 3) Run both end-to-end scenarios; traces export to traces/
python run.py --scenario both

# 4) Run the evaluation harness; metrics export to results/
python run.py --evaluate
```

- `traces/run-success.json` — disruption detected → feasible plan → escalated on cost threshold → human **approves** → executed.
- `traces/run-failure.json` — supply shock + no-stockout → optimizer **infeasible** → escalated → human **rejects**.
- `results/metrics.md` — signal P/R/F1, held-out forecast MAPE + interval coverage, optimizer gap / savings / constraint satisfaction, and plan sensitivity to forecast error.

### Optional: route agent reasoning through OpenAI
Set `USE_LLM=true` and `OPENAI_API_KEY` in `.env`. Defaults to **offline rule-based
agents** (temperature-0 equivalent) so results are byte-reproducible without a key.

---

## Repository layout

```
data/generate_data.py      seeded synthetic world + documented distributions
src/schema/messages.py     typed inter-agent message schema (+ example payloads)
src/tools/                 retrieval, forecaster, MILP allocator (agent tools)
src/agents/                signal, forecast, planner, supervisor + HITL checkpoint
src/orchestrator/graph.py  LangGraph state machine wiring it end-to-end
src/observability/trace.py JSON trace recorder (steps, messages, tokens, cost, latency)
src/evaluation/            held-out metrics: signal, forecast, optimizer, sensitivity
traces/                    exported agent interaction traces (2 scenarios)
results/                   evaluation.json + metrics.md
```

## Results (from `python run.py --evaluate`)

| Component | Metric | Value |
|---|---|---|
| Signal detection | Precision / Recall / F1 | 1.00 / 1.00 / 1.00 (clean synthetic signal) |
| Forecast | Held-out MAPE / 80% coverage | 7.5% / 0.82 |
| Optimizer | LP-relaxation gap | 0.04% |
| Optimizer | Cost vs. greedy baseline | 56.6% lower |
| Optimizer | Constraint satisfaction (MILP vs greedy) | 1.0 vs 0.8 |
| Sensitivity | Plan cost elasticity to demand | ~2.2 |

## Status
Schema, data generator, tools, agents, orchestrator, observability, evaluation harness, and the
reasoning notes are wired end-to-end. Next: the final PDF write-up and walkthrough video.
