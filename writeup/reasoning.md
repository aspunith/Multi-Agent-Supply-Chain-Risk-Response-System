# Reasoning Spine — Track A × S1

> Working document for the 1–2 page write-up. Every major choice uses the pattern
> **Claim → Named alternative → Rejection reason**. Section numbers map to the challenge's
> required Track-A reasoning questions (#1–#4).

**Name:** _[your name]_ · **Track:** A (Agentic / Multi-Agent AI) · **Scenario:** S1 (Supply Chain Risk & Response)

---

## Guardrail: this is ONE track, not two
The MILP allocator (PuLP) and the demand forecaster are **tools invoked by agents**, not the
deliverable. The deliverable is the *orchestration* — how autonomous agents sense, predict,
prescribe, arbitrate, and escalate. This keeps us cleanly inside Track A and avoids the
penalized "shallow multi-track" pattern. (One sentence in the final PDF, verbatim.)

---

## Section 1 — Problem & Domain
A manufacturer faces three coupled problems: supplier delays, demand volatility, and inventory
imbalance. The hard part is not any single prediction — it is the **decision control flow**: the
system must *decide* when a supplier signal is real, when to re-forecast, when a plan is
trustworthy enough to execute, and when a human must approve. That branching, judgment-laden
flow over heterogeneous sub-problems (unstructured news + structured demand + a constrained
plan) is exactly what an agent system is for.

Why this is a meaningful AI opportunity: it mirrors real prescriptive supply-chain operations
where a bad auto-executed order is expensive and auditability is mandatory.

## Section 2 — Approach & Algorithm Decisions

### #1 Why agentic, not a single LLM or a classical pipeline?
- **Claim:** The control flow itself is uncertain and must be *reasoned about*, so we need agents
  with dynamic routing and a human-escalation seam.
- **Alt A — one LLM with a long prompt:** rejected — no genuine role/tool separation, no
  observable inter-agent messages, and it collapses sense/predict/prescribe/arbitrate into one
  opaque step that we cannot audit when it goes wrong.
- **Alt B — static classical pipeline (ETL → forecast → optimizer):** rejected — hard-codes the
  control flow. It cannot reason over unstructured supplier news, cannot decide *when* to
  escalate, and has no natural human checkpoint. It also can't reconcile conflicting signals.
- **Kept:** four agents + a Supervisor that arbitrates and escalates.

### #2 Why LangGraph over CrewAI / AutoGen?
- **Claim:** We need an explicit, auditable state machine with a first-class HITL branch.
- **Alt — CrewAI:** faster to prototype, but abstracts away the control flow we must make
  observable; the interrupt/approval seam is less explicit.
- **Alt — AutoGen:** great for free-form agent chat; our flow is a *structured pipeline with
  typed hand-offs*, not open-ended conversation — we don't want emergent chatter or nondeterminism.
- **Trade-off accepted:** more boilerplate than CrewAI's role abstraction, in exchange for
  traceability. Auditable transitions matter more than prototyping speed for supply decisions.

### #3 Why this decomposition (4 agents), not fewer/more?
Each agent maps to a distinct problem sub-structure, tool, and failure mode:
| Agent | Sub-problem | Tool | Distinct failure mode |
|---|---|---|---|
| Signal | sense | TF-IDF retrieval + lead-time anomaly | false positive / missed disruption |
| Forecast | predict | seasonal-trend forecaster | high forecast error under regime change |
| Planner | prescribe | MILP allocator (PuLP/CBC) | infeasible / high-cost plan |
| Supervisor | arbitrate | reconciliation + thresholds | wrong escalate/auto decision |
- **Fewer (merge Signal+Forecast):** rejected — loses single-responsibility and the ability to
  attribute a failure to sensing vs. prediction in the trace.
- **More (split Planner into buy/allocate):** rejected — adds coordination latency and token cost
  for no reasoning gain at this problem size. **More agents = more tokens.**

### #4 Where could it fail silently, and how do we detect/mitigate?
- **Hallucinated / invalid tool call:** every tool output is validated against a Pydantic model;
  malformed output is rejected rather than trusted.
- **Conflicting agent outputs:** the Supervisor reconciles with explicit confidence/feasibility
  thresholds and **escalates rather than picks** when checks trip.
- **Stale data → confident-but-wrong plan:** freshness assumptions documented; a human checkpoint
  gates any high-cost action ($ threshold) so a wrong plan can't auto-execute.
- **Observability:** `traces/<run>.json` records every step and message with token/cost/latency,
  so post-hoc we can see exactly what each agent did and why.

## Section 3 — Results & Error Analysis _(fill after runs)_
- Success path: disruption detected → SKUs re-forecast → feasible plan (cost vs. greedy baseline
  = **[cost_delta]**, **[%]** savings) → Supervisor escalates on $-threshold → human **approves** →
  executed. Trace: `traces/run-success.json`.
- Failure/edge path: broad supply shock + no-stockout policy → optimizer **infeasible** →
  Supervisor escalates `infeasible_plan` → human **rejects** (hold & renegotiate). Trace:
  `traces/run-failure.json`.
- Honest failure analysis: forecast error propagates into the optimizer; sensitivity of the plan
  to ±X% forecast error = **[measure and report]**. Signal agent precision/recall vs. the injected
  ground-truth disruption = **[report]**.

## Section 4 — Production & Limitations
- **Production consideration:** fresh data arrives on a schedule/event; a re-solve is triggered by
  a new RiskAlert or a demand-forecast drift alarm. Agents are deployed behind FastAPI on
  container infra; blue-green deploy lets us update an agent (e.g. a better forecaster) without
  taking the system offline.
- **Limitation:** the world is synthetic (single independent disruption; no promotions/holidays/
  cross-SKU effects). Before real deployment we would calibrate on historical supplier and demand
  data and add per-SKU supplier eligibility to the optimizer.

---

## Video outline (5 min, decisions not code)
1. **(1:00)** Problem + why agentic (the control-flow argument, #1).
2. **(2:00)** Key decisions & what I ruled out (LangGraph vs CrewAI/AutoGen; 4-agent decomposition; MILP-as-tool).
3. **(1:30)** Results + what didn't work (baseline savings; the infeasible edge case; forecast-error sensitivity).
4. **(0:30)** With more time (async event-driven Supervisor; LLM-as-judge eval on plan quality; per-SKU sourcing constraints).
