# LLM Evaluation & Testing Guide

Production LLM systems fail in subtle ways: regressions in prompts, retrieval drift, brittle tool use, and “looks good” outputs that don’t meet real requirements. This guide focuses on **repeatable evaluation** you can run locally and in CI.

## Table of Contents

1. [What to Measure](#what-to-measure)
2. [Build a Golden Dataset](#build-a-golden-dataset)
3. [Offline Evaluation Harness (Python)](#offline-evaluation-harness-python)
4. [LLM-as-Judge (When and How)](#llm-as-judge-when-and-how)
5. [RAG Evaluation](#rag-evaluation)
6. [Tool/Agent Evaluation](#toolagent-evaluation)
7. [Regression Testing in CI](#regression-testing-in-ci)
8. [Reporting and Decision Thresholds](#reporting-and-decision-thresholds)
9. [Evaluation Checklist](#evaluation-checklist)

---

## What to Measure

Choose metrics based on the product behavior you care about:

- **Correctness**: factual accuracy, policy compliance, deterministic requirements.
- **Groundedness**: answers supported by retrieved sources.
- **Safety**: refusal and safe completion quality; prompt injection resilience.
- **Reliability**: variance across runs; sensitivity to prompt changes.
- **Latency / Cost**: p50/p95 latency, tokens, and $ per request.
- **UX quality**: structure, completeness, tone, citations.

Avoid a single “overall score” early on. Track **multiple narrow metrics**.

---

## Build a Golden Dataset

A good evaluation set is:

- **Representative**: matches real traffic distribution.
- **Versioned**: stored as data files in git (or a dataset registry).
- **Labeled**: includes expected behavior (answers, tags, acceptance rules).
- **Small but meaningful**: start with 50–200 cases; scale later.

Recommended schema:

```json
{
  "id": "support_refund_policy_001",
  "input": {
    "query": "Can I get a refund after 45 days?",
    "context": "...optional retrieved docs..."
  },
  "expected": {
    "must_include": ["30 days"],
    "must_not_include": ["guaranteed refund"],
    "tags": ["policy", "refunds"]
  }
}
```

---

## Offline Evaluation Harness (Python)

A minimal harness can run model calls (or stubs) and compute simple rule-based checks.

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    case_id: str
    query: str
    expected_must_include: list[str]
    expected_must_not_include: list[str]


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in raw:
        cases.append(
            EvalCase(
                case_id=item["id"],
                query=item["input"]["query"],
                expected_must_include=item.get("expected", {}).get("must_include", []),
                expected_must_not_include=item.get("expected", {}).get("must_not_include", []),
            )
        )
    return cases


def simple_rule_score(output_text: str, case: EvalCase) -> tuple[bool, dict[str, Any]]:
    text = output_text.lower()

    missing = [s for s in case.expected_must_include if s.lower() not in text]
    forbidden = [s for s in case.expected_must_not_include if s.lower() in text]

    ok = (len(missing) == 0) and (len(forbidden) == 0)
    details = {"missing": missing, "forbidden": forbidden}
    return ok, details


def evaluate(cases: list[EvalCase], generate_fn) -> dict[str, Any]:
    results = []
    for c in cases:
        out = generate_fn(c.query)
        ok, details = simple_rule_score(out, c)
        results.append({"id": c.case_id, "ok": ok, "details": details})

    passed = sum(1 for r in results if r["ok"])
    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": (passed / max(len(results), 1)),
        "results": results,
    }
```

This is intentionally simple: start with rule-based checks, then add richer scoring.

---

## LLM-as-Judge (When and How)

LLM-as-judge can help with rubric-based checks (helpfulness, tone, completeness), but it’s risky if used blindly.

Use it when:
- You have **clear rubrics** and stable prompts for the judge.
- You can run **pairwise comparisons** (A vs B) rather than absolute scores.

Avoid it when:
- The task needs strict factual correctness.
- You lack a baseline of human-labeled cases.

Recommended safeguards:
- Use **fixed judge prompts** with versioning.
- Use **multiple judges** (or multiple seeds) for variance.
- Require **human review** when scores are near threshold.

---

## RAG Evaluation

Evaluate RAG in two layers:

1) **Retrieval quality**
- *Recall@k*: does the correct chunk appear in top-k?
- *MRR*: how high does it rank?

2) **Answer groundedness**
- Citations present and accurate
- Answer claims supported by retrieved text

Practical tests:
- “No-answer” cases (question not in docs) must refuse or ask clarifying questions.
- Contradiction cases (two docs disagree) must surface uncertainty.

---

## Tool/Agent Evaluation

Agentic systems need behavioral tests:

- **Tool selection**: chooses correct tool for the task
- **Tool safety**: never calls destructive actions without explicit confirmation
- **State handling**: doesn’t lose critical context
- **Budget constraints**: caps steps, time, and spend

A simple approach:
- Store expected tool call sequences (or constraints like “must not call delete”).
- Validate tool args against allow-lists.

---

## Regression Testing in CI

Treat prompts as code:

- Run evaluations on every PR affecting prompts, retrieval logic, or policies.
- Gate merges on minimum pass-rate and max cost/latency thresholds.
- Save artifacts: JSON report, aggregate metrics, failing cases.

If you already use GitHub Actions, wire this into a job that runs a Python evaluation script.

---

## Reporting and Decision Thresholds

Avoid “green once = ship forever”. Use:

- **Minimum pass rate** (e.g., 0.85) on key slices
- **No critical regressions** on safety/policy cases
- **Budget guardrails** (e.g., p95 latency < X, average cost < Y)

Track results over time.

---

## Evaluation Checklist

- [ ] Golden dataset exists and is versioned
- [ ] Includes negative/no-answer cases
- [ ] Includes prompt-injection cases
- [ ] RAG eval covers retrieval and groundedness
- [ ] CI blocks regressions on critical slices
- [ ] Reports saved as artifacts and reviewed regularly

---

## Related Guides

- [LLM Operations Guide](llm-operations-guide.md) – production patterns, monitoring, security
- [LangChain Ecosystem Guide](langchain/langchain-ecosystem-guide.md) – chains, tools, agents
- [ML Testing Guide](../ml-engineering/ml-testing-guide.md) – broader ML testing patterns
