# Retrieval Provenance Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind provenance to exact ordered query strings and exact deterministic retrieval output.

**Architecture:** Hash canonical JSON of the complete query sequence without semantic normalization. At the diagnostic boundary, replay retrieval from the declared case, query sequence, tier limit, character limit, and tier, then require full frozen-result equality before emitting validated recall metrics.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, Ruff

## Global Constraints

- Preserve every query string, query order, duplicate, and token order in query provenance.
- Preserve exact reference, planned, production-query, production-result, tier, and case coverage.
- The high gate accepts recall only from replay-validated high metrics.

---

### Task 1: Exact query-sequence provenance

**Files:**
- Modify: `src/budget_crossover/retrieval.py:45-48`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `retrieval_query_hash(queries: Sequence[str]) -> str`
- Produces: the same function signature with exact-sequence hash semantics

- [x] **Step 1: Write the failing collision regressions**

```python
def test_query_hash_binds_exact_order_duplicates_and_token_order():
    assert retrieval_query_hash(("revenue", "expense")) != retrieval_query_hash(
        ("expense", "revenue")
    )
    assert retrieval_query_hash(("revenue", "revenue")) != retrieval_query_hash(
        ("revenue",)
    )
    assert retrieval_query_hash(("revenue expense",)) != retrieval_query_hash(
        ("expense revenue",)
    )
```

- [x] **Step 2: Run the test to verify RED**

Run: `.venv/bin/pytest -q tests/test_retrieval.py::test_query_hash_binds_exact_order_duplicates_and_token_order`

Expected: all three inequality assertions fail under sorted/deduplicated token-set hashing.

- [x] **Step 3: Hash canonical JSON of the exact sequence**

```python
def retrieval_query_hash(queries: Sequence[str]) -> str:
    payload = json.dumps(tuple(queries), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [x] **Step 4: Run retrieval tests to verify GREEN**

Run: `.venv/bin/pytest -q tests/test_retrieval.py`

Expected: all retrieval tests pass.

### Task 2: Deterministic production replay

**Files:**
- Modify: `src/budget_crossover/diagnostics.py:505-580`
- Test: `tests/test_diagnostics.py`
- Modify: `.superpowers/sdd/2026-08-03-conditional-crossover-rebuild/task-3-report.md`

**Interfaces:**
- Consumes: `retrieve(case, exact_queries, limit, max_chars_per_item, tier_id)`
- Produces: `retrieval_ladder_boundary` metrics only after full replay equality

- [x] **Step 1: Write the failing reconstructed-low-k regression**

```python
limits = {"low": 1, "middle": 2, "high": 2}
forged = low.model_copy(
    update={
        "tier_id": "high",
        "requested_k": 2,
        "query_hash": high.query_hash,
        "input_hash": high.input_hash,
    }
)
with pytest.raises(ValueError, match="high retrieval provenance"):
    retrieval_ladder_boundary(
        adapted.cases,
        reference_queries=queries,
        planned_queries=queries,
        production_queries=production_queries,
        production_results={
            "low": {case_id: low},
            "middle": {case_id: middle},
            "high": {case_id: forged},
        },
        tier_limits=limits,
        max_chars_per_item=1000,
    )
```

- [x] **Step 2: Run the diagnostic test to verify RED**

Run: `.venv/bin/pytest -q tests/test_diagnostics.py::test_retrieval_ladders_reject_forged_low_k_output_with_high_provenance`

Expected: failure because the current boundary checks metadata/hashes but not exact output.

- [x] **Step 3: Replay and compare the complete result**

```python
expected = retrieve(
    case.public,
    production_queries[tier][case_id],
    limit=tier_limits[tier],
    max_chars_per_item=max_chars_per_item,
    tier_id=tier,
)
if result != expected:
    raise ValueError(
        f"{tier} retrieval provenance must match deterministic retrieval replay"
    )
```

- [x] **Step 4: Run focused and final verification**

Run: `.venv/bin/pytest -q tests/test_retrieval.py tests/test_dataset.py tests/test_diagnostics.py tests/test_systems.py tests/test_checking.py`

Run: `.venv/bin/pytest -q`

Run: `.venv/bin/ruff check .`

Run: `git diff --check`

Expected: every command exits zero.

- [x] **Step 5: Append RED/GREEN/final evidence and commit**

Append exact command output to the Task 3 report, stage the scoped diff, and commit with:

```bash
git commit -m "fix: replay retrieval provenance"
```
