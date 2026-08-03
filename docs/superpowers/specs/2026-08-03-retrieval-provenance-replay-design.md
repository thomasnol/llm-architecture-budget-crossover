# Retrieval Provenance Replay Design

## Goal

Close two provenance bypasses in the FinanceComplexQA retrieval diagnostic. Query provenance must
bind the exact ordered query strings, and a production result must be the exact deterministic output
of the declared case, queries, retrieval limit, character limit, and tier before its recall can
participate in the high-tier gate.

## Design

`retrieval_query_hash()` will hash canonical compact JSON of the input query sequence. The
serialization preserves each complete string, query order, duplicates, token order, punctuation,
case, and whitespace. No semantic normalization is applied because provenance describes the exact
retrieval input rather than an equivalence class.

For every production tier and case, `retrieval_ladder_boundary()` will replay `retrieve()` with the
exact public case, exact captured query sequence, configured tier limit, configured
`max_chars_per_item`, and tier name. The submitted `RetrievalResult` must equal the replayed frozen
result in full. This comparison binds retrieved and truncated items, pre- and post-truncation ID
sequences, tier, requested limit, and query/input hashes. Exact query/result/tier/case coverage
continues to fail closed.

Only ladders that pass this replay emit `provenance_validated: true`; therefore the existing high
gate consumes recall only from replay-validated high metrics.

## Error Handling

Any replay mismatch raises the existing tier-specific provenance `ValueError`. The diagnostic does
not repair, relabel, or partially accept a supplied result.

## Tests

- Query hash differs for reordered queries.
- Query hash differs when a duplicate query is removed.
- Query hash differs for `revenue expense` versus `expense revenue`.
- A reconstructed low-k result carrying otherwise valid high-tier metadata and hashes is rejected
  because it differs from the configured high-k replay.
- Existing exact-coverage, stale-query, stale-input, tier/k, recall, and gate tests remain green.
