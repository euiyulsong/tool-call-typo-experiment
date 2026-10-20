# Elasticsearch term suggester vs BM25

## Design

- 100 paired, heavily typo-corrupted MultiDoc2Dial multi-turn cases (three mixed character operations on one eligible word per user turn).
- Elasticsearch 8.15.5 `term` suggester, candidate vocabulary indexed from the same official document chunks used for BM25.
- The suggester selects the top suggestion for every eligible term (`suggest_mode=always`, max edits 2, prefix length 1, minimum word length 4), and substitutions are applied before retrieval.
- BM25 corpus, chunking and ranking are held fixed. For the first three arms, the annotated gold domain is fixed, isolating query formulation. The ES-only end-to-end arm searches the whole corpus with no model router.
- No dense retrieval, cross-encoder or answer generation.

## Mean results

| Strategy | Query terms | Recall@5 | MRR@10 | Hit@1 | Hit@3 | Hit@5 | Mean suggest latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw typo dialogue → BM25, no suggestions | 128.72 | 0.71 | 0.5696 | 0.46 | 0.67 | 0.71 | 0 ms |
| ES term suggest → gold-domain BM25 | 128.76 | 0.69 | 0.5471 | 0.43 | 0.66 | 0.69 | 11.086 ms |
| Saved typo-aware LLM router query → gold-domain BM25 | 15.61 | 0.73 | 0.5786 | 0.46 | 0.65 | 0.73 | not timed here |
| ES term suggest → global ES BM25 (no router) | 128.76 | 0.70 | 0.5458 | 0.43 | 0.69 | 0.70 | 11.322 ms |

## Paired ES term-suggest minus raw BM25

| Metric | Mean delta | 95% paired bootstrap CI | Improved / tied / declined |
|---|---:|---:|---:|
| Document Recall@5 | −0.0200 | [−0.0500, 0.0000] | 0 / 98 / 2 |
| MRR@10 | −0.0225 | [−0.0599, +0.0095] | 7 / 84 / 9 |
| Hit@1 | −0.0300 | [−0.0800, +0.0200] | 2 / 93 / 5 |
| Hit@3 | −0.0100 | [−0.0300, 0.0000] | 0 / 99 / 1 |
| Hit@5 | −0.0200 | [−0.0500, 0.0000] | 0 / 98 / 2 |

## Interpretation

On this sample, unconditional ES term replacement before BM25 does not show a retrieval gain over raw BM25 and adds a network call. The point estimates are lower with suggestions; the paired intervals touch/include zero, so the differences are not conclusive. The typo-aware LLM query has higher Recall@5 and MRR point estimates, but the paired intervals versus raw query also include zero, and the comparison changes query length/focus as well as typo correction.

The ES-only pipeline includes global domain search while the other BM25 arms use the gold domain, so treat it as a separate end-to-end diagnostic, not a controlled query-only contrast. This tests automatic top-1 replacement, not confidence-gated suggestions, alternative-query expansion, phrase suggester, or production Elasticsearch analyzer tuning. A follow-up should test applying suggestions only to low-confidence/zero-result queries or retrieving both raw and suggested queries and fusing results.

Exact aggregate values are in `es_suggester_summary.csv` and `es_suggester_paired.csv`. Per-case queries and document IDs are not published.
