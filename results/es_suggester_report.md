# Elasticsearch term suggester: each query input evaluated separately

All results use the same 100 heavy-typo MultiDoc2Dial multi-turn cases and gold-domain BM25 index. Elasticsearch 8.15.5 term suggester uses the same corpus vocabulary and replaces each eligible token with its top suggestion.

## Input query definitions

- **Raw full dialogue:** all corrupted user and agent turns concatenated.
- **Last turn only:** only the corrupted final user utterance.
- **LLM router rewrite:** the saved typo-aware router's focused standalone search query.

Each input is tested without suggestion and with Elasticsearch term suggestions, then searched by the same gold-domain BM25 index. The domain is fixed to gold for all arms; this isolates retrieval and does not measure ES routing.

## Mean results

| Strategy | Input query | n | Mean terms | Recall@5 | MRR@10 | Hit@1 | Hit@3 | Hit@5 | Suggest ms | Search ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw BM25 | Raw full dialogue | 100 | 128.72 | 0.71 | 0.5696 | 0.46 | 0.67 | 0.71 | 0.0 | 9.201 |
| Raw BM25 | Last turn only | 100 | 13.05 | 0.44 | 0.3322 | 0.27 | 0.36 | 0.44 | 0.0 | 0.909 |
| Raw BM25 | LLM router rewrite | 100 | 15.61 | 0.73 | 0.5786 | 0.46 | 0.65 | 0.73 | 0.0 | 1.107 |
| ES term suggest → BM25 | Raw full dialogue | 100 | 128.76 | 0.69 | 0.5471 | 0.43 | 0.66 | 0.69 | 11.6 | 9.473 |
| ES term suggest → BM25 | Last turn only | 100 | 13.06 | 0.41 | 0.3109 | 0.25 | 0.34 | 0.41 | 3.368 | 1.443 |
| ES term suggest → BM25 | LLM router rewrite | 100 | 15.61 | 0.71 | 0.5346 | 0.41 | 0.60 | 0.71 | 2.618 | 1.654 |

## ES suggestion effect within each query input

| Input query | Metric | ES suggest − raw | Paired 95% bootstrap CI | W/T/L |
|---|---|---:|---:|---:|
| Raw full dialogue | Recall@5 | −0.0200 | [−0.0500, 0.0000] | 0/98/2 |
| Raw full dialogue | MRR@10 | −0.0225 | [−0.0599, +0.0095] | 7/84/9 |
| Last turn only | Recall@5 | −0.0300 | [−0.0800, +0.0200] | 2/93/5 |
| Last turn only | MRR@10 | −0.0213 | [−0.0613, +0.0158] | 5/86/9 |
| LLM router rewrite | Recall@5 | −0.0200 | [−0.0500, 0.0000] | 0/98/2 |
| LLM router rewrite | MRR@10 | **−0.0440** | **[−0.0789, −0.0149]** | 2/88/10 |

## Query-source comparison without suggestion

| Comparison | Metric | Δ | Paired 95% bootstrap CI |
|---|---|---:|---:|
| Last turn only − raw full dialogue | Recall@5 | −0.2700 | [−0.4000, −0.1400] |
| LLM router rewrite − raw full dialogue | Recall@5 | +0.0200 | [−0.0900, +0.1300] |
| LLM router rewrite − raw full dialogue | MRR@10 | +0.0090 | [−0.0764, +0.0957] |

## Interpretation

The three query types must be interpreted separately. The last-turn-only query performs much worse than the full history, demonstrating that earlier dialogue context carries retrieval information. The LLM rewrite is a compact context-aware standalone query and had the highest Recall@5/MRR point scores without ES suggestion. Across all three query inputs, unconditional top-1 ES suggestion did not improve recall; when applied to the rewritten query, MRR@10 decreased with its paired confidence interval below zero.

These are severe synthetic typos and a single 100-case run. ES term suggestion auto-replaced top-1 suggestions; confidence-gated fallback, candidate expansion/fusion, phrase suggester and production analyzer tuning were not tested. Per-case query rows are kept local.
