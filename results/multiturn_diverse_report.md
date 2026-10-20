# Diverse-typo multi-turn tool-routing experiment

- Model: `gpt-6-luna`
- Dataset: 100 MultiDoc2Dial multi-turn cases, 25 per domain
- Typo treatment: 2–4 mixed character errors (transposition, deletion, insertion, substitution, or repetition) in the final user utterance
- Metric: exact tool-index set accuracy
- Requests: one call per ordinary routing evaluation; two sequential calls for correction-then-route arms

| Input | Routing strategy | n | Exact accuracy | Mean latency (s) | Mean tokens |
|---|---|---:|---:|---:|---:|
| Clean | Direct | 100 | 96% | 1.8121 | 337.03 |
| Clean | Typo-aware prompt | 100 | 95% | 1.8792 | 336.52 |
| Diverse typo | Direct | 100 | 97% | 2.0021 | 375.57 |
| Diverse typo | Typo-aware prompt | 100 | 97% | 2.0171 | 372.51 |
| Diverse typo | Correct, then direct route | 100 | 97% | 5.9026 | 798.77 |
| Diverse typo | Correct, then typo-aware route | 100 | 97% | 5.9513 | 814.26 |

Each typo-arm comparison to direct has 96/100 cases with matching correctness. Of the remaining four, one changes wrong→right and one right→wrong, for no net accuracy difference. Separate correction has no net exact-accuracy gain over direct routing and adds a sequential model call. These are descriptive measurements from a single run; no significance claim is intended.

Raw per-case prompts and outputs are retained locally and not included in this public result summary.
