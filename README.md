# Tool-call typo experiment

Paired comparison of two multi-tool selection prompts using the public MultiDoc2Dial validation set and `gpt-6-luna`:

1. **Direct**: select the tool indices from the user dialogue as written.
2. **Typo-aware**: use the same dialogue, explicitly infer likely intended words despite typos, then select tool indices.

Every example returns only a JSON list of zero-based tool indices. Accuracy is exact set match against the dataset's annotated domain/tool. Conditions are paired on the same cases and typo variants.

The sample is 100 multi-turn target questions (25 per domain) plus 100 first-turn questions from distinct dialogues. The single-turn items are selected from the same validation data and are paired between prompt conditions. The baseline typo condition uses one adjacent-character transposition. An additional single-turn experiment injects 2–4 mixed character-level errors (transposition, deletion, insertion, substitution, or repeated character) into the question. A clean condition checks whether typo instructions change ordinary routing.

The separate-correction arm first asks the model to minimally correct spelling in the user turns, then routes the corrected dialogue in a second API call. For the diverse-singleton experiment this is compared with both a direct router and a typo-aware router after correction.

## Run

```bash
python3 benchmark.py --prepare-only
python3 benchmark.py
```

Dependencies: `openai`. The script reads `OPENAI_API_KEY` and uses the OpenAI Responses API. To use an OpenRouter-compatible base URL instead, set `OPENAI_BASE_URL` and set `OPENAI_API_KEY` accordingly. API credentials are never written to result files.

Results are saved to `results/rows.jsonl`, `results/summary.csv`, `results/paired.csv`, and `results/report.md`. Run the diverse single-turn extension with `python3 benchmark.py --out ./results --diverse-singleton --correction-ablation --correction-typo-aware --batch-size 50 --concurrency 50`. Private/local per-row traces, cached prompts, and source dataset files are excluded from the published repository.

Dataset: [MultiDoc2Dial](https://doc2dial.github.io/multidoc2dial/), validation dialogues and documents, downloaded from its official archive URL. The tool functions model four domain-specific search indices (`dmv`, `ssa`, `va`, `studentaid`).

## Interpretation

Metric: exact tool-set accuracy. Since the dataset labels one domain-specific search tool per question, this tests routing/tool-index selection, not compositional multi-tool tasks. Direct and typo-aware conditions make one API call each; separate correction uses two sequential calls. The single-turn control has lower accuracy largely because many standalone initial questions do not provide enough context to infer the annotated domain. Results concern this sample and synthetic adjacent-letter transpositions; they do not establish gains for other typo types or datasets.

## Measured results (gpt-6-luna)

| Sample | Input | Direct | Typo-aware | Correct then route |
|---|---|---:|---:|---:|
| Multi-turn (100) | Clean | 94% | 95% | — |
| Multi-turn (100) | Synthetic typo | 97% | 98% | 97% |
| Single-turn (100) | Clean | 84% | 85% | — |
| Single-turn (100) | One transposition | 85% | 87% | 85% |

The separate correction stage did not improve exact tool accuracy over routing the typo input directly in these samples, while requiring an extra sequential API call. Explicit typo-aware routing was 1–2 percentage points higher. These initial typo results use only one character transposition.

### Stronger diverse-typo single-turn run

| Input / routing strategy | Exact accuracy (n=100) |
|---|---:|
| Clean, direct | 86% |
| Clean, typo-aware prompt | 84% |
| Diverse typos, direct | 86% |
| Diverse typos, typo-aware prompt | 88% |
| Diverse typos, correct then direct route | 86% |
| Diverse typos, correct then typo-aware route | 86% |

The stronger corruption applied 2–4 varied character errors to each eligible question. The typo-aware prompt was +2 percentage points over direct routing on typo inputs but -2 points on clean inputs. Separate correction did not improve accuracy here and used two sequential calls. These are descriptive single-run measurements; paired case-level outcomes are in `results/paired.csv` for the run summary, and no significance claim is intended.

Paired with the same clean direct result for each of the 100 cases, the diverse-typo direct arm had 86/100 correct, with 4 cases changing from wrong→right and 4 changing right→wrong (net 0). The typo-aware arm had 5 wrong→right and 7 right→wrong (net -2 versus its clean typo-aware baseline; its overall accuracy was 88%). Correct-then-direct had 2 wrong→right and 2 right→wrong versus clean direct (net 0); correct-then-typo-aware had 3 wrong→right and 3 right→wrong (net 0). Against the same typo-direct baseline (86/100), typo-aware routing gained two net cases (88/100); either correction pipeline tied it at 86/100. Thus the modest typo-aware gain is visible in this one paired sample, while a separate correction call produced no net gain.

See `results/summary.csv`, `results/paired.csv`, and `results/report.md` for the updated initial summary. The detailed diverse-typo raw rows are retained locally for privacy and are not published.

### Diverse-typo multi-turn run

The same 100 multi-turn cases (25 per domain) were evaluated with clean and 2–4 mixed character errors in the latest user turn. For typo inputs, the run compared direct routing, typo-aware routing, a separate correction call followed by direct routing, and correction followed by typo-aware routing.

| Input / routing strategy | Exact accuracy (n=100) |
|---|---:|
| Clean, direct | 96% |
| Clean, typo-aware prompt | 95% |
| Diverse typos, direct | 97% |
| Diverse typos, typo-aware prompt | 97% |
| Diverse typos, correct then direct route | 97% |
| Diverse typos, correct then typo-aware route | 97% |

Against the same typo-direct baseline, typo-aware routing and both correction pipelines had zero net accuracy change (97/100 each). Pairwise outputs were identical on 96 cases; among the four changed outcomes, each comparison had one wrong→right and one right→wrong. Clean direct was 96/100, while clean typo-aware was 95/100. The separate correction pipelines added a sequential model call without improving exact tool accuracy in this sample. These are descriptive outcomes from one run, not a significance claim. The detailed raw rows stay local; this public summary reports aggregate counts only.

The multi-turn run summary is in `results/multiturn_diverse_summary.csv`, with paired accuracy comparisons in `results/multiturn_diverse_paired.csv` and a readable report in `results/multiturn_diverse_report.md`.

## BM25: raw typo dialogue vs typo-aware router query

We also tested whether BM25 needs term suggestion when retrieval can use the query produced by a typo-aware router. On the same 100 heavily typoed multi-turn cases and fixed gold-domain BM25 index, raw corrupted full-dialogue search achieved **0.71 document Recall@5 / 0.5696 MRR@10** without suggestions. Searching the router's focused standalone query achieved **0.73 / 0.5786**. Paired bootstrap 95% confidence intervals include zero for both differences, so the small numerical gain does not establish that the router query is better. BM25 without term suggestion performed similarly on this sample, despite using much longer queries (mean 128.72 terms vs. 15.61). This is a practical pipeline comparison, not a causal isolation of typo correction: query length and focus also differ. See `results/bm25_raw_vs_router_report.md` and its summary/pairwise CSVs. Raw per-case rows are excluded.
