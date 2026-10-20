# Tool-call typo experiment

Paired comparison of two multi-tool selection prompts using the public MultiDoc2Dial validation set and `gpt-6-luna`:

1. **Direct**: select the tool indices from the user dialogue as written.
2. **Typo-aware**: use the same dialogue, explicitly infer likely intended words despite typos, then select tool indices.

Every example returns only a JSON list of zero-based tool indices. Accuracy is exact set match against the dataset's annotated domain/tool. Conditions are paired on the same cases and typo variants.

The sample is 100 multi-turn target questions (25 per domain) plus 100 first-turn questions from distinct dialogues. The single-turn items are selected from the same validation data and are paired between prompt conditions. One adjacent-character transposition is inserted in the final question for typo conditions. A clean condition checks whether typo instructions change ordinary routing.

The separate-correction arm first asks the model to minimally correct spelling in the user turns, then routes the corrected dialogue in a second API call. It is reported for typo inputs only.

## Run

```bash
python3 benchmark.py --prepare-only
python3 benchmark.py
```

Dependencies: `openai`. The script reads `OPENAI_API_KEY` and uses the OpenAI Responses API. To use an OpenRouter-compatible base URL instead, set `OPENAI_BASE_URL` and set `OPENAI_API_KEY` accordingly. API credentials are never written to result files.

Results are saved to `results/rows.jsonl`, `results/summary.csv`, `results/paired.csv`, and `results/report.md`. Private/local per-row traces, cached prompts, and source dataset files are excluded from the published repository.

Dataset: [MultiDoc2Dial](https://doc2dial.github.io/multidoc2dial/), validation dialogues and documents, downloaded from its official archive URL. The tool functions model four domain-specific search indices (`dmv`, `ssa`, `va`, `studentaid`).

## Interpretation

Metric: exact tool-set accuracy. Since the dataset labels one domain-specific search tool per question, this tests routing/tool-index selection, not compositional multi-tool tasks. Direct and typo-aware conditions make one API call each; separate correction uses two sequential calls. The single-turn control has lower accuracy largely because many standalone initial questions do not provide enough context to infer the annotated domain. Results concern this sample and synthetic adjacent-letter transpositions; they do not establish gains for other typo types or datasets.

## Measured results (gpt-6-luna)

| Sample | Input | Direct | Typo-aware | Correct then route |
|---|---|---:|---:|---:|
| Multi-turn (100) | Clean | 94% | 95% | — |
| Multi-turn (100) | Synthetic typo | 97% | 98% | 97% |
| Single-turn (100) | Clean | 84% | 85% | — |
| Single-turn (100) | Synthetic typo | 85% | 87% | 85% |

The separate correction stage did not improve exact tool accuracy over routing the typo input directly in these samples, while requiring an extra sequential API call. Explicit typo-aware routing was 1–2 percentage points higher. See `results/report.md` and CSV files for paired counts, latency, and token summaries. These small differences should be interpreted as descriptive results, not a broad claim of statistical significance.
