# Tool-call typo experiment

Metric: exact set accuracy over selected tool indices.

| Sample | Input | Prompt | n | Exact accuracy | Tool F1 | Mean latency (s) | Mean tokens |
|---|---|---|---:|---:|---:|---:|---:|
| single | clean | direct | 100 | 0.86 | 0.865 | 2.9717 | 267.83 |
| single | clean | typo_aware | 100 | 0.84 | 0.85 | 4.0347 | 268.37 |
| single | typo | correct_then_route | 100 | 0.86 | 0.87 | 6.3139 | 488.92 |
| single | typo | correct_then_typo_aware | 100 | 0.86 | 0.87 | 6.3388 | 494.04 |
| single | typo | direct | 100 | 0.86 | 0.86 | 2.7304 | 310.11 |
| single | typo | typo_aware | 100 | 0.88 | 0.88 | 2.926 | 308.96 |

## Paired accuracy difference

| Sample | Input | Comparison | Pairs | Direct | Other arm | Difference |
|---|---|---|---:|---:|---:|---:|
| single | clean | typo_aware_minus_direct | 100 | 0.86 | 0.84 | -0.02 |
| single | typo | typo_aware_minus_direct | 100 | 0.86 | 0.88 | 0.02 |
| single | typo | correct_then_route_minus_direct | 100 | 0.86 | 0.86 | 0.0 |
| single | typo | correct_then_typo_aware_minus_direct | 100 | 0.86 | 0.86 | 0.0 |
