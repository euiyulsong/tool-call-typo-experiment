# Tool-call typo experiment

Metric: exact set accuracy over selected tool indices.

| Sample | Input | Prompt | n | Exact accuracy | Tool F1 | Mean latency (s) | Mean tokens |
|---|---|---|---:|---:|---:|---:|---:|
| multi | clean | direct | 100 | 0.94 | 0.9467 | 2.6769 | 336.1 |
| multi | clean | typo_aware | 100 | 0.95 | 0.9567 | 3.5285 | 335.75 |
| multi | typo | correct_then_route | 100 | 0.97 | 0.97 | 7.1144 | 723.18 |
| multi | typo | direct | 100 | 0.97 | 0.97 | 1.8502 | 372.87 |
| multi | typo | typo_aware | 100 | 0.98 | 0.98 | 1.8478 | 368.84 |
| single | clean | direct | 100 | 0.84 | 0.849 | 2.9316 | 272.2 |
| single | clean | typo_aware | 100 | 0.85 | 0.8617 | 2.8874 | 268.01 |
| single | typo | correct_then_route | 100 | 0.85 | 0.8667 | 6.286 | 468.83 |
| single | typo | direct | 100 | 0.85 | 0.8567 | 2.1785 | 304.85 |
| single | typo | typo_aware | 100 | 0.87 | 0.87 | 2.3716 | 302.85 |

## Paired accuracy difference

| Sample | Input | Comparison | Pairs | Direct | Other arm | Difference |
|---|---|---|---:|---:|---:|---:|
| multi | clean | typo_aware_minus_direct | 100 | 0.94 | 0.95 | 0.01 |
| multi | typo | typo_aware_minus_direct | 100 | 0.97 | 0.98 | 0.01 |
| multi | typo | correct_then_route_minus_direct | 100 | 0.97 | 0.97 | 0.0 |
| single | clean | typo_aware_minus_direct | 100 | 0.84 | 0.85 | 0.01 |
| single | typo | typo_aware_minus_direct | 100 | 0.85 | 0.87 | 0.02 |
| single | typo | correct_then_route_minus_direct | 100 | 0.85 | 0.85 | 0.0 |
