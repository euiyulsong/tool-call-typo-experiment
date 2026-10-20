#!/usr/bin/env python3
"""Compare raw typo queries to typo-aware router queries in a fixed BM25 index."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BENCH = Path("/mnt/c/Users/thddm/Downloads/multiturn_typo_benchmark")
DEFAULT_ARCHIVE = HERE / "results/data/multidoc2dial.zip"
DEFAULT_DETAILS = DEFAULT_BENCH / "results-heavy-typos-router-rewrite-100/details_fresh.jsonl"
sys.path.insert(0, str(DEFAULT_BENCH))
from benchmark import build_cases, build_chunks, stable_seed  # noqa: E402


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = int(pos)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (pos - lo)


def bootstrap_ci(deltas: list[float], seed: int = 20261021, iters: int = 10000) -> tuple[float, float]:
    import random
    rng = random.Random(seed)
    boot = [statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(iters)]
    return percentile(boot, .025), percentile(boot, .975)


def load_data(archive: Path) -> tuple[dict, dict]:
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]
        def read(suffix: str) -> dict:
            name = next(n for n in names if n.endswith(suffix))
            return json.loads(zf.read(name).decode("utf-8-sig"))
        return read("multidoc2dial_doc.json")["doc_data"], read("multidoc2dial_dial_validation.json")["dial_data"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    ap.add_argument("--out", type=Path, default=HERE / "results/bm25_raw_vs_router")
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()

    docs, dials = load_data(args.archive)
    cases = build_cases(docs, dials, args.seed, 25)
    case_by_id = {c["id"]: c for c in cases}
    details = [json.loads(s) for s in args.details.read_text(encoding="utf-8").splitlines() if s.strip()]
    typos = {r["case_id"]: r for r in details if r["condition"] == "typo"}
    if set(case_by_id) != set(typos):
        raise RuntimeError(f"Case mismatch: selected={len(case_by_id)}, typo details={len(typos)}")

    chunks = build_chunks(docs)
    from rank_bm25 import BM25Okapi
    by_domain = {}
    bm25 = {}
    for domain in ("dmv", "ssa", "va", "studentaid"):
        group = [c for c in chunks if c.domain == domain]
        by_domain[domain] = group
        bm25[domain] = BM25Okapi([tokens(c.text) for c in group])

    rows = []
    for case_id, case in case_by_id.items():
        detail = typos[case_id]
        domain = case["domain"]
        gold_docs = set(case["gold_docs"])
        raw_dialogue = "\n".join(f'{m["role"].upper()}: {m["text"]}' for m in detail["variant_history"] + [{"role":"user","text":detail["variant_question"]}])
        router_query = detail["selection_and_search_query"]
        for condition, query in (("typo_raw_no_suggest", raw_dialogue), ("typo_router_query", router_query)):
            scores = bm25[domain].get_scores(tokens(query))
            order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
            ranked_docs = [by_domain[domain][i].doc_id for i in order]
            first_rank = next((rank for rank, docid in enumerate(ranked_docs[:10], 1) if docid in gold_docs), None)
            recall = len(set(ranked_docs[:5]) & gold_docs) / len(gold_docs)
            rows.append({"case_id": case_id, "domain": domain, "condition": condition,
                         "query_characters": len(query), "query_terms": len(tokens(query)),
                         "gold_doc_count": len(gold_docs), "bm25_doc_recall@5": recall,
                         "bm25_mrr@10": 1 / first_rank if first_rank else 0.0,
                         "bm25_hit@1": float(bool(first_rank and first_rank <= 1)),
                         "bm25_hit@3": float(bool(first_rank and first_rank <= 3)),
                         "bm25_hit@5": float(bool(first_rank and first_rank <= 5)),
                         "first_gold_rank": first_rank or ""})

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    paired = []
    metrics = ("bm25_doc_recall@5", "bm25_mrr@10", "bm25_hit@1", "bm25_hit@3", "bm25_hit@5")
    indexed = {(r["case_id"], r["condition"]): r for r in rows}
    for metric in metrics:
        deltas = [float(indexed[(c["id"], "typo_router_query")][metric]) -
                  float(indexed[(c["id"], "typo_raw_no_suggest")][metric]) for c in cases]
        lo, hi = bootstrap_ci(deltas)
        raw_mean = statistics.fmean(float(indexed[(c["id"], "typo_raw_no_suggest")][metric]) for c in cases)
        router_mean = statistics.fmean(float(indexed[(c["id"], "typo_router_query")][metric]) for c in cases)
        paired.append({"metric": metric, "n_pairs": len(deltas), "raw_no_suggest": round(raw_mean, 4),
                       "typo_aware_router_query": round(router_mean, 4),
                       "mean_paired_delta": round(statistics.fmean(deltas), 4),
                       "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                       "improved": sum(d > 0 for d in deltas), "tied": sum(d == 0 for d in deltas),
                       "declined": sum(d < 0 for d in deltas)})
    with (args.out / "paired.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0]))
        w.writeheader(); w.writerows(paired)

    summary = []
    for cond in ("typo_raw_no_suggest", "typo_router_query"):
        batch = [r for r in rows if r["condition"] == cond]
        summary.append({"condition": cond, "n": len(batch),
                        "mean_query_terms": round(statistics.fmean(r["query_terms"] for r in batch), 2),
                        "mean_bm25_doc_recall@5": round(statistics.fmean(r["bm25_doc_recall@5"] for r in batch), 4),
                        "mean_bm25_mrr@10": round(statistics.fmean(r["bm25_mrr@10"] for r in batch), 4),
                        "hit@1": round(statistics.fmean(r["bm25_hit@1"] for r in batch), 4),
                        "hit@3": round(statistics.fmean(r["bm25_hit@3"] for r in batch), 4),
                        "hit@5": round(statistics.fmean(r["bm25_hit@5"] for r in batch), 4)})
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)

    report = ["# BM25: raw typo query vs typo-aware router query", "",
              "Same 100 MultiDoc2Dial multi-turn cases, same heavy typo variants, same BM25 index and gold domain. No term suggestion, spelling dictionary, fuzzy expansion, dense retrieval, or cross-encoder reranking is used.", "",
              "- **Raw no-suggest:** BM25 receives the complete corrupted dialogue as the search string.",
              "- **Router query:** BM25 receives the standalone search query generated by the existing typo-aware router run from the same corrupted dialogue.",
              "- This is a retrieval-query-source comparison, not an isolated causal effect of the typo prompt: the inputs differ in focus and length as well as typo normalization.", "",
              "## Summary", "", "| Query source | Cases | Mean terms | Doc Recall@5 | MRR@10 | Hit@1 | Hit@3 | Hit@5 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        report.append("| " + " | ".join(str(r[k]) for k in ("condition", "n", "mean_query_terms", "mean_bm25_doc_recall@5", "mean_bm25_mrr@10", "hit@1", "hit@3", "hit@5")) + " |")
    report += ["", "## Paired router-query minus raw-query deltas", "", "| Metric | Mean delta | Paired bootstrap 95% CI | Improved / tied / declined |", "|---|---:|---:|---:|"]
    for r in paired:
        report.append(f'| {r["metric"]} | {r["mean_paired_delta"]:+.4f} | [{r["ci95_low"]:+.4f}, {r["ci95_high"]:+.4f}] | {r["improved"]} / {r["tied"]} / {r["declined"]} |')
    (args.out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
