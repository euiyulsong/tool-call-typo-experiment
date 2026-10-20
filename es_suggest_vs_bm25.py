#!/usr/bin/env python3
"""Compare Elasticsearch term suggester pipelines to raw BM25 on paired typo queries."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = Path("/mnt/c/Users/thddm/Downloads/multiturn_typo_benchmark")
sys.path.insert(0, str(BENCH))
from benchmark import build_cases, build_chunks  # noqa: E402
from bm25_query_experiment import bootstrap_ci, tokens, DEFAULT_ARCHIVE, DEFAULT_DETAILS


def load_source(archive: Path, details_path: Path):
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]
        def read(suffix):
            n = next(n for n in names if n.endswith(suffix))
            return json.loads(zf.read(n).decode("utf-8-sig"))
        docs = read("multidoc2dial_doc.json")["doc_data"]
        dial = read("multidoc2dial_dial_validation.json")["dial_data"]
    cases = build_cases(docs, dial, 20261020, 25)
    by_id = {x["id"]: x for x in cases}
    details = [json.loads(s) for s in details_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    typo = {x["case_id"]: x for x in details if x["condition"] == "typo"}
    if set(by_id) != set(typo):
        raise RuntimeError("Case IDs differ between dataset sample and heavy-typo detail rows")
    return docs, by_id, typo


def es_suggest(es, index: str, query: str) -> str:
    resp = es.search(index=index, size=0, suggest={
        "terms": {"text": query, "term": {"field": "text", "suggest_mode": "always",
                  "sort": "score", "max_edits": 2, "prefix_length": 1,
                  "min_word_length": 4, "max_inspections": 8, "min_doc_freq": 1}}
    })
    replacements = {}
    for entry in resp.get("suggest", {}).get("terms", []):
        options = entry.get("options", [])
        if options:
            replacements[entry["text"]] = options[0]["text"]
    # Substitute token spans, preserving punctuation/whitespace and untouched terms.
    return re.sub(r"[A-Za-z0-9]+", lambda m: replacements.get(m.group(), m.group()), query)


def load_bm25(cases, docs, es, index):
    from rank_bm25 import BM25Okapi
    chunks = build_chunks(docs)
    domains = {}
    indices = {}
    for domain in ("dmv", "ssa", "va", "studentaid"):
        group = [c for c in chunks if c.domain == domain]
        domains[domain] = group
        indices[domain] = BM25Okapi([tokens(c.text) for c in group])
    return domains, indices


def rank_query(domain, query, domains, indices):
    chunks = domains[domain]
    scores = indices[domain].get_scores(tokens(query))
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return [chunks[i].doc_id for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    ap.add_argument("--out", type=Path, default=HERE / "results/es_suggester")
    ap.add_argument("--es-url", default="http://localhost:19200")
    ap.add_argument("--index", default="multidoc2dial-suggest")
    ap.add_argument("--suggestor", choices=("term", "phrase"), default="term")
    args = ap.parse_args()
    from elasticsearch import Elasticsearch, helpers

    es = Elasticsearch(args.es_url, request_timeout=120)
    if not es.ping():
        raise RuntimeError(f"Elasticsearch unavailable at {args.es_url}")
    docs, cases, typo_rows = load_source(args.archive, args.details)
    chunks = build_chunks(docs)
    if es.indices.exists(index=args.index):
        es.indices.delete(index=args.index)
    es.indices.create(index=args.index, settings={"analysis": {"analyzer": {
        "terms_lower": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase"]}}}},
        mappings={"properties": {"text": {"type": "text", "analyzer": "terms_lower"},
                                  "domain": {"type": "keyword"}, "doc_id": {"type": "keyword"}}})
    def actions():
        for i, chunk in enumerate(chunks):
            yield {"_index": args.index, "_id": str(i),
                   "_source": {"text": chunk.text, "domain": chunk.domain, "doc_id": chunk.doc_id}}
    helpers.bulk(es, actions(), chunk_size=1000, request_timeout=120)
    es.indices.refresh(index=args.index)

    domains, bm25 = load_bm25(cases, docs, es, args.index)
    records = []
    condition_map = {
        # Oracle-domain BM25 arms isolate query text and suggestion effects.
        "raw_full_dialogue_bm25": lambda raw, rewrite, last: raw,
        "last_turn_raw_bm25": lambda raw, rewrite, last: last,
        "llm_rewrite_bm25": lambda raw, rewrite, last: rewrite,
        "es_suggest_raw_full_dialogue_bm25": lambda raw, rewrite, last: es_suggest(es, args.index, raw),
        "es_suggest_last_turn_bm25": lambda raw, rewrite, last: es_suggest(es, args.index, last),
        "es_suggest_llm_rewrite_bm25": lambda raw, rewrite, last: es_suggest(es, args.index, rewrite),
    }
    for case_id, case in cases.items():
        detail = typo_rows[case_id]
        raw_query = "\n".join(f'{m["role"].upper()}: {m["text"]}'
                               for m in detail["variant_history"] + [{"role": "user", "text": detail["variant_question"]}])
        last_turn_query = detail["variant_question"]
        rewrite_query = detail["selection_and_search_query"]
        domain = case["domain"]  # hold routing fixed at gold domain to isolate retrieval/query strategy
        gold = set(case["gold_docs"])
        for condition, builder in condition_map.items():
            t0 = time.perf_counter()
            q = builder(raw_query, rewrite_query, last_turn_query)
            suggest_s = time.perf_counter() - t0 if condition.startswith("es_suggest_") else 0.0
            t1 = time.perf_counter()
            ranked = rank_query(domain, q, domains, bm25)
            first = next((i for i, docid in enumerate(ranked[:10], 1) if docid in gold), None)
            recall = len(set(ranked[:5]) & gold) / len(gold)
            search_s = time.perf_counter() - t1
            records.append({"case_id": case_id, "domain": domain, "condition": condition,
                            "input_query_kind": ("raw_full_dialogue" if "full_dialogue" in condition else
                                                 "last_turn_only" if "last_turn" in condition else "llm_router_rewrite"),
                            "query_text": q,
                            "suggested_query": q if "es_suggest_" in condition else "",
                            "query_terms": len(tokens(q)), "query_chars": len(q),
                            "suggest_latency_s": suggest_s, "search_latency_s": search_s,
                            "total_latency_s": suggest_s + search_s,
                            "bm25_doc_recall@5": len(set(ranked[:5]) & gold) / len(gold),
                            "bm25_mrr@10": 1 / first if first else 0.0,
                            "hit@1": float(first == 1), "hit@3": float(bool(first and first <= 3)),
                            "hit@5": float(bool(first and first <= 5)), "first_gold_rank": first or ""})
            print(f'{len(records):3d}/400 {condition} {case_id} recall@5={records[-1]["bm25_doc_recall@5"]:.2f}', flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
    by_case = {(r["case_id"], r["condition"]): r for r in records}
    conditions = tuple(condition_map)
    metrics = ("bm25_doc_recall@5", "bm25_mrr@10", "hit@1", "hit@3", "hit@5")
    summaries = []
    for condition in conditions:
        batch = [r for r in records if r["condition"] == condition]
        summaries.append({"condition": condition, "n": len(batch),
                          "input_query_kind": batch[0]["input_query_kind"],
                          "mean_query_terms": round(statistics.fmean(r["query_terms"] for r in batch), 2),
                          "mean_suggest_latency_ms": round(statistics.fmean(r["suggest_latency_s"] for r in batch) * 1000, 3),
                          "mean_search_latency_ms": round(statistics.fmean(r["search_latency_s"] for r in batch) * 1000, 3),
                          "mean_total_latency_ms": round(statistics.fmean(r["total_latency_s"] for r in batch) * 1000, 3),
                          **{m: round(statistics.fmean(r[m] for r in batch), 4) for m in metrics}})
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    paired = []
    comparison_pairs = [("es_suggest_raw_full_dialogue_bm25", "raw_full_dialogue_bm25"),
                        ("es_suggest_last_turn_bm25", "last_turn_raw_bm25"),
                        ("es_suggest_llm_rewrite_bm25", "llm_rewrite_bm25"),
                        ("last_turn_raw_bm25", "raw_full_dialogue_bm25"),
                        ("llm_rewrite_bm25", "raw_full_dialogue_bm25")]
    for condition, baseline in comparison_pairs:
        comparison_name = f"{condition}_minus_{baseline}"
        if condition == "es_suggest_raw_full_dialogue_bm25":
            comparison_name = "es_suggest_raw_query_minus_raw_query"
        elif condition == "es_suggest_last_turn_bm25":
            comparison_name = "es_suggest_last_turn_query_minus_last_turn_query"
        elif condition == "es_suggest_llm_rewrite_bm25":
            comparison_name = "es_suggest_llm_rewrite_query_minus_llm_rewrite_query"
        for metric in metrics:
            delta = [by_case[(case_id, condition)][metric] - by_case[(case_id, baseline)][metric]
                     for case_id in cases]
            lo, hi = bootstrap_ci(delta, seed=20261021)
            paired.append({"comparison": comparison_name, "metric": metric,
                           "n_pairs": len(delta), "mean_delta": round(statistics.fmean(delta), 4),
                           "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                           "improved": sum(x > 0 for x in delta), "tied": sum(x == 0 for x in delta),
                           "declined": sum(x < 0 for x in delta)})
    with (args.out / "paired.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
    report = ["# Elasticsearch term suggester vs BM25: query-by-query comparison", "",
              "All arms use the same 100 heavily typoed MultiDoc2Dial multi-turn cases, same gold-domain oracle, same chunk corpus and BM25 ranking. No generated answers are involved. Elasticsearch 8.15.5 term suggester draws candidates from the same full document corpus.", "",
              "Query definitions: `raw_full_dialogue` is all corrupted user and agent turns concatenated; `last_turn_only` is just the corrupted final user utterance; `llm_router_rewrite` is the saved typo-aware router's standalone search query. For each query source, separate arms run without suggestion and with ES top-1 term suggestions applied before BM25. Gold domain is fixed to isolate query/suggest behavior; this is not a test of ES routing.", "",
              "## Means", "", "| Strategy | Input query | n | Terms | Recall@5 | MRR@10 | Hit@1 | Hit@3 | Hit@5 | Suggest ms | Search ms |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        row = next(x for x in records if x["condition"] == r["condition"])
        report.append("| " + " | ".join(str(x) for x in (r["condition"], row["input_query_kind"], r["n"], r["mean_query_terms"], r["bm25_doc_recall@5"], r["bm25_mrr@10"], r["hit@1"], r["hit@3"], r["hit@5"], r["mean_suggest_latency_ms"], r["mean_search_latency_ms"])) + " |")
    report += ["", "## Paired comparisons", "", "| Comparison | Metric | Δ | 95% paired bootstrap CI | W/T/L |", "|---|---|---:|---:|---:|"]
    for r in paired:
        report.append(f'| {r["comparison"]} | {r["metric"]} | {r["mean_delta"]:+.4f} | [{r["ci95_low"]:+.4f}, {r["ci95_high"]:+.4f}] | {r["improved"]}/{r["tied"]}/{r["declined"]} |')
    (args.out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    es.indices.delete(index=args.index)


if __name__ == "__main__":
    main()
