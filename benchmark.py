#!/usr/bin/env python3
"""Paired typo-aware vs direct tool-index selection on MultiDoc2Dial."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

DATASET_URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"
DOMAINS = ("dmv", "ssa", "va", "studentaid")
TOOLS = ["search_dmv", "search_ssa", "search_va", "search_studentaid"]
EXPLAIN = {
    "search_dmv": "Search the DMV index for driver's licenses, vehicle registration, and state motor vehicles.",
    "search_ssa": "Search the Social Security index for retirement, benefits, and disability.",
    "search_va": "Search the Veterans Affairs index for veterans and VA benefits.",
    "search_studentaid": "Search the federal student aid index for FAFSA, grants, and student loans.",
}
STOPWORDS = set("about above after again all also any are because before being between can could does each from have into more most other over some such than that them then there these they this those through under until very were what when where which while will would your please thanks thank hello hi need want tell help just like know".split())


def stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:15], 16)


def load_data(data_dir: Path) -> tuple[dict, dict]:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "multidoc2dial.zip"
    if not archive.exists():
        print(f"Downloading official MultiDoc2Dial archive: {DATASET_URL}", flush=True)
        urllib.request.urlretrieve(DATASET_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]
        def load(suffix: str) -> dict:
            matches = [n for n in names if n.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one {suffix}, got {matches}")
            return json.loads(zf.read(matches[0]).decode("utf-8-sig"))
        docs = load("multidoc2dial_doc.json")["doc_data"]
        dial = load("multidoc2dial_dial_validation.json")["dial_data"]
    return docs, dial


def gold_tool(agent_turn: dict, domain: str) -> str | None:
    refs = [r for r in agent_turn.get("references", []) if r.get("label") in ("solution", "precondition")]
    if refs and any(r.get("doc_id") for r in refs):
        return f"search_{domain}"
    return None


def select_cases(doc_data: dict, dial_data: dict, seed: int = 20261020) -> list[dict]:
    multi: list[dict] = []
    single: list[dict] = []
    for domain in DOMAINS:
        dialogs = dial_data.get(domain, [])
        eligible_multi = []
        eligible_single = []
        for dialog in dialogs:
            turns = dialog.get("turns", [])
            for i, turn in enumerate(turns[:-1]):
                if turn.get("role") != "user" or turns[i + 1].get("role") != "agent":
                    continue
                tool = gold_tool(turns[i + 1], domain)
                if not tool:
                    continue
                row = {"id": f'{dialog["dial_id"]}:{turn["turn_id"]}',
                       "domain": domain, "question": turn["utterance"], "gold_tool": tool}
                prior_user = [t["utterance"] for t in turns[:i] if t.get("role") == "user"]
                if prior_user:
                    eligible_multi.append({**row, "history": prior_user})
                else:
                    eligible_single.append({**row, "history": []})
            # Keep all eligible first-user-turn dialogues; deduplicate below across domains.
            single.extend(eligible_single)
        eligible_multi.sort(key=lambda x: stable_seed(f'{seed}|{x["id"]}'))
        domain_multi = eligible_multi[:25]
        if len(domain_multi) < 25:
            raise RuntimeError(f"Only {len(domain_multi)} qualified multi-turn {domain} cases; need 25")
        multi.extend(domain_multi)
    single.sort(key=lambda x: stable_seed(f'{seed}|single|{x["id"]}'))
    single = list({c["id"]: c for c in single}.values())
    if len(single) < 100:
        raise RuntimeError(f"Only {len(single)} distinct single-turn cases; need 100")
    cases = [{"sample": "multi", **c} for c in multi]
    cases += [{"sample": "single", **c} for c in single[:100]]
    return sorted(cases, key=lambda x: (x["sample"], x["id"]))


def corrupt_one(text: str, seed: int) -> tuple[str, dict | None]:
    words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z]{5,}", text)
             if m.group().lower() not in STOPWORDS]
    if not words:
        words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z]{4,}", text)]
    rng = random.Random(seed)
    if not words:
        return text, None
    start, end, word = rng.choice(words)
    candidates = [i for i in range(len(word) - 1) if word[i].lower() != word[i + 1].lower()]
    if not candidates:
        return text, None
    j = rng.choice(candidates)
    typo = word[:j] + word[j + 1] + word[j] + word[j + 2:]
    return text[:start] + typo + text[end:], {"before": word, "after": typo}


def corrupt_diverse(text: str, seed: int) -> tuple[str, list[dict]]:
    """Apply several mixed, deterministic character-level errors to content words."""
    rng = random.Random(seed)
    words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z]{4,}", text)
             if m.group().lower() not in STOPWORDS]
    if not words:
        words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z]{4,}", text)]
    if not words:
        return text, []
    desired = min(len(words), rng.randint(2, 4))
    chosen = rng.sample(words, desired)
    edits = []
    # Apply right-to-left so earlier character offsets remain stable.
    for start, end, word in sorted(chosen, reverse=True):
        kind = rng.choice(("transpose", "delete", "insert", "substitute", "repeat"))
        pos = rng.randrange(len(word))
        if kind == "transpose":
            candidates = [i for i in range(len(word)-1) if word[i].lower() != word[i+1].lower()]
            if not candidates: kind = "delete"
            else:
                pos = rng.choice(candidates)
                changed = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
        if kind == "delete":
            changed = word[:pos] + word[pos+1:]
        elif kind == "insert":
            changed = word[:pos] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[pos:]
        elif kind == "substitute":
            letter = rng.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c != word[pos].lower()])
            changed = word[:pos] + (letter.upper() if word[pos].isupper() else letter) + word[pos+1:]
        elif kind == "repeat":
            changed = word[:pos+1] + word[pos] + word[pos+1:]
        new_text = text[:start] + changed + text[end:]
        edits.append({"kind": kind, "before": word, "after": changed})
        text = new_text
    edits.reverse()
    return text, edits


def make_prompt(case: dict, typo: bool, diverse: bool = False) -> tuple[str, dict | None]:
    if typo and diverse:
        question, edit = corrupt_diverse(case["question"], stable_seed("diverse|" + case["id"]))
    else:
        question, one_edit = corrupt_one(case["question"], stable_seed("typo|" + case["id"])) if typo else (case["question"], None)
        edit = [one_edit] if one_edit else []
    history = case["history"]
    system = ("You route user questions to domain-specific search functions. Select the relevant function(s) "
              "from the available tools. Return ONLY a JSON object with `indices`, an array of zero-based integer "
              "indices into the tool list. Do not answer the user. Use the smallest sufficient set, usually one tool. "
              "Tool list in index order: " + json.dumps(TOOLS) + ". Descriptions: " + json.dumps(EXPLAIN) + ".")
    if typo:
        system += (" The dialogue may contain typos. Infer intended words from context and account for likely "
                   "spelling errors when selecting the tool indices. Do not output corrected text, only indices.")
    messages = "\n".join(f"USER (earlier): {x}" for x in history)
    if messages:
        messages += "\n"
    messages += "USER (current): " + question
    return system, {"question": messages, "typo_edit": edit}


def correct_prompt(case: dict, typo: bool, diverse: bool = False) -> tuple[str, dict | None]:
    """Prepare an input first corrected in a separate LLM call, then route it."""
    packed_history = list(case["history"])
    question = case["question"]
    edits = []
    if typo:
        for i, message in enumerate(packed_history):
            if diverse:
                message, history_edits = corrupt_diverse(message, stable_seed(f'diverse|{case["id"]}|history|{i}'))
                edit = history_edits
            else:
                message, one_edit = corrupt_one(message, stable_seed(f'typo|{case["id"]}|history|{i}'))
                edit = [one_edit] if one_edit else []
            packed_history[i] = message
            if edit: edits.extend({"turn": i, **e} for e in edit)
        if diverse:
            question, question_edits = corrupt_diverse(question, stable_seed("diverse|" + case["id"]))
            edits.extend({"turn": "target", **e} for e in question_edits)
        else:
            question, one_edit = corrupt_one(question, stable_seed("typo|" + case["id"]))
            if one_edit: edits.append({"turn": "target", **one_edit})
    context = "\n".join(f"USER (earlier): {m}" for m in packed_history)
    if context: context += "\n"
    context += "USER (current): " + question
    correct_system = ("Correct spelling typos in the user dialogue only. Preserve meaning, entities, references, "
                      "and wording; do not answer or route the question. Return JSON with `history` (array of "
                      "strings, same length/order) and `question` (string). If no typo is evident, copy the text.")
    route_system = ("You route user questions to domain-specific search functions. Select the relevant function(s) "
                    "from the available tools. Return ONLY a JSON object with `indices`, an array of zero-based integer "
                    "indices into the tool list. Do not answer the user. Use the smallest sufficient set, usually one tool. "
                    "Tool list in index order: " + json.dumps(TOOLS) + ". Descriptions: " + json.dumps(EXPLAIN) + ".")
    route_input = "\n".join(f"USER (earlier): {m}" for m in packed_history)
    if route_input: route_input += "\n"
    route_input += "USER (current): " + question
    return correct_system, {"input": context, "route_system": route_system, "route_input": route_input,
                           "typo_edit": edits, "case": case, "condition": "typo" if typo else "clean",
                           "prompt_arm": "correct_then_route"}


class Runner:
    def __init__(self, model: str, cache_path: Path):
        from openai import OpenAI
        self.client = OpenAI(max_retries=2, timeout=90)
        self.model = model
        self.cache_path = cache_path
        self.cache = {}
        self.correction_cache = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "history" in row and "question" in row:
                        self.correction_cache[row["key"]] = row
                    else:
                        self.cache[row["key"]] = row

    def choose(self, system: str, prompt: str, key: str) -> dict:
        if key in self.cache:
            return self.cache[key]
        st = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=prompt,
            text={"format": {"type": "json_schema", "name": "tool_indices", "strict": True,
                             "schema": {"type": "object", "properties": {"indices": {
                                 "type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 3}}},
                                 "required": ["indices"], "additionalProperties": False}}},
        )
        elapsed = time.perf_counter() - st
        result = json.loads(response.output_text)
        row = {"key": key, "indices": result["indices"],
               "latency_s": elapsed,
               "tokens": int(getattr(getattr(response, "usage", None), "total_tokens", 0) or 0)}
        self.cache[key] = row
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def correct(self, system: str, prompt: str, key: str) -> dict:
        if key in self.correction_cache:
            return self.correction_cache[key]
        st = time.perf_counter()
        response = self.client.responses.create(
            model=self.model, instructions=system, input=prompt,
            text={"format": {"type": "json_schema", "name": "corrected_dialogue", "strict": True,
                "schema": {"type": "object", "properties": {
                    "history": {"type": "array", "items": {"type": "string"}}, "question": {"type": "string"}},
                    "required": ["history", "question"], "additionalProperties": False}}})
        elapsed = time.perf_counter() - st
        obj = json.loads(response.output_text)
        row = {"key": key, "history": obj["history"], "question": obj["question"],
               "latency_s": elapsed,
               "tokens": int(getattr(getattr(response, "usage", None), "total_tokens", 0) or 0)}
        self.correction_cache[key] = row
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def correct_many(self, jobs: list[dict], concurrency: int) -> dict[str, dict]:
        pending = [j for j in jobs if j["key"] not in self.correction_cache]
        results = {}
        if pending:
            with ThreadPoolExecutor(max_workers=min(concurrency, len(pending))) as pool:
                futures = {pool.submit(self.correct, j["system"], j["prompt"], j["key"]): j for j in pending}
                for future in as_completed(futures):
                    job = futures[future]
                    results[job["key"]] = future.result()
        for job in jobs:
            if job["key"] not in results:
                results[job["key"]] = self.correction_cache[job["key"]]
        return results

    def corrected_route(self, case: dict, packed: dict, correction_key: str,
                        existing: dict, rows_path: Path) -> dict:
        arm = packed.get("correction_arm", "correct_then_route")
        row_key = (case["id"], "typo", arm)
        if row_key in existing:
            return existing[row_key]
        corrected = self.correct(packed["correction_system"], packed["input"], correction_key)
        history = corrected["history"]
        if len(history) != len(case["history"]):
            history = case["history"]
        route_input = "\n".join(f"USER (earlier): {m}" for m in history)
        if route_input: route_input += "\n"
        route_input += "USER (current): " + corrected["question"]
        route_key = json.dumps([self.model, "corrected_route", case["sample"], case["id"], route_input], ensure_ascii=False)
        routed = self.choose(packed["route_system"], route_input, route_key)
        pred = sorted(set(i for i in routed["indices"] if 0 <= i < len(TOOLS)))
        gold_idx = [TOOLS.index(case["gold_tool"])]
        overlap = len(set(pred) & set(gold_idx))
        row = {"case_id": case["id"], "sample": case["sample"], "domain": case["domain"],
               "condition": "typo", "prompt_arm": arm, "gold_indices": gold_idx,
               "predicted_indices": pred, "correct": int(set(pred) == set(gold_idx)),
               "tool_f1": round(2 * overlap / (len(pred) + len(gold_idx)), 4) if pred or gold_idx else 1.0,
               "typo_edit": packed["typo_edit"],
               "latency_s": corrected["latency_s"] + routed["latency_s"],
               "tokens": corrected["tokens"] + routed["tokens"],
               "corrected_question": corrected["question"]}
        with rows_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        existing[row_key] = row
        return row

    def choose_batch(self, jobs: list[dict]) -> list[dict]:
        """Submit independent cases in one API request using the Responses batch input format."""
        uncached = [j for j in jobs if j["key"] not in self.cache]
        if not uncached:
            return [self.cache[j["key"]] for j in jobs]
        payload = [{"custom_id": str(i), "method": "POST", "url": "/v1/responses",
                    "body": {"model": self.model, "instructions": j["system"], "input": j["prompt"],
                             "text": {"format": {"type": "json_schema", "name": "tool_indices", "strict": True,
                                  "schema": {"type": "object", "properties": {"indices": {
                                      "type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 3}}},
                                      "required": ["indices"], "additionalProperties": False}}}}}
                   for i, j in enumerate(uncached)]
        started = time.perf_counter()
        batch_input = self.client.files.create(file=("batch.jsonl", "\n".join(json.dumps(x) for x in payload)), purpose="batch")
        batch = self.client.batches.create(input_file_id=batch_input.id, endpoint="/v1/responses",
                                           completion_window="24h")
        print(f"Submitted batch with {len(uncached)} requests (batch id {batch.id})", flush=True)
        while True:
            status = self.client.batches.retrieve(batch.id)
            if status.status in ("completed", "failed", "expired", "cancelled"): break
            time.sleep(10)
        if status.status != "completed" or not status.output_file_id:
            raise RuntimeError(f"Batch {batch.id} ended with status={status.status}; errors={status.errors}")
        output = self.client.files.content(status.output_file_id).text
        response_map = {}
        for line in output.splitlines():
            if line.strip():
                result = json.loads(line)
                response_map[result["custom_id"]] = result
        elapsed_each = max(0.0, time.perf_counter() - started) / len(uncached)
        for i, job in enumerate(uncached):
            item = response_map.get(str(i))
            if not item or item.get("error"):
                raise RuntimeError(f"Batch item {i} failed: {item}")
            body = item["response"]["body"]
            output_text = next((part.get("text") for block in body.get("output", [])
                                for part in block.get("content", []) if part.get("type") == "output_text"), None)
            if not output_text:
                raise RuntimeError(f"Batch item {i} had no output text")
            obj = json.loads(output_text)
            usage = body.get("usage", {})
            row = {"key": job["key"], "indices": obj["indices"], "latency_s": elapsed_each,
                   "tokens": int(usage.get("total_tokens", 0) or 0), "batch_id": batch.id}
            self.cache[job["key"]] = row
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        return [self.cache[j["key"]] for j in jobs]


def summarize(rows: list[dict], out: Path) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["sample"], row["condition"], row["prompt_arm"])].append(row)
    summary = []
    for (sample, condition, arm), batch in sorted(groups.items()):
        summary.append({"sample": sample, "condition": condition, "prompt_arm": arm,
                        "n": len(batch), "tool_set_accuracy": round(sum(r["correct"] for r in batch) / len(batch), 4),
                        "mean_tool_f1": round(sum(r["tool_f1"] for r in batch) / len(batch), 4),
                        "mean_latency_s": round(sum(r["latency_s"] for r in batch) / len(batch), 4),
                        "mean_tokens": round(sum(r["tokens"] for r in batch) / len(batch), 2)})
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    paired = []
    samples = sorted({r["sample"] for r in rows})
    for sample in samples:
        for condition in ("clean", "typo"):
            direct = {f'{r["sample"]}:{r["case_id"]}': r for r in rows if r["sample"] == sample and r["condition"] == condition and r["prompt_arm"] == "direct"}
            aware = {f'{r["sample"]}:{r["case_id"]}': r for r in rows if r["sample"] == sample and r["condition"] == condition and r["prompt_arm"] == "typo_aware"}
            ids = sorted(set(direct) & set(aware))
            if ids:
                delta = sum(aware[i]["correct"] - direct[i]["correct"] for i in ids)
                paired.append({"sample": sample, "condition": condition, "n_pairs": len(ids),
                               "typo_aware_minus_direct_correct": delta,
                               "accuracy_delta": round(delta / len(ids), 4),
                               "direct_accuracy": round(sum(direct[i]["correct"] for i in ids) / len(ids), 4),
                                "typo_aware_accuracy": round(sum(aware[i]["correct"] for i in ids) / len(ids), 4)})
            for correction_arm in ("correct_then_route", "correct_then_typo_aware"):
                corrected = {f'{r["sample"]}:{r["case_id"]}': r for r in rows if r["sample"] == sample and r["condition"] == condition and r["prompt_arm"] == correction_arm}
                if corrected:
                    ids2 = sorted(set(direct) & set(corrected))
                    delta2 = sum(corrected[i]["correct"] - direct[i]["correct"] for i in ids2)
                    paired.append({"sample": sample, "condition": condition,
                                   "comparison": f"{correction_arm}_minus_direct", "n_pairs": len(ids2),
                                   "corrected_minus_direct_correct": delta2,
                                   "accuracy_delta": round(delta2 / len(ids2), 4),
                                   "direct_accuracy": round(sum(direct[i]["correct"] for i in ids2) / len(ids2), 4),
                                   "comparison_accuracy": round(sum(corrected[i]["correct"] for i in ids2) / len(ids2), 4)})
    with (out / "paired.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for row in paired for k in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(paired)
    report = ["# Tool-call typo experiment", "", "Metric: exact set accuracy over selected tool indices.", "",
              "| Sample | Input | Prompt | n | Exact accuracy | Tool F1 | Mean latency (s) | Mean tokens |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        report.append("| " + " | ".join(str(r[k]) for k in ("sample", "condition", "prompt_arm", "n", "tool_set_accuracy", "mean_tool_f1", "mean_latency_s", "mean_tokens")) + " |")
    report += ["", "## Paired accuracy difference", "", "| Sample | Input | Comparison | Pairs | Direct | Other arm | Difference |", "|---|---|---|---:|---:|---:|---:|"]
    for r in paired:
        other = r.get("typo_aware_accuracy", r.get("comparison_accuracy"))
        comparison = r.get("comparison", "typo_aware_minus_direct")
        report.append("| " + " | ".join(str(x) for x in (r["sample"], r["condition"], comparison, r["n_pairs"], r["direct_accuracy"], other, r["accuracy_delta"])) + " |")
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("BENCH_MODEL", "gpt-6-luna"))
    parser.add_argument("--out", default="./results")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=25,
                        help="Concurrent synchronous API requests per wave; use 1 to disable parallel calls")
    parser.add_argument("--batch-id", action="append", default=[], help="Poll and collect a previously submitted Batch API ID")
    parser.add_argument("--correction-ablation", action="store_true",
                        help="Also compare typo input after a separate correction call; only runs the typo condition")
    parser.add_argument("--diverse-singleton", action="store_true",
                        help="Run only 100 single-turn cases with mixed, stronger typo corruption into a new output directory")
    parser.add_argument("--diverse-multiturn", action="store_true",
                        help="Run only 100 multi-turn cases with mixed, stronger typo corruption into a new output directory")
    parser.add_argument("--correction-typo-aware", action="store_true",
                        help="Add a separate correction followed by a typo-aware router arm")
    args = parser.parse_args()
    out = Path(args.out)
    data_dir = out / "data"
    docs, dial = load_data(data_dir)
    cases = select_cases(docs, dial, args.seed)
    if args.diverse_singleton or args.diverse_multiturn:
        target_sample = "single" if args.diverse_singleton else "multi"
        cases = [c for c in cases if c["sample"] == target_sample]
        if len(cases) != 100:
            raise RuntimeError(f"Expected 100 distinct {target_sample} cases, got {len(cases)}")
    (out / "selected_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(cases)} cases: {sum(c['sample']=='multi' for c in cases)} multi-turn, {sum(c['sample']=='single' for c in cases)} single-turn")
    if args.prepare_only:
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    runner = Runner(args.model, out / "model_cache.jsonl")
    rows_path = out / "rows.jsonl"
    existing = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); existing[(r["case_id"], r["condition"], r["prompt_arm"])] = r
    if args.diverse_singleton or args.diverse_multiturn:
        diverse_tag = "diverse_single" if args.diverse_singleton else "diverse_multi"
        rows_path = out / ("diverse_singleton_rows.jsonl" if args.diverse_singleton else "diverse_multiturn_rows.jsonl")
        existing = {}
        if rows_path.exists():
            for line in rows_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    existing[(r["case_id"], r["condition"], r["prompt_arm"])] = r
        runner = Runner(args.model, out / f"{diverse_tag}_model_cache.jsonl")

    # The first batch was submitted before local job manifests were added. Recover its
    # case mapping from the deterministic ordering used by that initial invocation.
    legacy_ids = {"batch_6ad7825bbb4c8190ae1c6d05e4dc1ad4",
                  "batch_6ad782da91b481908ca6ef347e4e7e29",
                  "batch_6ad7842018d88190abbd71d9bce2b2b4"}

    def record_output(batch_id: str, item: dict, case: dict, condition: str, arm: str,
                      packed: dict, cache_key: str) -> None:
        if item.get("error") or not item.get("response"):
            raise RuntimeError(f"Batch item {item.get('custom_id')} failed: {item.get('error')}")
        body = item["response"]["body"]
        output_text = next((p.get("text") for b in body.get("output", [])
                            for p in b.get("content", []) if p.get("type") == "output_text"), None)
        if not output_text:
            raise RuntimeError(f"No output text for batch item {item.get('custom_id')}")
        indices_obj = json.loads(output_text)
        usage = body.get("usage", {})
        indices = sorted(set(i for i in indices_obj["indices"] if 0 <= i < len(TOOLS)))
        gold_idx = [TOOLS.index(case["gold_tool"])]
        overlap = len(set(indices) & set(gold_idx))
        tokens = int(usage.get("total_tokens", 0) or 0)
        cached_answer = {"key": cache_key, "indices": indices, "latency_s": 0.0,
                         "tokens": tokens, "batch_id": batch_id}
        runner.cache[cache_key] = cached_answer
        with runner.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cached_answer) + "\n")
        row = {"case_id": case["id"], "sample": case["sample"], "domain": case["domain"],
               "condition": condition, "prompt_arm": arm, "gold_indices": gold_idx,
               "predicted_indices": indices, "correct": int(set(indices) == set(gold_idx)),
               "tool_f1": round(2 * overlap / (len(indices) + len(gold_idx)), 4) if indices or gold_idx else 1.0,
               "typo_edit": packed["typo_edit"], "latency_s": 0.0, "tokens": tokens,
               "batch_id": batch_id}
        existing[(case["id"], condition, arm)] = row

    def collect_legacy_batch(batch_id: str) -> None:
        batch = runner.client.batches.retrieve(batch_id)
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            print(f"Batch {batch_id}: {batch.status}, completed={batch.request_counts.completed}/{batch.request_counts.total}", flush=True)
            time.sleep(20)
            batch = runner.client.batches.retrieve(batch_id)
        if batch.status != "completed" or not batch.output_file_id:
            raise RuntimeError(f"Batch {batch_id} ended with status={batch.status}; errors={batch.errors}")
        output = runner.client.files.content(batch.output_file_id).text
        answer_rows = {x["custom_id"]: x for x in (json.loads(line) for line in output.splitlines() if line.strip())}
        ordered_jobs = [(case, condition, arm) for case in cases for condition in ("clean", "typo")
                        for arm in ("direct", "typo_aware")]
        position = {"batch_6ad7825bbb4c8190ae1c6d05e4dc1ad4": 0,
                    "batch_6ad782da91b481908ca6ef347e4e7e29": 100,
                    "batch_6ad7842018d88190abbd71d9bce2b2b4": 200}[batch_id]
        if batch_id == "batch_6ad7842018d88190abbd71d9bce2b2b4":
            # 16 initial rows had already been written before the first interruption.
            ordered_jobs = [x for x in ordered_jobs if (x[0]["id"], x[1], x[2]) not in existing]
            position = 0
        for n in range(batch.request_counts.total):
            if str(n) not in answer_rows:
                continue
            case, condition, arm = ordered_jobs[position + n]
            system, packed = make_prompt(case, condition == "typo")
            cache_key = json.dumps([args.model, (case["id"], condition, arm), system, packed["question"]], ensure_ascii=False)
            record_output(batch_id, answer_rows[str(n)], case, condition, arm, packed, cache_key)

    for legacy_id in args.batch_id:
        if legacy_id in legacy_ids:
            collect_legacy_batch(legacy_id)
            with rows_path.open("w", encoding="utf-8") as f:
                for row in existing.values():
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Collected {legacy_id}; saved {len(existing)} rows", flush=True)
            continue

    def collect_batch(batch_id: str) -> None:
        batch = runner.client.batches.retrieve(batch_id)
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            print(f"Batch {batch_id}: {batch.status}, completed={batch.request_counts.completed}/{batch.request_counts.total}", flush=True)
            time.sleep(20)
            batch = runner.client.batches.retrieve(batch_id)
        if batch.status != "completed" or not batch.output_file_id:
            raise RuntimeError(f"Batch {batch_id} ended with status={batch.status}; errors={batch.errors}")
        content = runner.client.files.content(batch.output_file_id).text
        output_map = {x["custom_id"]: x for x in (json.loads(line) for line in content.splitlines() if line.strip())}
        manifest = out / "submitted_batches.jsonl"
        submissions = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        submission = next((x for x in submissions if x["batch_id"] == batch_id), None)
        if submission is None:
            raise RuntimeError(f"No local job manifest for batch {batch_id}")
        for item in submission["jobs"]:
            result = output_map.get(item["custom_id"])
            if not result or result.get("error"):
                raise RuntimeError(f"Batch item {item['custom_id']} failed: {result}")
            body = result["response"]["body"]
            output_text = next((p.get("text") for b in body.get("output", [])
                                for p in b.get("content", []) if p.get("type") == "output_text"), None)
            if not output_text:
                raise RuntimeError(f"No output text for batch item {item['custom_id']}")
            obj = json.loads(output_text)
            usage = body.get("usage", {})
            answer = {"key": item["cache_key"], "indices": obj["indices"],
                      "latency_s": 0.0, "tokens": int(usage.get("total_tokens", 0) or 0), "batch_id": batch_id}
            runner.cache[item["cache_key"]] = answer
            with runner.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(answer) + "\n")
            case, condition, arm, packed = item["case"], item["condition"], item["arm"], item["packed"]
            indices = sorted(set(i for i in answer["indices"] if 0 <= i < len(TOOLS)))
            gold_idx = [TOOLS.index(case["gold_tool"])]
            overlap = len(set(indices) & set(gold_idx))
            f1 = 2 * overlap / (len(indices) + len(gold_idx)) if indices or gold_idx else 1.0
            row = {"case_id": case["id"], "sample": case["sample"], "domain": case["domain"],
                   "condition": condition, "prompt_arm": arm, "gold_indices": gold_idx,
                   "predicted_indices": indices, "correct": int(set(indices) == set(gold_idx)),
                   "tool_f1": round(f1, 4), "typo_edit": packed["typo_edit"],
                   "latency_s": 0.0, "tokens": answer["tokens"], "batch_id": batch_id}
            row_key = tuple(item["row_key"])
            existing[row_key] = row
        with rows_path.open("w", encoding="utf-8") as f:
            for row in existing.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Collected batch {batch_id}; now have {len(existing)} scored rows", flush=True)

    for batch_id in args.batch_id:
        collect_batch(batch_id)
    if args.batch_id:
        remaining = any((case["id"], condition, arm) not in existing
                        for case in cases for condition in ("clean", "typo") for arm in ("direct", "typo_aware"))
        if not remaining:
            rows = [existing[(case["id"], condition, arm)] for case in cases
                    for condition in ("clean", "typo") for arm in ("direct", "typo_aware")]
            summarize(rows, out)
            print(f"Completed {len(rows)} rows; report: {out / 'report.md'}")
        return
    total = len(cases) * (4 + int(args.correction_ablation) + int(args.correction_typo_aware))
    if args.diverse_multiturn:
        rows_path = out / "diverse_multiturn_rows.jsonl"
        existing = {}
        if rows_path.exists():
            for line in rows_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    existing[(r["case_id"], r["condition"], r["prompt_arm"])] = r
        runner = Runner(args.model, out / "diverse_multiturn_model_cache.jsonl")
    jobs = []
    for case in cases:
        for condition in ("clean", "typo"):
            for arm in ("direct", "typo_aware"):
                row_key = (case["id"], condition, arm)
                if row_key in existing:
                    continue
                system, packed = make_prompt(case, condition == "typo", diverse=(args.diverse_singleton or args.diverse_multiturn))
                cache_key = json.dumps([args.model, row_key, system, packed["question"]], ensure_ascii=False)
                jobs.append({"row_key": row_key, "case": case, "condition": condition, "arm": arm,
                             "system": system, "prompt": packed["question"], "packed": packed,
                             "key": cache_key})
        if args.correction_ablation:
            condition = "typo"
            arm = "correct_then_route"
            row_key = (case["id"], condition, arm)
            if row_key not in existing:
                fix_system, packed = correct_prompt(case, True, diverse=(args.diverse_singleton or args.diverse_multiturn))
                fix_key = json.dumps([args.model, "correction", case["sample"], case["id"], packed["input"]], ensure_ascii=False)
                packed["correction_system"] = fix_system
                packed["correction_arm"] = arm
                jobs.append({"row_key": row_key, "case": case, "condition": condition, "arm": arm,
                             "system": fix_system, "prompt": packed["input"], "packed": packed,
                             "key": fix_key, "correction_job": True})
        if args.correction_typo_aware:
            condition = "typo"
            arm = "correct_then_typo_aware"
            row_key = (case["id"], condition, arm)
            if row_key not in existing:
                fix_system, packed = correct_prompt(case, True, diverse=(args.diverse_singleton or args.diverse_multiturn))
                fix_key = json.dumps([args.model, "correction_typo_aware", case["sample"], case["id"], packed["input"]], ensure_ascii=False)
                packed["correction_system"] = fix_system
                packed["correction_arm"] = arm
                packed["route_system"] += (" The dialogue may contain spelling mistakes. Infer intended words from context "
                                            "and account for likely typos while selecting the tool indices.")
                jobs.append({"row_key": row_key, "case": case, "condition": condition, "arm": arm,
                             "system": fix_system, "prompt": packed["input"], "packed": packed,
                             "key": fix_key, "correction_job": True})
    batch_size = max(1, args.batch_size)
    # Use parallel synchronous Responses requests instead of the asynchronous Batch API.
    # This keeps the same one-request-per-example design while allowing several calls at once.
    concurrency = max(1, args.concurrency)
    for offset in range(0, len(jobs), batch_size):
        batch_jobs = jobs[offset:offset + batch_size]
        answers_by_key = {}
        correction_jobs = [job for job in batch_jobs if job.get("correction_job")]
        correction_results = runner.correct_many(correction_jobs, concurrency) if correction_jobs else {}
        standard_jobs = [job for job in batch_jobs if not job.get("correction_job")]
        with ThreadPoolExecutor(max_workers=min(concurrency, max(1, len(standard_jobs)))) as pool:
            future_jobs = {pool.submit(runner.choose, job["system"], job["prompt"], job["key"]): job
                           for job in standard_jobs}
            for future in as_completed(future_jobs):
                job = future_jobs[future]
                answers_by_key[job["key"]] = future.result()
        for job in batch_jobs:
                if job.get("correction_job"):
                    continue
                answer = answers_by_key[job["key"]]
                case, condition, arm, packed = job["case"], job["condition"], job["arm"], job["packed"]
                pred = sorted(set(i for i in answer["indices"] if 0 <= i < len(TOOLS)))
                gold_idx = [TOOLS.index(case["gold_tool"])]
                overlap = len(set(pred) & set(gold_idx))
                f1 = 2 * overlap / (len(pred) + len(gold_idx)) if pred or gold_idx else 1.0
                row = {"case_id": case["id"], "sample": case["sample"], "domain": case["domain"],
                       "condition": condition, "prompt_arm": arm, "gold_indices": gold_idx,
                       "predicted_indices": pred, "correct": int(set(pred) == set(gold_idx)),
                       "tool_f1": round(f1, 4), "typo_edit": packed["typo_edit"],
                       "latency_s": answer["latency_s"], "tokens": answer["tokens"]}
                with rows_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing[job["row_key"]] = row
                print(f"[{len(existing)}/{total}] {case['sample']} {condition} {arm}: correct={row['correct']}", flush=True)
        if correction_jobs:
            with ThreadPoolExecutor(max_workers=min(concurrency, len(correction_jobs))) as pool:
                route_futures = {}
                for job in correction_jobs:
                    case, packed = job["case"], job["packed"]
                    corrected = correction_results[job["key"]]
                    history = corrected["history"] if len(corrected["history"]) == len(case["history"]) else case["history"]
                    route_input = "\n".join(f"USER (earlier): {m}" for m in history)
                    if route_input: route_input += "\n"
                    route_input += "USER (current): " + corrected["question"]
                    route_key = json.dumps([args.model, "corrected_route", case["sample"], case["id"], route_input], ensure_ascii=False)
                    route_futures[pool.submit(runner.choose, packed["route_system"], route_input, route_key)] = (job, corrected)
                for future in as_completed(route_futures):
                    job, corrected = route_futures[future]
                    routed = future.result()
                    case, packed, arm = job["case"], job["packed"], job["arm"]
                    pred = sorted(set(i for i in routed["indices"] if 0 <= i < len(TOOLS)))
                    gold_idx = [TOOLS.index(case["gold_tool"])]
                    overlap = len(set(pred) & set(gold_idx))
                    row = {"case_id": case["id"], "sample": case["sample"], "domain": case["domain"],
                           "condition": "typo", "prompt_arm": arm, "gold_indices": gold_idx,
                           "predicted_indices": pred, "correct": int(set(pred) == set(gold_idx)),
                           "tool_f1": round(2 * overlap / (len(pred) + len(gold_idx)), 4) if pred or gold_idx else 1.0,
                           "typo_edit": packed["typo_edit"], "latency_s": corrected["latency_s"] + routed["latency_s"],
                           "tokens": corrected["tokens"] + routed["tokens"], "corrected_question": corrected["question"]}
                    with rows_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    existing[job["row_key"]] = row
                    print(f"[{len(existing)}/{total}] single typo {arm}: correct={row['correct']}", flush=True)
    rows = [existing[(case["id"], condition, arm)] for case in cases
            for condition in ("clean", "typo") for arm in ("direct", "typo_aware")
            if (case["id"], condition, arm) in existing]
    if args.correction_ablation or args.correction_typo_aware:
        correction_arms = []
        if args.correction_ablation: correction_arms.append("correct_then_route")
        if args.correction_typo_aware: correction_arms.append("correct_then_typo_aware")
        for arm in correction_arms:
            rows.extend(existing[(case["id"], "typo", arm)] for case in cases
                        if (case["id"], "typo", arm) in existing)
    summarize(rows, out)
    print(f"Completed {len(rows)} calls; report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
