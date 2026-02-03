"""Evaluate LLM responses against heuristic gold labels and produce summary metrics."""
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import json
import sys

# Support running as script or as installed package
try:
    from app.heuristic_gold import build_gold_map
except Exception:
    try:
        from heuristic_gold import build_gold_map
    except Exception:
        # last-resort: add app directory to path
        sys.path.append(str(Path(__file__).resolve().parent))
        from heuristic_gold import build_gold_map


def normalize_answer(q_short: str, ans: str) -> str:
    if ans is None:
        return ""
    a = ans.strip()
    if q_short == "highest_degree":
        return a.lower()
    if q_short == "years_experience":
        return a.strip().lower()
    if q_short == "has_python":
        return a.strip().lower()
    return a.lower()


def compare(q_short: str, pred: str, gold: str) -> bool:
    pp = normalize_answer(q_short, pred)
    gg = normalize_answer(q_short, gold)
    if gg == "unknown":
        return pp == "unknown"
    if q_short == "years_experience":
        # compare integers if possible
        try:
            return int(pp) == int(gg)
        except Exception:
            return pp == gg
    return pp == gg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resumes", default="data/resumes.csv")
    parser.add_argument("--results", default="results/results_sample_100.csv")
    parser.add_argument("--out", default="results/results_eval_sample_100.csv")
    parser.add_argument("--summary", default="results/summary_sample_100.json")
    args = parser.parse_args()

    gold_map = build_gold_map(Path(args.resumes))

    results = []
    with open(args.results, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            results.append(r)

    # augment rows with gold and correctness
    for r in results:
        rid = r.get("resume_id")
        q_short = r.get("question_short")
        gold = gold_map.get(rid, {}).get(q_short)
        pred = r.get("response")
        correct = False
        try:
            correct = compare(q_short, pred or "", gold or "Unknown")
        except Exception:
            correct = False
        r["gold"] = gold
        r["correct"] = str(correct)

    # write augmented results
    out_fields = list(results[0].keys()) if results else []
    with open(args.out, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # compute summary
    summary = {}
    stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0, "latencies": []}))
    for r in results:
        llm = r.get("llm")
        q = r.get("question_short")
        corr = r.get("correct") == "True"
        stats[llm][q]["total"] += 1
        if corr:
            stats[llm][q]["correct"] += 1
        lat = r.get("latency_ms")
        try:
            if lat:
                stats[llm][q]["latencies"].append(float(lat))
        except Exception:
            pass

    for llm, qmap in stats.items():
        summary[llm] = {}
        for q, vals in qmap.items():
            total = vals["total"]
            correct = vals["correct"]
            acc = correct / total if total else None
            avg_lat = (sum(vals["latencies"]) / len(vals["latencies"])) if vals["latencies"] else None
            summary[llm][q] = {"accuracy": acc, "total": total, "correct": correct, "avg_latency_ms": avg_lat}

    with open(args.summary, "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote augmented results to {args.out} and summary to {args.summary}")


if __name__ == "__main__":
    main()