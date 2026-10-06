"""Run an extractor over a labelled dataset, cache every prediction, and score it.

    python -m eval.run_eval --dataset invoices
    python -m eval.run_eval --dataset cord --limit 30
    python -m eval.run_eval --dataset all --provider ollama --model qwen2.5vl:3b

Predictions are cached per model under eval/predictions/, so a run that hits the free-tier
daily limit can simply be restarted and continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

from eval.datasets import DATASETS, DATE_ORDER, load_samples
from eval.metrics import aggregate, score_document
from invoice_extractor.extractors import ExtractionError, get_extractor
from invoice_extractor.postprocess import finalize
from invoice_extractor.schema import Document
from invoice_extractor.validation import check

ROOT = Path(__file__).parent
PRED_DIR = ROOT / "predictions"
RESULT_DIR = ROOT / "results"


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", name).strip("-")


def predict(extractor, sample, cache_dir: Path, rerun: bool) -> dict:
    path = cache_dir / f"{sample.id}.json"
    if path.exists() and not rerun:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        res = extractor.extract(sample.payload)
        record = {"document": res.document.model_dump(), "latency_s": res.latency_s,
                  "usage": res.usage, "error": None}
    except ExtractionError as exc:
        if "429" in str(exc):
            # Daily quota exhausted: stop instead of caching a failure.
            raise SystemExit(f"\nRate limit reached at {sample.id}: {exc}\nRe-run later to resume.")
        record = {"document": None, "latency_s": None, "usage": {}, "error": str(exc)}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def routing_stats(rows: list[dict]) -> dict:
    """How well 'no validation issues' predicts 'extraction fully correct'."""
    accepted = [r for r in rows if not r["flagged"]]
    wrong = [r for r in rows if not r["exact"]]
    return {
        "auto_accept_rate": len(accepted) / len(rows) if rows else 0.0,
        "auto_accept_precision": (sum(r["exact"] for r in accepted) / len(accepted)) if accepted else None,
        "errors_caught": (sum(r["flagged"] for r in wrong) / len(wrong)) if wrong else None,
    }


def run(dataset: str, extractor, limit: int | None, rerun: bool, fix_labels: bool = True) -> dict:
    cache_dir = PRED_DIR / slug(extractor.name) / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    scores, rows, latencies, failures = [], [], [], 0
    total = limit or "all"
    for i, sample in enumerate(load_samples(dataset, limit=limit, fix_labels=fix_labels), 1):
        rec = predict(extractor, sample, cache_dir, rerun)
        doc = Document.model_validate(rec["document"]) if rec["document"] else None
        if doc:
            doc = finalize(doc, DATE_ORDER[dataset])
        failures += doc is None
        if rec["latency_s"]:
            latencies.append(rec["latency_s"])
        score = score_document(doc, sample.truth, sample.truth_items)
        issues = [x.code for x in check(doc)] if doc else ["extraction_failed"]
        scores.append(score)
        rows.append({"id": sample.id, "exact": score.exact, "flagged": bool(issues), "issues": issues,
                     "wrong_fields": [k for k, ok in score.fields.items() if not ok],
                     "items": f"{score.items_correct}/{score.items_truth} (pred {score.items_pred})"})
        print(f"\r{dataset}: {i}/{total}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    summary = aggregate(scores)
    summary.update({
        "dataset": dataset, "model": extractor.name, "failures": failures,
        "labels": "fixed" if fix_labels else "raw",
        "median_latency_s": statistics.median(latencies) if latencies else None,
        "routing": routing_stats(rows),
    })
    RESULT_DIR.mkdir(exist_ok=True)
    suffix = "" if fix_labels else "__raw-labels"
    out = RESULT_DIR / f"{slug(extractor.name)}__{dataset}{suffix}.json"
    out.write_text(json.dumps({"summary": summary, "documents": rows}, indent=1), encoding="utf-8")
    return summary


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def print_summary(s: dict) -> None:
    r = s["routing"]
    print(f"\n## {s['model']} on {s['dataset']} ({s['documents']} docs, {s['failures']} failed, "
          f"{s['labels']} labels)")
    print("| field | accuracy |\n|---|---|")
    for k, v in s["field_accuracy"].items():
        print(f"| {k} | {pct(v)} |")
    print(f"| **all fields (micro)** | **{pct(s['field_accuracy_micro'])}** |")
    print(f"\nline items: precision {pct(s['item_precision'])}, recall {pct(s['item_recall'])}, "
          f"F1 {pct(s['item_f1'])}")
    print(f"documents fully correct: {pct(s['doc_exact'])}")
    print(f"auto-accepted (no validation issue): {pct(r['auto_accept_rate'])}, of which correct: "
          f"{pct(r['auto_accept_precision'])}; wrong documents flagged: {pct(r['errors_caught'])}")
    if s["median_latency_s"]:
        print(f"median latency: {s['median_latency_s']:.1f} s")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    ap.add_argument("--provider", default=None, help="gemini (default) or ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rerun", action="store_true", help="ignore cached predictions")
    ap.add_argument("--raw-labels", action="store_true", help="score against the original labels")
    args = ap.parse_args()

    extractor = get_extractor(args.provider, args.model)
    for name in DATASETS if args.dataset == "all" else [args.dataset]:
        print_summary(run(name, extractor, args.limit, args.rerun, not args.raw_labels))


if __name__ == "__main__":
    main()
