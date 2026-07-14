#!/usr/bin/env python3
"""Run and compare reproducible research-extraction benchmark rungs."""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from extract import analyze_paper, sha256_text


def summarize_runs(runs):
    completed = [run for run in runs if not run.get("error")]
    elapsed = [run["metrics"]["elapsed_seconds"] for run in completed]
    completion_tokens = [
        run["metrics"]["completion_tokens"] for run in completed
        if isinstance(run["metrics"].get("completion_tokens"), (int, float))
    ]
    token_rates = [
        run["metrics"]["completion_tokens"] / run["metrics"]["elapsed_seconds"]
        for run in completed
        if isinstance(run["metrics"].get("completion_tokens"), (int, float))
        and run["metrics"]["elapsed_seconds"] > 0
    ]
    grounded = sum(run["grounding"]["grounded"] for run in completed)
    quotes = sum(run["grounding"]["total"] for run in completed)
    return {
        "runs_requested": len(runs),
        "runs_completed": len(completed),
        "valid_json_schema_rate": (
            sum(run["valid"] for run in completed) / len(completed) if completed else 0
        ),
        "grounded_quote_rate": grounded / quotes if quotes else None,
        "median_elapsed_seconds": statistics.median(elapsed) if elapsed else None,
        "median_completion_tokens": (
            statistics.median(completion_tokens) if completion_tokens else None
        ),
        "median_end_to_end_completion_tok_s": (
            statistics.median(token_rates) if token_rates else None
        ),
    }


def current_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def one_run(args, paper_text, schema):
    try:
        result = analyze_paper(
            paper_text, schema, args.endpoint, args.model, args.api_key,
            args.timeout, args.max_tokens,
        )
        return {
            "valid": result["valid"],
            "schema_errors": result["schema_errors"],
            "grounding": result["grounding"],
            "metrics": result["metrics"],
        }
    except (OSError, RuntimeError, ValueError) as exc:
        return {"error": str(exc)}


def run_rung(args):
    paper_text = args.paper.read_text(encoding="utf-8")
    schema_text = args.schema.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    for _ in range(args.warmup):
        warmup = one_run(args, paper_text, schema)
        if warmup.get("error"):
            raise RuntimeError("warm-up failed: " + warmup["error"])
    runs = [one_run(args, paper_text, schema) for _ in range(args.runs)]
    report = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "configuration": {
            "endpoint": args.endpoint,
            "model": args.model,
            "commit": args.commit,
            "hardware": args.hardware,
            "storage": args.storage,
            "os": args.os,
            "paper": str(args.paper),
            "paper_sha256": sha256_text(paper_text),
            "schema": str(args.schema),
            "schema_sha256": sha256_text(schema_text),
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": args.max_tokens,
        },
        "runs": runs,
        "summary": summarize_runs(runs),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["runs_completed"] == args.runs else 2


def percent_change(before, after):
    if before in (None, 0) or after is None:
        return None
    return (after - before) / before * 100


def compare_rungs(args):
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    schema = json.loads(args.schema_run.read_text(encoding="utf-8"))
    for key in ("paper_sha256", "schema_sha256", "model", "commit", "hardware", "storage", "os"):
        left = baseline["configuration"].get(key)
        right = schema["configuration"].get(key)
        if left != right:
            raise ValueError(f"cannot compare reports with different {key}")
    b = baseline["summary"]
    s = schema["summary"]
    latency_delta = percent_change(b["median_elapsed_seconds"], s["median_elapsed_seconds"])
    comparison = {
        "baseline": baseline["label"],
        "schema": schema["label"],
        "latency_change_percent": latency_delta,
        "validity_change_points": (
            s["valid_json_schema_rate"] - b["valid_json_schema_rate"]
        ) * 100,
        "grounding_change_points": (
            None if b["grounded_quote_rate"] is None or s["grounded_quote_rate"] is None
            else (s["grounded_quote_rate"] - b["grounded_quote_rate"]) * 100
        ),
    }
    print("| metric | baseline | schema | change |")
    print("|---|---:|---:|---:|")
    print(f"| median latency (s) | {b['median_elapsed_seconds']:.2f} | {s['median_elapsed_seconds']:.2f} | {latency_delta:+.1f}% |")
    print(f"| valid JSON + schema | {b['valid_json_schema_rate']:.1%} | {s['valid_json_schema_rate']:.1%} | {comparison['validity_change_points']:+.1f} pp |")
    if (b.get("median_end_to_end_completion_tok_s") is not None and
            s.get("median_end_to_end_completion_tok_s") is not None):
        rate_delta = percent_change(
            b["median_end_to_end_completion_tok_s"],
            s["median_end_to_end_completion_tok_s"],
        )
        print(f"| end-to-end completion tok/s | {b['median_end_to_end_completion_tok_s']:.3f} | {s['median_end_to_end_completion_tok_s']:.3f} | {rate_delta:+.1f}% |")
    if b["grounded_quote_rate"] is not None and s["grounded_quote_rate"] is not None:
        print(f"| grounded evidence | {b['grounded_quote_rate']:.1%} | {s['grounded_quote_rate']:.1%} | {comparison['grounding_change_points']:+.1f} pp |")
    if args.output:
        args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="measure one server configuration")
    run.add_argument("paper", type=Path)
    run.add_argument("--schema", type=Path,
                     default=Path(__file__).with_name("paper.schema.json"))
    run.add_argument("--label", required=True)
    run.add_argument("--endpoint", default=os.environ.get("COLI_ENDPOINT", "http://127.0.0.1:8000/v1"))
    run.add_argument("--model", default=os.environ.get("COLI_MODEL_ID", "glm-5.2-colibri"))
    run.add_argument("--commit", default=current_commit())
    run.add_argument("--hardware", required=True,
                     help="CPU, RAM, and GPU description")
    run.add_argument("--storage", required=True,
                     help="model disk, filesystem, and iobench result")
    run.add_argument("--os", default=platform.platform())
    run.add_argument("--api-key", default=os.environ.get("COLI_API_KEY"))
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--runs", type=int, default=3)
    run.add_argument("--max-tokens", type=int, default=1200)
    run.add_argument("--timeout", type=float, default=1800)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(function=run_rung)

    compare = sub.add_parser("compare", help="compare baseline and schema reports")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("schema_run", type=Path)
    compare.add_argument("--output", type=Path)
    compare.set_defaults(function=compare_rungs)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
