"""Monitoring summary from the local trace file. Offline: reads out/traces.jsonl, calls nothing.

Run from app/:
  python -m scripts.trace_summary                  # the whole file
  python -m scripts.trace_summary --last 200       # the 200 most recent requests
  python -m scripts.trace_summary --file other.jsonl

Prints, for the requests in the file: how many, how many failed, latency percentiles, the routes taken;
then per model the calls, errors, latency and tokens; per tool the calls, failures and retries; and how
often the input gate and the safety reviewer stepped in. The same figures a metrics dashboard would
show, computed from the spans the application already writes (src/telemetry.py).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.config import OUT_DIR
from src.telemetry import INPUT_GUARD_SPAN, REVIEWER_SPAN, SpanKind

TRACES = OUT_DIR / "traces.jsonl"


def get_percentile(values: list[float], percent: float) -> float:
    """The percentile `percent` (0..100) of the values, by nearest rank; 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(percent / 100 * len(ordered)) - 1))]


def _format_latency(values: list[float]) -> str:
    return f"p50 {get_percentile(values, 50):,.0f} ms, p95 {get_percentile(values, 95):,.0f} ms, max {max(values, default=0):,.0f} ms"


def load_spans(path: Path, last: int | None = None) -> list[dict]:
    """The spans in the file; with `last`, only those of the most recent `last` traces."""
    spans = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if last:
        recent = list(dict.fromkeys(span["trace_id"] for span in spans if span.get("kind") == SpanKind.TRACE))[-last:]
        spans = [span for span in spans if span["trace_id"] in set(recent)]
    return spans


def summarise(spans: list[dict]) -> dict:
    """The figures printed by main(), as a dict (also used by the tests)."""
    roots = [span for span in spans if span.get("kind") == SpanKind.TRACE]
    models: dict[str, dict] = defaultdict(lambda: {"calls": 0, "errors": 0, "ms": [], "in": 0, "out": 0})
    tools: dict[str, dict] = defaultdict(lambda: {"calls": 0, "failed": 0, "retries": 0, "ms": []})
    for span in spans:
        if span.get("kind") == SpanKind.GENERATION:
            model_stats = models[span.get("model") or "unknown"]
            model_stats["calls"] += 1
            model_stats["errors"] += span.get("status") == "error"
            model_stats["ms"].append(span.get("duration_ms") or 0.0)
            model_stats["in"] += span.get("input_tokens") or 0
            model_stats["out"] += span.get("output_tokens") or 0
        elif span.get("kind") == SpanKind.TOOL:
            tool_stats = tools[span.get("tool") or span.get("name", "?")]
            tool_stats["calls"] += 1
            tool_stats["failed"] += bool(span.get("unavailable")) or span.get("status") == "error"
            tool_stats["retries"] += max(0, (span.get("attempts") or 1) - 1)
            tool_stats["ms"].append(span.get("duration_ms") or 0.0)
    guards = [span for span in spans if span.get("kind") == SpanKind.GUARDRAIL]
    return {
        "requests": len(roots),
        "failed": sum(span.get("status") == "error" for span in roots),
        "latency_ms": [span.get("duration_ms") or 0.0 for span in roots],
        "routes": Counter(span.get("route") or "(chat)" for span in roots),
        "models": dict(models),
        "tools": dict(tools),
        "gate_blocked": sum(span.get("name") == INPUT_GUARD_SPAN and span.get("safe") is False for span in guards),
        "reviewer_vetoes": sum(span.get("name", "").startswith(REVIEWER_SPAN) and span.get("safe") is False for span in guards),
        "breaker_refusals": sum(span.get("attempts") == 0 and bool(span.get("unavailable")) for span in spans),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=TRACES, help="trace file (default: out/traces.jsonl)")
    parser.add_argument("--last", type=int, default=None, help="only the N most recent requests")
    args = parser.parse_args()
    if not args.file.exists():
        raise SystemExit(f"{args.file} not found: run the API or a demo with CAREFLOW_TRACING=jsonl first")
    summary = summarise(load_spans(args.file, args.last))

    print(f"Requests: {summary['requests']}  failed: {summary['failed']}  ({_format_latency(summary['latency_ms'])})")
    print("Routes:   " + ", ".join(f"{route} {count}" for route, count in summary["routes"].most_common()))
    print(f"Safety:   input gate blocked {summary['gate_blocked']}, reviewer vetoes {summary['reviewer_vetoes']}, "
          f"calls refused by an open circuit breaker {summary['breaker_refusals']}")
    print("\nModel calls")
    for name, model_stats in sorted(summary["models"].items()):
        print(f"  {name}: {model_stats['calls']} calls, {model_stats['errors']} errors, {_format_latency(model_stats['ms'])}, "
              f"tokens in {model_stats['in']:,} out {model_stats['out']:,}")
    print("\nTool calls")
    for name, tool_stats in sorted(summary["tools"].items()):
        print(f"  {name}: {tool_stats['calls']} calls, {tool_stats['failed']} failed, {tool_stats['retries']} retries, {_format_latency(tool_stats['ms'])}")


if __name__ == "__main__":
    main()
