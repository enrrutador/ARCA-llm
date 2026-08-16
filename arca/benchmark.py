from __future__ import annotations

import argparse
import json
import statistics
from collections import deque
from pathlib import Path
from typing import Any

from arca.app import build_executive
from arca.kernel.budget import Budget
from arca.model import Task


def logic_cases() -> list[tuple[Task, Any]]:
    cases = []
    rules = [
        {"premises": [["parent", "?x", "?y"]], "conclusion": ["ancestor", "?x", "?y"]},
        {"premises": [["parent", "?x", "?y"], ["ancestor", "?y", "?z"]], "conclusion": ["ancestor", "?x", "?z"]},
    ]
    for i in range(30):
        a, b, c = f"a{i}", f"b{i}", f"c{i}"
        positive = i % 3 != 0
        query = ["ancestor", a, c if positive else f"missing{i}"]
        payload = {"facts": [["parent", a, b], ["parent", b, c]], "rules": rules, "query": query}
        cases.append((Task("datalog", payload, "prove ancestry"), positive))
    return cases


def shortest(width: int, height: int, start: tuple[int, int], goal: tuple[int, int], blocked: set[tuple[int, int]]) -> int | None:
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        point, distance = queue.popleft()
        if point == goal:
            return distance
        x, y = point
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nxt[0] < width and 0 <= nxt[1] < height and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def path_cases() -> list[tuple[Task, Any]]:
    cases = []
    for i in range(35):
        width = 7 + i % 3
        height = 7
        gap = i % height
        wall_x = 2 + i % (width - 3)
        blocked = {(wall_x, y) for y in range(height) if y != gap}
        start, goal = (0, i % height), (width - 1, (i * 2) % height)
        expected = shortest(width, height, start, goal, blocked)
        payload = {"width": width, "height": height, "start": start, "goal": goal, "blocked": sorted(blocked)}
        cases.append((Task("astar", payload, "find shortest path"), expected))
    return cases


def arithmetic_cases() -> list[tuple[Task, Any]]:
    cases = []
    for i in range(35):
        a, b, c = i + 2, i % 7 + 1, i % 5 + 2
        expression = f"({a} + {b}) * {c} - {b} ** 2"
        expected = (a + b) * c - b**2
        cases.append((Task("cas", {"expression": expression}, "evaluate expression"), expected))
    return cases


def run() -> dict[str, Any]:
    cases = logic_cases() + path_cases() + arithmetic_cases()
    executive = build_executive()
    rows = []
    for index, (task, expected) in enumerate(cases):
        record = executive.execute(task, Budget(max_seconds=1.5, max_rss_mb=256))
        if task.kind == "astar":
            actual = None if record.result is None else len(record.result) - 1
        else:
            actual = record.result
        rows.append({
            "index": index,
            "kind": task.kind,
            "correct": record.telemetry["success"] and actual == expected,
            "latency_ms": record.telemetry["latency_ms"],
            "python_peak_mb": record.telemetry["python_peak_mb"],
            "rss_peak_mb": record.telemetry["rss_peak_mb"],
            "trace_steps": len(record.trace),
        })
    latencies = [r["latency_ms"] for r in rows]
    correct = [r["correct"] for r in rows]
    first = sum(correct[:20]) / 20
    last = sum(correct[-20:]) / 20
    ordered = sorted(latencies)
    report = {
        "cases": len(rows),
        "mix": {"datalog": 30, "astar": 35, "cas": 35},
        "accuracy": sum(correct) / len(rows),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "python_peak_mb": max(r["python_peak_mb"] for r in rows),
        "rss_peak_mb": max(r["rss_peak_mb"] for r in rows),
        "trace_coverage": sum(r["trace_steps"] > 0 for r in rows) / len(rows),
        "first_20_accuracy": first,
        "last_20_accuracy": last,
        "degradation": first - last,
        "targets": {
            "accuracy_gte_0_80": sum(correct) / len(rows) >= 0.80,
            "rss_lte_256_mb": max(r["rss_peak_mb"] for r in rows) <= 256,
            "median_latency_lte_1500_ms": statistics.median(latencies) <= 1500,
            "full_traceability": all(r["trace_steps"] > 0 for r in rows),
            "non_degradation": last >= first - 0.05,
        },
        "rows": rows,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic ARCA M0 benchmark")
    parser.add_argument("--json", type=Path, help="write the complete report to this file")
    args = parser.parse_args()
    report = run()
    summary = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
