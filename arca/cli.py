from __future__ import annotations

import argparse
import json

from arca.app import build_executive
from arca.benchmark import run
from arca.model import Task


def demo() -> None:
    task = Task(
        "datalog",
        {
            "facts": [["parent", "ana", "bea"], ["parent", "bea", "carla"]],
            "rules": [
                {"premises": [["parent", "?x", "?y"]], "conclusion": ["ancestor", "?x", "?y"]},
                {"premises": [["parent", "?x", "?y"], ["ancestor", "?y", "?z"]], "conclusion": ["ancestor", "?x", "?z"]},
            ],
            "query": ["ancestor", "ana", "carla"],
        },
        "Is Ana an ancestor of Carla?",
    )
    record = build_executive().execute(task)
    print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(prog="arca")
    parser.add_argument("command", choices=["demo", "benchmark"])
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    else:
        report = run()
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
