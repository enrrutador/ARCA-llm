from __future__ import annotations

import argparse
import json

from arca.app import build_executive
from arca.assistant import CognitiveAssistant
from arca.benchmark import run
from arca.model import Task


def demo() -> None:
    task = Task("cas", {"expression": "(21 + 21) * 2"}, "Compute a verifiable result")
    print(json.dumps(build_executive().execute(task).to_dict(), indent=2, ensure_ascii=False))


def chat(db: str) -> None:
    assistant = CognitiveAssistant(db)
    print("ARCA local cognitive assistant. Type 'help' or 'exit'.")
    while True:
        try:
            text = input("arca> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.casefold() in {"exit", "quit", "salir"}:
            return
        response = assistant.ask(text)
        print(response["answer"])


def main() -> None:
    parser = argparse.ArgumentParser(prog="arca")
    parser.add_argument("command", choices=["demo", "benchmark", "chat", "ask"])
    parser.add_argument("text", nargs="*")
    parser.add_argument("--db", default="arca.db")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    elif args.command == "benchmark":
        report = run()
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    elif args.command == "chat":
        chat(args.db)
    else:
        response = CognitiveAssistant(args.db).ask(" ".join(args.text))
        print(json.dumps(response if args.trace else {"answer": response["answer"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
