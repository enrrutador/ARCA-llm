from __future__ import annotations

import argparse
import json
from pathlib import Path

from arca.app import build_executive
from arca.assistant import CognitiveAssistant
from arca.benchmark import run
from arca.model import Task
from arca.native_lm import ARCALanguageModel


def model_from_args(path: str | None):
    return ARCALanguageModel.load(Path(path)) if path else None


def demo() -> None:
    task = Task("cas", {"expression": "(21 + 21) * 2"}, "Compute a verifiable result")
    print(json.dumps(build_executive().execute(task).to_dict(), indent=2, ensure_ascii=False))


def chat(args) -> None:
    assistant = CognitiveAssistant(args.db, model=model_from_args(args.model))
    print("ARCA native language model. Type 'help' or 'exit'.")
    while True:
        try:
            text = input("arca> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.casefold() in {"exit", "quit", "salir"}:
            return
        print(assistant.ask(text)["answer"])


def main() -> None:
    parser = argparse.ArgumentParser(prog="arca")
    parser.add_argument("command", choices=["demo", "benchmark", "chat", "ask"])
    parser.add_argument("text", nargs="*")
    parser.add_argument("--db", default="arca.db")
    parser.add_argument("--model", help="path to ARCA-native .npz weights")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    elif args.command == "benchmark":
        report = run()
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    elif args.command == "chat":
        chat(args)
    else:
        response = CognitiveAssistant(args.db, model=model_from_args(args.model)).ask(" ".join(args.text))
        print(json.dumps(response if args.trace else {"answer": response["answer"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
