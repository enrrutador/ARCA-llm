from __future__ import annotations

import argparse
import json

from arca.app import build_executive
from arca.assistant import CognitiveAssistant
from arca.benchmark import run
from arca.llm import LLMConfig, LocalLLM
from arca.model import Task


def demo() -> None:
    task = Task("cas", {"expression": "(21 + 21) * 2"}, "Compute a verifiable result")
    print(json.dumps(build_executive().execute(task).to_dict(), indent=2, ensure_ascii=False))


def assistant_from_args(args) -> CognitiveAssistant:
    llm = LocalLLM(LLMConfig(args.model, context_size=args.context, threads=args.threads, max_tokens=args.max_tokens)) if args.model else None
    return CognitiveAssistant(args.db, llm=llm)


def chat(args) -> None:
    assistant = assistant_from_args(args)
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
    parser.add_argument("command", choices=["demo", "benchmark", "chat", "ask", "download-model"])
    parser.add_argument("text", nargs="*")
    parser.add_argument("--db", default="arca.db")
    parser.add_argument("--model", help="path to a compatible GGUF model")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    elif args.command == "benchmark":
        report = run()
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    elif args.command == "download-model":
        from arca.llm.download import main as download_main
        download_main()
    elif args.command == "chat":
        chat(args)
    else:
        response = assistant_from_args(args).ask(" ".join(args.text))
        print(json.dumps(response if args.trace else {"answer": response["answer"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
