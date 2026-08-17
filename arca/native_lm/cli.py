from __future__ import annotations

import argparse
import json
from pathlib import Path

from arca.memory import MemoryStore
from arca.native_lm.corpus import CorpusStore
from arca.native_lm.learner import WebLearner
from arca.native_lm.model import ARCALanguageModel, ModelConfig
from arca.native_lm.pipeline import NativeTrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run the native ARCA language model")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--corpus", type=Path, default=Path("corpus"))
    train.add_argument("--text", type=Path, help="legacy single text file input")
    train.add_argument("--output", type=Path, default=Path("models/arca-native.npz"))
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--hidden", type=int, default=192)
    train.add_argument("--embedding", type=int, default=64)
    run = sub.add_parser("generate")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("prompt")
    run.add_argument("--tokens", type=int, default=160)
    run.add_argument("--temperature", type=float, default=0.8)
    learn = sub.add_parser("learn-web")
    learn.add_argument("--model", type=Path, required=True)
    learn.add_argument("--query", required=True)
    learn.add_argument("--db", type=Path, default=Path("arca.db"))
    learn.add_argument("--corpus", type=Path, default=Path("corpus"))
    learn.add_argument("--limit", type=int, default=3)
    learn.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.command == "train":
        corpus = CorpusStore(args.corpus)
        if args.text:
            corpus.add(args.text.read_text(encoding="utf-8"), "file://" + str(args.text), args.text.name)
        report = NativeTrainingPipeline(corpus).train(args.output, ModelConfig(hidden_size=args.hidden, embedding_size=args.embedding), args.epochs)
        print(json.dumps({"status": "trained", **report.__dict__}, indent=2))
    elif args.command == "generate":
        print(ARCALanguageModel.load(args.model).generate(args.prompt, args.tokens, args.temperature))
    else:
        model = ARCALanguageModel.load(args.model)
        corpus = CorpusStore(args.corpus)
        report = WebLearner(model, MemoryStore(args.db), corpus_dir=args.corpus).learn(args.query, args.limit, args.epochs)
        model.save(args.model)
        print(WebLearner.report_json(report))


if __name__ == "__main__":
    main()
