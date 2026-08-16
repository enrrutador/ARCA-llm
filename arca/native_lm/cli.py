from __future__ import annotations

import argparse
from pathlib import Path

from arca.native_lm.model import ARCALanguageModel, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run the native ARCA language model")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--text", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("models/arca-native.npz"))
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--hidden", type=int, default=192)
    train.add_argument("--embedding", type=int, default=64)
    run = sub.add_parser("generate")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("prompt")
    run.add_argument("--tokens", type=int, default=160)
    run.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    if args.command == "train":
        model = ARCALanguageModel(ModelConfig(hidden_size=args.hidden, embedding_size=args.embedding))
        text = args.text.read_text(encoding="utf-8")
        losses = model.train_text(text, epochs=args.epochs)
        model.save(args.output)
        print(f"saved {args.output}; losses={', '.join(f'{x:.4f}' for x in losses)}")
    else:
        model = ARCALanguageModel.load(args.model)
        print(model.generate(args.prompt, args.tokens, args.temperature))


if __name__ == "__main__":
    main()
