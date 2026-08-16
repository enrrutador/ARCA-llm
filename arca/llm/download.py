from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen


# Small instruct model candidate; users can replace this with any compatible GGUF.
DEFAULT_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "ARCA-llm/0.1"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a GGUF model for ARCA")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("models/qwen2.5-0.5b-instruct-q4_k_m.gguf"))
    args = parser.parse_args()
    print(f"Downloading model to {args.output} ...")
    download(args.url, args.output)
    print(f"Ready: {args.output}")


if __name__ == "__main__":
    main()
