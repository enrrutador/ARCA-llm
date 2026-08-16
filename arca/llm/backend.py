from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LLMConfig:
    model_path: str | Path
    context_size: int = 2048
    threads: int = 4
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


class LocalLLM:
    """Real local GGUF inference through llama.cpp, loaded only when requested.

    The cognitive kernel remains independent of the model. The model is the
    language organ: it interprets open-ended text and verbalizes verified state.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        path = Path(config.model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {path}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("Install the optional local extra: pip install -e '.[local-llm]'") from exc
        self._model = Llama(
            model_path=str(path),
            n_ctx=config.context_size,
            n_threads=config.threads,
            n_gpu_layers=0,
            verbose=False,
        )

    def complete(self, prompt: str, system: str = "") -> dict[str, Any]:
        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": system or "You are ARCA, a concise local assistant. Do not invent facts."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        choice = response["choices"][0]
        return {
            "text": choice["message"]["content"],
            "usage": response.get("usage", {}),
            "model": str(self.config.model_path),
        }
