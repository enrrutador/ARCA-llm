from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from arca.native_lm.corpus import CorpusStore
from arca.native_lm.model import ARCALanguageModel, ModelConfig


@dataclass(slots=True)
class TrainingReport:
    model_path: str
    corpus_bytes: int
    corpus_documents: int
    epochs: int
    losses: list[float]
    elapsed_seconds: float
    model_parameters: int


class NativeTrainingPipeline:
    """Reproducible local training pipeline for the native ARCA model."""

    def __init__(self, corpus: CorpusStore) -> None:
        self.corpus = corpus

    def train(self, output: str | Path, config: ModelConfig | None = None, epochs: int = 10, max_bytes: int = 10_000_000) -> TrainingReport:
        text = self.corpus.read_all(max_bytes=max_bytes)
        if len(text.encode("utf-8")) < 256:
            raise ValueError("corpus is too small; add real documents before training")
        model = ARCALanguageModel(config or ModelConfig())
        started = time.perf_counter()
        losses = model.train_text(text, epochs=epochs)
        model.save(output)
        docs = len(list(self.corpus.root.glob("*.txt")))
        report = TrainingReport(str(output), len(text.encode("utf-8")), docs, epochs, losses, time.perf_counter() - started, self.parameter_count(model))
        Path(str(output) + ".report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        return report

    @staticmethod
    def parameter_count(model: ARCALanguageModel) -> int:
        return sum(int(value.size) for value in (model.embedding, model.w_ih, model.w_hh, model.b_h, model.w_ho, model.b_o))

    @staticmethod
    def evaluate(model: ARCALanguageModel, prompts: list[str]) -> dict[str, Any]:
        outputs = [model.generate(prompt, max_tokens=96, temperature=0.2, seed=0) for prompt in prompts]
        return {"prompts": len(prompts), "nonempty": sum(bool(output[len(prompt):].strip()) for prompt, output in zip(prompts, outputs)), "outputs": outputs}
