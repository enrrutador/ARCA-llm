from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 256
    embedding_size: int = 64
    hidden_size: int = 192
    seed: int = 42


class ARCALanguageModel:
    """Native ARCA byte-level recurrent LM.

    This is an actual autoregressive language model trained from scratch. It is
    deliberately not a Transformer and has no dependency on external weights.
    The recurrent state is the active working memory; long-term knowledge stays
    in ARCA's explicit memory stores.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        rng = np.random.default_rng(self.config.seed)
        v, e, h = self.config.vocab_size, self.config.embedding_size, self.config.hidden_size
        scale = 1.0 / np.sqrt(h)
        self.embedding = rng.normal(0, scale, (v, e)).astype(np.float32)
        self.w_ih = rng.normal(0, scale, (e, h)).astype(np.float32)
        self.w_hh = rng.normal(0, scale, (h, h)).astype(np.float32)
        self.b_h = np.zeros(h, dtype=np.float32)
        self.w_ho = rng.normal(0, scale, (h, v)).astype(np.float32)
        self.b_o = np.zeros(v, dtype=np.float32)

    @staticmethod
    def encode(text: str) -> np.ndarray:
        return np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8).astype(np.int64)

    @staticmethod
    def decode(tokens: Iterable[int]) -> str:
        return bytes(int(x) & 0xFF for x in tokens).decode("utf-8", errors="replace")

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -20, 20))

    def step(self, token: int, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        state = np.zeros(self.config.hidden_size, dtype=np.float32) if state is None else state
        hidden = self._tanh(self.embedding[token] @ self.w_ih + state @ self.w_hh + self.b_h)
        logits = hidden @ self.w_ho + self.b_o
        logits -= np.max(logits)
        probabilities = np.exp(logits).astype(np.float64)
        probabilities /= probabilities.sum()
        return hidden, probabilities.astype(np.float32)

    def train_text(self, text: str, epochs: int = 3, learning_rate: float = 0.03, sequence_length: int = 96) -> list[float]:
        tokens = self.encode(text)
        if len(tokens) < 2:
            raise ValueError("training text must contain at least two UTF-8 bytes")
        losses: list[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            count = 0
            for start in range(0, len(tokens) - 1, sequence_length):
                window = tokens[start : start + sequence_length + 1]
                state = np.zeros(self.config.hidden_size, dtype=np.float32)
                states: list[np.ndarray] = [state.copy()]
                hiddens: list[np.ndarray] = []
                probs: list[np.ndarray] = []
                for index in range(len(window) - 1):
                    state, probability = self.step(int(window[index]), state)
                    hiddens.append(state.copy())
                    probs.append(probability)
                    target = int(window[index + 1])
                    epoch_loss -= float(np.log(max(float(probability[target]), 1e-12)))
                    count += 1
                    states.append(state.copy())
                self._update(window[:-1], window[1:], states, hiddens, probs, learning_rate)
            losses.append(epoch_loss / max(count, 1))
        return losses

    def _update(self, inputs, targets, states, hiddens, probs, learning_rate: float) -> None:
        e, h, v = self.config.embedding_size, self.config.hidden_size, self.config.vocab_size
        grad_e = np.zeros_like(self.embedding)
        grad_w_ih = np.zeros_like(self.w_ih)
        grad_w_hh = np.zeros_like(self.w_hh)
        grad_b_h = np.zeros_like(self.b_h)
        grad_w_ho = np.zeros_like(self.w_ho)
        grad_b_o = np.zeros_like(self.b_o)
        dh_next = np.zeros(h, dtype=np.float32)
        for index in range(len(hiddens) - 1, -1, -1):
            probability = probs[index].copy()
            probability[int(targets[index])] -= 1.0
            grad_w_ho += np.outer(hiddens[index], probability)
            grad_b_o += probability
            dh = probability @ self.w_ho.T + dh_next
            dz = dh * (1.0 - hiddens[index] ** 2)
            grad_b_h += dz
            grad_w_ih += np.outer(self.embedding[int(inputs[index])], dz)
            grad_w_hh += np.outer(states[index], dz)
            grad_e[int(inputs[index])] += dz @ self.w_ih.T
            dh_next = dz @ self.w_hh.T
        gradients = [grad_e, grad_w_ih, grad_w_hh, grad_b_h, grad_w_ho, grad_b_o]
        for gradient in gradients:
            np.clip(gradient, -1.0, 1.0, out=gradient)
        self.embedding -= learning_rate * grad_e
        self.w_ih -= learning_rate * grad_w_ih
        self.w_hh -= learning_rate * grad_w_hh
        self.b_h -= learning_rate * grad_b_h
        self.w_ho -= learning_rate * grad_w_ho
        self.b_o -= learning_rate * grad_b_o

    def generate(self, prompt: str, max_tokens: int = 160, temperature: float = 0.8, seed: int | None = None) -> str:
        tokens = self.encode(prompt).tolist()
        state = np.zeros(self.config.hidden_size, dtype=np.float32)
        for token in tokens:
            state, _ = self.step(token, state)
        rng = np.random.default_rng(seed)
        generated: list[int] = []
        for _ in range(max_tokens):
            state, probabilities = self.step(tokens[-1] if tokens else 32, state)
            logits = np.log(np.maximum(probabilities, 1e-12)) / max(temperature, 0.05)
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            token = int(rng.choice(self.config.vocab_size, p=probabilities))
            tokens.append(token)
            generated.append(token)
        return prompt + self.decode(generated)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, embedding=self.embedding, w_ih=self.w_ih, w_hh=self.w_hh, b_h=self.b_h, w_ho=self.w_ho, b_o=self.b_o, config=json.dumps(asdict(self.config)))

    @classmethod
    def load(cls, path: str | Path) -> "ARCALanguageModel":
        with np.load(path, allow_pickle=False) as data:
            config = ModelConfig(**json.loads(str(data["config"])))
            model = cls(config)
            for name in ("embedding", "w_ih", "w_hh", "b_h", "w_ho", "b_o"):
                setattr(model, name, data[name].astype(np.float32))
            return model
