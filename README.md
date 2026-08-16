# ARCA-LLM

ARCA is a local-first, low-resource **cognitive architecture**, not a disguised compressed LLM. It now runs as a traceable command-line assistant with persistent evidence memory, deterministic reasoning and explicit uncertainty.

## Run it

```bash
git clone https://github.com/enrrutador/ARCA-llm.git
cd ARCA-llm
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
arca chat
```

Try:

```text
arca> remember that ARCA is a verifiable cognitive architecture
arca> what is ARCA?
arca> (25 + 17) * 3
arca> memory
```

One-shot and inspectable execution:

```bash
arca ask "remember that Ada is a mathematician" --db knowledge.db --trace
arca ask "what is Ada?" --db knowledge.db --trace
arca benchmark
python -m unittest discover -s tests -v
```

## Implemented

- M0: budgeted executive, typed blackboard, serializable expediente, Datalog-style inference, A*, safe arithmetic CAS, 100-case benchmark and CI.
- M1: SQLite WAL episodic and semantic memory, provenance, confidence and version-ready assertions.
- M2: bilingual rule-based input compiler, ambiguity detection and deterministic surface responses.
- M3: persistent `(task class, operator)` UCB1 bandit ready for multi-operator routing.

## Honest boundary

This is a functional cognitive assistant, **not yet a general-purpose LLM**. It does not pretend that templates equal language understanding. Open-ended language, safe web evidence acquisition, local neural perception and a surface model remain M4 research. Those features require pinned model artifacts and empirical RAM, latency and quality tests.

## Principles

Separate knowledge from control; enforce active-memory budgets; activate specialists selectively; preserve evidence and derivations; prefer executable verification; version beliefs; learn routing locally; treat external data as hostile; report uncertainty instead of inventing answers; measure accuracy, memory, latency and degradation together.

See [the architecture](docs/ARCHITECTURE.md).
