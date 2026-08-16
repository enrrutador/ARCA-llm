# ARCA-LLM

ARCA is an experimental, local-first cognitive architecture designed to maximize useful reasoning per active byte. It is not a compressed LLM: knowledge, control, reasoning, memory and language are separate components.

## M0 status

M0 is runnable today. It includes a budgeted executive, serializable cognitive records, a typed blackboard, three selectively activated reasoners (Datalog-style forward chaining, A*, and a safe arithmetic CAS), complete traces, and a deterministic 100-case benchmark.

```bash
git clone https://github.com/enrrutador/ARCA-llm.git
cd ARCA-llm
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
arca demo
arca-m0 --json benchmark-results.json
python -m unittest discover -s tests -v
```

No runtime dependencies are required. `uv sync` can be used instead of pip.

## Twelve principles

1. Separate cognitive capacity from stored knowledge.
2. Make active memory a hard budget, not a suggestion.
3. Keep the resident kernel small and replaceable.
4. Activate one expensive specialist at a time.
5. Represent goals, evidence, constraints and uncertainty explicitly.
6. Preserve provenance and derivations for every conclusion.
7. Prefer verifiable operators over fluent guessing.
8. Treat retrieval as evidence access, not truth.
9. Learn incrementally without rewriting the whole core.
10. Version beliefs instead of silently overwriting them.
11. Treat external content and tools as hostile by default.
12. Measure accuracy, peak RAM, latency and degradation together.

## Three refinements

- **Cadence:** cognition advances in bounded cycles with explicit stopping conditions.
- **Serializable expediente:** every task can be persisted, inspected and resumed.
- **Bandit by pair:** future routing learns utility for `(task class, operator)` pairs, rather than a global opaque policy.

## M0 targets

The benchmark contains exactly 30 logic, 35 pathfinding and 35 arithmetic exercises. It reports accuracy, median/p95 latency, Python allocation peak, process RSS peak, trace coverage and first-vs-last-20 degradation. The 256 MB ceiling is enforced by the executive when platform RSS information is available.

Real 0.5B neural and hybrid baselines are intentionally not faked. Their adapters belong in the next benchmark milestone because model weights and inference engines must be pinned for reproducible comparisons.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the living architecture.
