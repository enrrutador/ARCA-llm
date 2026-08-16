# ARCA-2 consolidated architecture

## Thesis

ARCA is a verifiable cognitive microkernel. A task is compiled into an explicit cognitive record. A budget-aware executive selects a specialist, records its operations, verifies the result when possible, and preserves evidence. HDC, neural models and web retrieval are optional organs, never universal representations.

## Resident M0

`Executive` owns cadence, routing and budgets. `Blackboard` accepts typed events. `Expediente` stores objective, assertions, constraints, hypotheses, evidence, plan, result, budget and trace in JSON. Only one reasoner runs for each M0 task.

## Cognitive cycle

1. Accept input and create an expediente.
2. Classify the requested operation.
3. Check time and memory budgets.
4. Select the registered specialist for the task kind.
5. Execute and append trace events.
6. Verify success against the specialist contract.
7. Serialize the result and telemetry.
8. Stop, or schedule another bounded cycle in later milestones.

## Representations

Canonical state is typed and symbolic. Assertions support provenance, confidence and timestamps. Hypervectors may later provide compact approximate indexing, but never replace the canonical evidence record.

## Reasoning portfolio

M0 contains forward-chaining Horn rules, A* over finite grids, and a safe AST arithmetic evaluator. Future operators can include Datalog engines, SAT/SMT, HTN planning, graph algorithms, document retrieval and small perception models. Each implements the same `Reasoner` protocol.

## Memory

M0 serializes expedientes and keeps events in memory. Planned stores are episodic append-only logs, versioned semantic assertions, compressed source documents and sandboxed procedures. Contradictions coexist until resolved by evidence; old claims are not silently erased.

## Learning

The first adaptive component will be a contextual bandit keyed by `(task class, operator)`. Reward combines correctness, information gain, latency and memory cost. Learned procedures remain quarantined until tests pass.

## Security

External data is evidence, never executable instruction. Future web and tool modules must isolate downloads, retain original sources, separate quoted instructions from control messages, and execute only allowlisted operators.

## Resource policy

M0 targets 256 MB peak RSS. The executive samples process RSS before and after execution and rejects work after deadline or over-budget observations. Measurements are telemetry, not proof of universal memory behavior: allocator, OS and platform affect RSS.

## Milestones

- M0: explicit state, kernel, three reasoners, 100-case benchmark.
- M1: persistent evidence memory, versioned assertions, safe web acquisition.
- M2: language compiler and ambiguity handling.
- M3: pairwise routing bandit and procedural learning.
- M4: optional local surface model and multimodal specialists.
