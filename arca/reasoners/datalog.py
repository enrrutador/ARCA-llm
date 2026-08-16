from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from arca.kernel.budget import Budget
from arca.model import ReasonResult, Task, TraceStep

Atom = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Rule:
    premises: tuple[Atom, ...]
    conclusion: Atom


def is_var(value: str) -> bool:
    return value.startswith("?")


def unify(pattern: Atom, fact: Atom, env: dict[str, str]) -> dict[str, str] | None:
    if len(pattern) != len(fact) or pattern[0] != fact[0]:
        return None
    out = dict(env)
    for expected, actual in zip(pattern[1:], fact[1:]):
        if is_var(expected):
            bound = out.get(expected)
            if bound is not None and bound != actual:
                return None
            out[expected] = actual
        elif expected != actual:
            return None
    return out


def substitute(atom: Atom, env: dict[str, str]) -> Atom:
    return tuple(env.get(part, part) for part in atom)


class DatalogReasoner:
    kind = "datalog"

    def solve(self, task: Task, budget: Budget) -> ReasonResult:
        facts = {tuple(x) for x in task.payload["facts"]}
        rules = [Rule(tuple(tuple(p) for p in r["premises"]), tuple(r["conclusion"])) for r in task.payload["rules"]]
        query = tuple(task.payload["query"])
        trace = [TraceStep("load_facts", f"loaded {len(facts)} facts")]
        derived = 0
        changed = True
        while changed:
            budget.check_time()
            changed = False
            snapshot = tuple(facts)
            for rule in rules:
                environments = [dict()]
                for premise in rule.premises:
                    next_envs: list[dict[str, str]] = []
                    for env, fact in product(environments, snapshot):
                        match = unify(premise, fact, env)
                        if match is not None:
                            next_envs.append(match)
                    environments = next_envs
                    if not environments:
                        break
                for env in environments:
                    conclusion = substitute(rule.conclusion, env)
                    if conclusion not in facts and not any(is_var(x) for x in conclusion):
                        facts.add(conclusion)
                        derived += 1
                        changed = True
                        trace.append(TraceStep("derive", "(" + ", ".join(conclusion) + ")"))
            if derived > 10_000:
                return ReasonResult(False, False, trace, {"error": "derivation limit"})
        answer = query in facts
        trace.append(TraceStep("query", f"{query!r} -> {answer}"))
        return ReasonResult(answer, True, trace, {"facts_derived": derived})
