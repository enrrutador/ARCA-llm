from __future__ import annotations

import ast
import operator

from arca.kernel.budget import Budget
from arca.model import ReasonResult, Task, TraceStep

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate(node: ast.AST, depth: int = 0) -> int | float:
    if depth > 32:
        raise ValueError("expression is too deep")
    if isinstance(node, ast.Expression):
        return evaluate(node.body, depth + 1)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](evaluate(node.operand, depth + 1))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left = evaluate(node.left, depth + 1)
        right = evaluate(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent outside safe range")
        value = _BINARY[type(node.op)](left, right)
        if abs(value) > 10**15:
            raise ValueError("result outside safe range")
        return value
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


class CASReasoner:
    kind = "cas"

    def solve(self, task: Task, budget: Budget) -> ReasonResult:
        budget.check_time()
        expression = str(task.payload["expression"])
        try:
            tree = ast.parse(expression, mode="eval")
            answer = evaluate(tree)
            trace = [
                TraceStep("parse", ast.dump(tree, include_attributes=False)),
                TraceStep("evaluate", f"{expression} = {answer}"),
            ]
            return ReasonResult(answer, True, trace)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
            return ReasonResult(None, False, [TraceStep("reject", str(exc))], {"error": str(exc)})
