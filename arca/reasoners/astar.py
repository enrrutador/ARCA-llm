from __future__ import annotations

import heapq

from arca.kernel.budget import Budget
from arca.model import ReasonResult, Task, TraceStep

Point = tuple[int, int]


class AStarReasoner:
    kind = "astar"

    def solve(self, task: Task, budget: Budget) -> ReasonResult:
        width = int(task.payload["width"])
        height = int(task.payload["height"])
        start = tuple(task.payload["start"])
        goal = tuple(task.payload["goal"])
        blocked = {tuple(x) for x in task.payload.get("blocked", [])}
        frontier: list[tuple[int, int, Point]] = [(0, 0, start)]
        came_from: dict[Point, Point | None] = {start: None}
        cost = {start: 0}
        expanded = 0
        while frontier:
            budget.check_time()
            _, _, current = heapq.heappop(frontier)
            expanded += 1
            if current == goal:
                path: list[Point] = []
                node: Point | None = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                trace = [
                    TraceStep("search", f"expanded {expanded} nodes"),
                    TraceStep("path", f"found {len(path) - 1} steps"),
                ]
                return ReasonResult(path, True, trace, {"expanded": expanded, "path_length": len(path) - 1})
            x, y = current
            for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                nx, ny = nxt
                if not (0 <= nx < width and 0 <= ny < height) or nxt in blocked:
                    continue
                new_cost = cost[current] + 1
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    came_from[nxt] = current
                    heuristic = abs(goal[0] - nx) + abs(goal[1] - ny)
                    heapq.heappush(frontier, (new_cost + heuristic, new_cost, nxt))
        return ReasonResult(None, False, [TraceStep("search", "goal unreachable")], {"expanded": expanded})
