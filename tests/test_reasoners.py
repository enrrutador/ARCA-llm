import unittest

from arca.app import build_executive
from arca.model import Task


class ReasonerTests(unittest.TestCase):
    def setUp(self):
        self.executive = build_executive()

    def test_cas(self):
        record = self.executive.execute(Task("cas", {"expression": "(2 + 3) * 4"}))
        self.assertEqual(record.result, 20)
        self.assertTrue(record.telemetry["success"])

    def test_astar(self):
        task = Task("astar", {"width": 3, "height": 3, "start": (0, 0), "goal": (2, 2), "blocked": [(1, 1)]})
        record = self.executive.execute(task)
        self.assertEqual(len(record.result) - 1, 4)

    def test_datalog_transitive_rule(self):
        task = Task("datalog", {
            "facts": [["parent", "a", "b"], ["parent", "b", "c"]],
            "rules": [
                {"premises": [["parent", "?x", "?y"]], "conclusion": ["ancestor", "?x", "?y"]},
                {"premises": [["parent", "?x", "?y"], ["ancestor", "?y", "?z"]], "conclusion": ["ancestor", "?x", "?z"]},
            ],
            "query": ["ancestor", "a", "c"],
        })
        record = self.executive.execute(task)
        self.assertIs(record.result, True)
        self.assertGreater(len(record.trace), 1)

    def test_cas_rejects_code(self):
        record = self.executive.execute(Task("cas", {"expression": "__import__('os').system('echo nope')"}))
        self.assertFalse(record.telemetry["success"])


if __name__ == "__main__":
    unittest.main()
