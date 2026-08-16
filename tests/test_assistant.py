import tempfile
import unittest
from pathlib import Path

from arca.assistant import CognitiveAssistant
from arca.language import compile_text


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.assistant = CognitiveAssistant(Path(self.temp.name) / "test.db")

    def tearDown(self):
        self.temp.cleanup()

    def test_memory_persists_and_recalls_provenance(self):
        saved = self.assistant.ask("remember that ARCA is a cognitive architecture")
        self.assertTrue(saved["expediente"]["telemetry"]["success"])
        recalled = self.assistant.ask("what is ARCA?")
        self.assertIn("cognitive architecture", recalled["answer"])
        self.assertIn("source: user", recalled["answer"])

    def test_arithmetic_language_route(self):
        response = self.assistant.ask("(12 + 3) * 2")
        self.assertIn("30", response["answer"])
        self.assertTrue(response["expediente"]["telemetry"]["success"])

    def test_unknown_request_admits_missing_evidence(self):
        response = self.assistant.ask("Explain a topic absent from memory")
        self.assertIn("don't have reliable", response["answer"])

    def test_compiler_detects_ambiguity(self):
        intent = compile_text("tell me something vague")
        self.assertLess(intent.confidence, 1.0)
        self.assertIsNotNone(intent.ambiguity)


if __name__ == "__main__":
    unittest.main()
