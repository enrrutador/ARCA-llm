import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arca.assistant import CognitiveAssistant
from arca.web import WebEvidenceClient
from arca.web.client import WebResult


class FakeWeb:
    def search(self, query, limit=5):
        return [WebResult("ARCA", "https://example.test/arca", "architecture evidence", "ARCA is a cognitive architecture.", "test")]

    def fetch(self, url):
        return WebResult("Example", url, "fetched evidence", "safe document", "test")


class WebTests(unittest.TestCase):
    def test_fetch_rejects_non_http(self):
        with self.assertRaises(ValueError):
            WebEvidenceClient().fetch("file:///etc/passwd")

    def test_search_persists_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            assistant = CognitiveAssistant(Path(folder) / "arca.db", web=FakeWeb())
            response = assistant.ask("buscar ARCA")
            self.assertIn("https://example.test/arca", response["answer"])
            docs = assistant.memory.search_documents("architecture")
            self.assertEqual(len(docs), 1)


if __name__ == "__main__":
    unittest.main()
