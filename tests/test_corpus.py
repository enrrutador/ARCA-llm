import tempfile
import unittest
from pathlib import Path

from arca.native_lm.corpus import CorpusStore


class CorpusTests(unittest.TestCase):
    def test_deduplication_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            store = CorpusStore(Path(folder))
            self.assertTrue(store.add("A" * 100, "https://example.test/a", "A"))
            self.assertFalse(store.add("A" * 100, "https://example.test/b", "B"))
            self.assertIn("https://example.test/a", (Path(folder) / "manifest.jsonl").read_text())
            self.assertEqual(len(store.read_all()), 100)


if __name__ == "__main__": unittest.main()
