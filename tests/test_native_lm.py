import tempfile
import unittest
from pathlib import Path

from arca.native_lm import ARCALanguageModel, ModelConfig
from arca.native_lm.learner import WebLearner
from arca.web.client import WebResult
from arca.memory import MemoryStore


class FakeWeb:
    def search(self, query, limit=3):
        return [WebResult("ARCA", f"https://example.test/{i}", "evidence", "ARCA is a local cognitive architecture. " * 20, "test") for i in range(limit)]


class NativeLMTests(unittest.TestCase):
    def test_train_generate_save_load(self):
        model = ARCALanguageModel(ModelConfig(embedding_size=16, hidden_size=24, seed=1))
        losses = model.train_text("ARCA es un modelo local. " * 4, epochs=2, sequence_length=32)
        self.assertEqual(len(losses), 2)
        self.assertLessEqual(losses[-1], losses[0])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "arca.npz"
            model.save(path)
            loaded = ARCALanguageModel.load(path)
            output = loaded.generate("ARCA", max_tokens=8, seed=1)
            self.assertTrue(output.startswith("ARCA"))

    def test_byte_round_trip(self):
        text = "área local"
        self.assertEqual(ARCALanguageModel.decode(ARCALanguageModel.encode(text)), text)

    def test_web_learning_is_bounded_and_persistent(self):
        with tempfile.TemporaryDirectory() as folder:
            model = ARCALanguageModel(ModelConfig(embedding_size=8, hidden_size=12, seed=1))
            store = MemoryStore(Path(folder) / "arca.db")
            report = WebLearner(model, store, FakeWeb(), Path(folder) / "corpus").learn("ARCA", limit=2, epochs=1, max_bytes=100_000)
            self.assertEqual(report.documents, 2)
            self.assertEqual(len(store.search_documents("architecture")), 2)
            self.assertTrue(all(Path(folder).joinpath("corpus").glob("*.txt")))


if __name__ == "__main__":
    unittest.main()
