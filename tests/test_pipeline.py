import tempfile
import unittest
from pathlib import Path

from arca.native_lm.corpus import CorpusStore
from arca.native_lm.model import ModelConfig
from arca.native_lm.pipeline import NativeTrainingPipeline


class PipelineTests(unittest.TestCase):
    def test_real_seed_corpus_trains_and_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); corpus = CorpusStore(root / "corpus")
            corpus.add(("ARCA es un modelo local entrenable. " * 20), "user:test", "seed")
            report = NativeTrainingPipeline(corpus).train(root / "model.npz", ModelConfig(embedding_size=8, hidden_size=12), epochs=1)
            self.assertTrue(Path(report.model_path).exists())
            self.assertEqual(report.corpus_documents, 1)
            self.assertGreater(report.model_parameters, 0)
            self.assertTrue(Path(report.model_path + ".report.json").exists())


if __name__ == "__main__": unittest.main()
