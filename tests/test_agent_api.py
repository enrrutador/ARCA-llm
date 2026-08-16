import tempfile
import unittest
from pathlib import Path

from arca.agent_api import ARCAAgentBackend
from arca.native_lm import ARCALanguageModel, ModelConfig


class AgentAPITests(unittest.TestCase):
    def test_backend_responds_with_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "model.npz"
            model = ARCALanguageModel(ModelConfig(embedding_size=8, hidden_size=12))
            model.train_text("ARCA local model. " * 3, epochs=1)
            model.save(model_path)
            result = ARCAAgentBackend(model_path, Path(folder) / "memory.db").respond("write a short greeting")
            self.assertTrue(result.success)
            self.assertTrue(result.text)
            self.assertTrue(result.trace)


if __name__ == "__main__": unittest.main()
