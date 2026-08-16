import tempfile
import unittest
from pathlib import Path

from arca.native_lm import ARCALanguageModel, ModelConfig


class NativeLMTests(unittest.TestCase):
    def test_train_generate_save_load(self):
        model = ARCALanguageModel(ModelConfig(embedding_size=16, hidden_size=24, seed=1))
        losses = model.train_text("ARCA es un modelo local. " * 4, epochs=2, sequence_length=32)
        self.assertEqual(len(losses), 2)
        self.assertLess(losses[-1], losses[0])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "arca.npz"
            model.save(path)
            loaded = ARCALanguageModel.load(path)
            output = loaded.generate("ARCA", max_tokens=8, seed=1)
            self.assertTrue(output.startswith("ARCA"))

    def test_byte_round_trip(self):
        text = "área local"
        self.assertEqual(ARCALanguageModel.decode(ARCALanguageModel.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
