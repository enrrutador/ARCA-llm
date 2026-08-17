import tempfile
import unittest
from pathlib import Path

from arca.agent_api import ARCAAgentBackend
from arca.native_lm import ARCALanguageModel, ModelConfig


class AgentLoopTests(unittest.TestCase):
    def test_simulated_agent_loop_persists_context_and_traces_each_turn(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = ARCALanguageModel(ModelConfig(embedding_size=8, hidden_size=12, seed=7))
            model.train_text("ARCA is a local cognitive model. " * 4, epochs=1, sequence_length=32)
            model_path = root / "model.npz"
            model.save(model_path)
            backend = ARCAAgentBackend(model_path, root / "memory.db")

            messages = [
                "remember that the project is ARCA",
                "what is the project?",
                "write a short greeting",
            ]
            responses = [backend.respond(message) for message in messages]

            self.assertEqual(len(responses), 3)
            self.assertTrue(all(response.text for response in responses))
            self.assertTrue(all(response.trace for response in responses))
            self.assertTrue(all("success" in response.telemetry for response in responses))
            self.assertIn("arca", responses[1].text.casefold())
            self.assertTrue(responses[2].telemetry.get("model") == "ARCA-native-recurrent")


if __name__ == "__main__":
    unittest.main()
