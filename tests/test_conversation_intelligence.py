import tempfile
import unittest
from pathlib import Path

from arca.agent_api import ARCAAgentBackend
from arca.native_lm import ARCALanguageModel, ModelConfig


class ConversationIntelligenceTests(unittest.TestCase):
    def test_ten_upgrades_work_in_a_multi_turn_loop(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = ARCALanguageModel(ModelConfig(embedding_size=8, hidden_size=12, seed=7))
            model.train_text("ARCA is a local cognitive model. " * 5, epochs=1, sequence_length=32)
            model_path = root / "model.npz"
            model.save(model_path)
            backend = ARCAAgentBackend(model_path, root / "memory.db", max_turns=8)
            messages = [
                "remember that the project is ARCA",
                "what is the project?",
                "plan three steps to improve it",
                "hello ARCA",
            ]
            responses = [backend.respond(message, "test-session") for message in messages]
            self.assertEqual(len(responses), 4)
            self.assertTrue(all(response.text for response in responses))
            self.assertTrue(all(response.trace for response in responses))
            self.assertTrue(all(response.telemetry["session_id"] == "test-session" for response in responses))
            self.assertTrue(all("intent" in response.telemetry for response in responses))
            session = Path(str(model_path).replace(".npz", ".sessions.json"))
            self.assertTrue(session.exists())


if __name__ == "__main__":
    unittest.main()
