import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from arca.agent_api import ARCAAgentBackend
from arca.native_lm import ARCALanguageModel, ModelConfig
from arca.server import ARCAHTTPHandler


class ServerTests(unittest.TestCase):
    def test_openai_compatible_endpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = ARCALanguageModel(ModelConfig(embedding_size=8, hidden_size=12, seed=2))
            model.train_text("ARCA local language model. " * 4, epochs=1, sequence_length=32)
            model.save(root / "model.npz")
            server = ThreadingHTTPServer(("127.0.0.1", 0), ARCAHTTPHandler)
            server.backend = ARCAAgentBackend(root / "model.npz", root / "memory.db")
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urllib.request.urlopen(base + "/v1/models") as response:
                    self.assertEqual(json.load(response)["data"][0]["id"], "arca-native")
                request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps({"model": "arca-native", "messages": [{"role": "user", "content": "hello"}]}).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request) as response:
                    payload = json.load(response)
                self.assertEqual(payload["object"], "chat.completion")
                self.assertTrue(payload["choices"][0]["message"]["content"])
                self.assertIn("trace", payload["arca"])
            finally:
                server.shutdown(); thread.join(timeout=2); server.server_close()


if __name__ == "__main__": unittest.main()
