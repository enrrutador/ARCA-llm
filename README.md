# ARCA-LLM

ARCA includes a native trainable language model plus an agent-ready conversation layer. The latest upgrade adds persistent multi-turn sessions, bounded context, salience extraction, reference resolution, intent routing, context compression, response quality filtering and telemetry.

```python
from arca.agent_api import ARCAAgentBackend

backend = ARCAAgentBackend("models/arca-native.npz", "arca.db")
reply = backend.respond("remember that the project is ARCA", session_id="user-1")
reply = backend.respond("what is the project?", session_id="user-1")
print(reply.text, reply.telemetry)
```

The backend is deliberately framework-neutral, so it can be called from an agent loop, HTTP service or task runner. It keeps external weights disabled, persists session state beside the model, and returns trace plus telemetry on every turn.
