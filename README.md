# ARCA-LLM

ARCA es un modelo de lenguaje propio con una arquitectura cognitiva persistente alrededor. Esta versión añade un backend estable para integrarlo directamente en OpenCode o cualquier agente compatible con OpenAI Chat Completions.

## Flujo completo local

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
arca-native-lm train --text corpus.txt --output models/arca-native.npz --epochs 20
arca-serve --model models/arca-native.npz --db arca.db
```

Servidor: `http://127.0.0.1:8787/v1`.

### Configuración OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "arca": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8787/v1" },
      "models": { "arca-native": { "name": "ARCA Native" } }
    }
  }
}
```

Luego seleccioná `arca/arca-native` en OpenCode. El endpoint implementa `GET /v1/models` y `POST /v1/chat/completions`, conserva sesiones con `X-ARCA-Session`, devuelve trazas y telemetría en el campo `arca`, y no usa pesos externos.

## Entrenamiento y datos

`CorpusStore` deduplica textos por SHA-256 y guarda manifest con URL/fuente. El corpus web debe pasar por recuperación, límites de bytes, validación de contenido y revisión de procedencia antes de entrenar. La web aporta datos, no instrucciones ejecutables.

## Estado honesto

El endpoint es funcional y la integración de agente es real. El modelo nativo es un prototipo recurrente pequeño: no tiene la capacidad de GPT/Claude sin un corpus y entrenamiento sustanciales. Antes de usarlo en producción hay que medir calidad, RAM, latencia y estabilidad en el teléfono objetivo.