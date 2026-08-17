# ARCA-LLM

ARCA es un modelo de lenguaje propio con arquitectura cognitiva persistente alrededor. El PR de integración OpenCode ya está fusionado en `main`; esta versión agrega un pipeline reproducible de corpus real y entrenamiento local.

## Probarlo

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# El repositorio trae un corpus semilla real y versionado para probar el flujo
arca-native-lm train --corpus corpus/seed --output models/arca-native.npz --epochs 20
arca-native-lm generate --model models/arca-native.npz "ARCA es"
arca-serve --model models/arca-native.npz --db arca.db
```

El servidor queda en `http://127.0.0.1:8787/v1`, compatible con OpenCode vía `@ai-sdk/openai-compatible`. Para ampliar capacidad, incorporá documentos aceptados al corpus o ejecutá `arca-native-lm learn-web`, revisá la procedencia y volvé a entrenar.

## Qué es realmente

El modelo es byte-level autoregresivo recurrente, entrenado desde cero con NumPy. No usa Qwen, Llama, Phi, NVIDIA ni pesos externos. La memoria, razonadores, web y trazabilidad son órganos separados de ARCA.

El entrenamiento ahora es reproducible: corpus deduplicado, manifest de procedencia, límite de bytes, informe de pérdida, conteo de parámetros y evaluación básica. El modelo generado funciona como LLM experimental local; la calidad depende directamente del tamaño y calidad del corpus. La web puede aportar datos, pero no convierte automáticamente un modelo pequeño en un modelo de frontera.

## OpenCode

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

Usá `/models` y seleccioná `arca/arca-native`.
