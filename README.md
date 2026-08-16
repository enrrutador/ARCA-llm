# ARCA-LLM

Ahora sí: ARCA puede ejecutar un **modelo de lenguaje real local**. La arquitectura cognitiva sigue controlando memoria, herramientas, trazabilidad y razonadores; el GGUF es el órgano generativo para lenguaje abierto.

## Ejecutar el LLM local

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[local-llm]'
arca-download-model
arca chat --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

O una consulta directa:

```bash
arca ask "Explica qué es ARCA" --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --trace
```

El backend usa `llama.cpp`, CPU solamente (`n_gpu_layers=0`), carga GGUF bajo demanda y permite sustituir el modelo por otro compatible. Qwen2.5-0.5B-Instruct Q4 es un punto de partida razonable para el límite de 1 GB, pero el consumo real debe medirse en el teléfono incluyendo runtime, contexto y KV cache.

## Qué es funcional ahora

- Modelo generativo local real, no plantillas.
- Conversación abierta con `chat` y `ask`.
- Memoria SQLite persistente y procedencia.
- Razonadores verificables para aritmética, grafos y reglas.
- Búsqueda y apertura web como evidencia separada de instrucciones.
- Trazas serializables de cada interacción.

## Límite honesto

No afirmamos todavía que Qwen2.5-0.5B tenga capacidad comparable a un LLM grande ni que entre automáticamente en 1 GB en cualquier móvil. Hay que medir RSS pico, KV cache, latencia y calidad con el dispositivo objetivo.

Consulta [Qwen2.5](https://doi.org/10.48550/arxiv.2412.15115) y [llama.cpp](https://github.com/ggerganov/llama.cpp).