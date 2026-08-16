# ARCA-LLM

Ahora ARCA es un **modelo de lenguaje propio**, no un LLM externo conectado. El núcleo generativo es un modelo autoregresivo recurrente byte-level, entrenable desde cero con NumPy, sin Transformer obligatorio, sin pesos de Qwen/Llama/Phi y sin `llama.cpp`.

## Entrenar y ejecutar

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# corpus propio, UTF-8
arca-native-lm train --text corpus.txt --output models/arca-native.npz --epochs 20
arca-native-lm generate --model models/arca-native.npz "ARCA:"
arca chat --model models/arca-native.npz
```

También funciona sin un modelo entrenado: memoria, web, cálculo, Datalog y A* siguen disponibles.

## Qué se corrigió

- **No se añadió un LLM externo.** Qwen/llama.cpp fue descartado y el PR correspondiente se cerró.
- ARCA tiene su propio modelo autoregresivo entrenable desde cero.
- La representación es byte-level y el estado recurrente funciona como memoria de trabajo activa.
- El modelo se puede guardar/cargar como pesos `.npz` y el asistente lo usa para lenguaje abierto.
- Memoria persistente, procedencia, herramientas y razonadores siguen fuera de los pesos.

## Límite técnico real

Esto es funcional como modelo de lenguaje experimental, pero un modelo pequeño entrenado con un corpus pequeño no posee la capacidad de un LLM grande. Para obtener competencia amplia hacen falta corpus, entrenamiento, evaluación y mucho tiempo de CPU. La especificación de 1 GB de RAM activa es un objetivo medible, no una garantía automática.

La arquitectura cumple la decisión importante: **ARCA es el modelo**, no un orquestador que envuelve un modelo ajeno.