# ARCA-LLM

ARCA es un **modelo de lenguaje propio** y su capacidad se adquiere mediante un ciclo de aprendizaje local controlado: consulta la web, recupera evidencia, la guarda con procedencia, la incorpora a un corpus local y actualiza sus propios pesos con entrenamiento incremental. No utiliza Qwen, Llama, Phi ni pesos externos.

## Entrenar, generar y aprender de la web

```bash
pip install -e .
arca-native-lm train --text corpus.txt --output models/arca-native.npz --epochs 20
arca-native-lm generate --model models/arca-native.npz "ARCA:"
arca-native-lm learn-web --model models/arca-native.npz --query "computación cognitiva eficiente" --epochs 1
arca chat --model models/arca-native.npz
```

El aprendizaje web es deliberadamente acotado: solo acepta HTTP(S), ignora binarios, limita bytes, deduplica contenido por SHA-256, conserva URL/fuente/fecha, no ejecuta instrucciones recuperadas y actualiza el modelo en ciclos pequeños. La web aporta conocimiento y datos de entrenamiento; no sustituye al modelo.

## Arquitectura

El modelo es byte-level autoregresivo recurrente, entrenable desde cero con NumPy. Su estado recurrente es memoria de trabajo activa; memoria semántica, episódica y documental viven fuera de los pesos. El kernel decide cuándo recuperar, verificar, guardar y consolidar.

## Capacidad real

Funciona como LLM nativo experimental, pero la capacidad no aparece mágicamente: necesita corpus inicial, tiempo de CPU y ciclos de aprendizaje. La web puede ampliar conocimiento, pero no convierte automáticamente un modelo pequeño en uno frontier; medimos pérdida, RAM, latencia, procedencia y degradación.