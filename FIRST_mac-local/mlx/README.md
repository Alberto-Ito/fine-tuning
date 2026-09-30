# Fine-tuning local con MLX en Mac M2 de 24 GB

Estas instrucciones son específicas para ejecutar fine-tuning con **MLX en macOS sobre Apple Silicon**. La prueba reproduce la tarea de clasificación de sentimiento y usa los mismos 10 000 ejemplos. El entrenamiento local es **QLoRA/SFT**: la etiqueta `ground_truth` de Fireworks pasa a ser la respuesta del asistente. No reproduce el algoritmo RFT ni ejecuta el evaluator de Fireworks. No requiere un trabajo de entrenamiento ni una API de pago.

Desde Terminal, coloca este proyecto en la Mac y entra en `fine-tuning/mac-local/mlx`. Necesitas Apple Silicon, Python 3.9 o posterior y espacio libre para el modelo cuantizado (el repositorio indica aproximadamente 11,5 GB), cachés y adaptadores. Recomiendo comprobar al menos 25 GB libres antes de descargarlo. Cierra aplicaciones que consuman mucha memoria durante la prueba. Los datasets compartidos permanecen en `../data`.

```bash
uname -m                    # debe mostrar arm64
df -h .                    # espacio libre
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'mlx-lm[train]'
# Los archivos de data/ ya están preparados. Ejecuta este paso solamente si
# también tienes los JSONL fuente rft-*.jsonl en `fine-tuning/`:
python prepare_mlx_data.py
```

El modelo de MLX usado aquí es [`mlx-community/llama2-13b-qnt4bit`](https://huggingface.co/mlx-community/llama2-13b-qnt4bit), cuya ficha indica que fue convertido desde `meta-llama/Llama-2-13b-chat-hf`. Confirma que puedes usarlo conforme a la [licencia de Llama 2](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) antes de descargarlo. Esta copia ya está cuantizada a 4 bits, por lo que evita convertir los pesos originales de aproximadamente 26 GB dentro de los 24 GB de memoria de la Mac.

Primero ejecuta una prueba pequeña. El primer comando descargará el modelo y puede tardar bastante, según tu conexión. `--iters 30` significa 30 pasos de entrenamiento, no 30 ejemplos ni una época completa.

```bash
mlx_lm.lora \
  --model mlx-community/llama2-13b-qnt4bit \
  --train --data ../data --iters 30 \
  --batch-size 1 --num-layers 4 \
  --mask-prompt --grad-checkpoint \
  --max-seq-length 512 \
  --adapter-path ./adapters --save-every 10
```

Si termina sin falta de memoria, amplía la prueba gradualmente, por ejemplo a 300 pasos. Para continuar desde un adaptador existente, consulta `mlx_lm.lora --help` y usa `--resume-adapter-file` con el archivo guardado en `adapters/`. No se puede garantizar que 13B entrene en todas las configuraciones de 24 GB; si falla, reduce `--max-seq-length` a 384 y prueba `--num-layers 2`. El entrenamiento puede ser lento en un M2.

```bash
mlx_lm.lora \
  --model mlx-community/llama2-13b-qnt4bit \
  --adapter-path ./adapters --data ../data --test
```

`--test` mide perplejidad sobre `test.jsonl`, no precisión de clasificación. Para medir la precisión exacta y el cumplimiento del formato JSON con el mismo esquema de respuesta que evalúa Fireworks, ejecuta estas pruebas sobre los 60 casos reservados:

```bash
python evaluate_sentiment.py --baseline --limit 5  # prueba corta del modelo base
python evaluate_sentiment.py --limit 5             # prueba corta del adaptador
python evaluate_sentiment.py --baseline           # precisión del modelo base
python evaluate_sentiment.py                      # precisión del adaptador
```

La prueba supervisada depende de la calidad de los ejemplos. Muchos de los 10 000 registros son variaciones sintéticas de pocas frases originales; una perplejidad menor o una buena respuesta en ejemplos parecidos no demostraría buena generalización. Para evaluar el resultado, conviene ampliar las pruebas con mensajes reales e independientes que no estén en el entrenamiento.
