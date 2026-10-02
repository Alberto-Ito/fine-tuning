# Resultados de Hugging Face hasta 4.000 iteraciones

El nombre de este archivo se conserva según lo solicitado. En este dataset,
`0.4 epoch` corresponde al checkpoint 1000; la corrida completa de 4000 pasos
equivale a 1,6 épocas.

## Configuración ejecutada

- Modelo: `Qwen/Qwen2.5-0.5B-Instruct`
- Backend: PyTorch 2.8.0 sobre Apple M2/MPS
- Datos: los mismos 10.000 ejemplos de `../data/train.jsonl`, sin modificarlos
- Método: SFT con PEFT/LoRA sobre `q_proj` y `v_proj` de las últimas 4 capas
- LoRA: rank 8, alpha 160 (`alpha/r = 20`), dropout 0
- Learning rate: `1e-5`, constante
- Batch size: 4, sin acumulación
- Longitud máxima: 512 tokens
- Prompt enmascarado: sí; la pérdida sólo cubre la respuesta del asistente
- Checkpoints: cada 250 pasos
- Validación: cada 200 pasos sobre los mismos 15 ejemplos
- Evaluación exacta: los mismos 60 casos usados por MLX
- Seed: 0
- Tiempo de entrenamiento: 2.193,4 s (36m33s)
- Fin: exactamente 4.000 pasos, 16.000 ejemplos vistos, 1,6 épocas

No fue posible usar el mismo Llama 2 13B: los 11 GB ya descargados son pesos
cuantizados en un formato propio de MLX que PyTorch no carga, y los pesos HF
FP16 requieren aproximadamente 26 GB antes de activaciones y optimizador. Eso
supera los 24 GB unificados de esta Mac. Por ello se usó un modelo HF causal
que sí permite completar el proceso real en MPS. Los resultados no son una
comparación directa de calidad entre arquitecturas.

## Evolución de la pérdida

Los valores de train loss son promedios de los 50 pasos anteriores. La
validación se ejecutó cada 200 pasos.

| Iteración | Fracción de época | Train loss | Validation loss |
|---:|---:|---:|---:|
| 200 | 0,08 | 0,0582 | 0,035593 |
| 400 | 0,16 | 0,0272 | 0,025087 |
| 600 | 0,24 | 0,0200 | 0,016906 |
| 800 | 0,32 | 0,0054 | 0,015272 |
| 1.000 | 0,40 | 0,0110 | 0,015816 |
| 1.200 | 0,48 | 0,0022 | 0,018961 |
| 1.400 | 0,56 | 0,0044 | 0,016009 |
| 1.600 | 0,64 | 0,0083 | 0,011148 |
| 1.800 | 0,72 | 0,0084 | 0,029201 |
| 2.000 | 0,80 | 0,0042 | 0,011411 |
| 2.200 | 0,88 | 0,0023 | 0,007811 |
| 2.400 | 0,96 | 0,0096 | 0,015209 |
| 2.600 | 1,04 | 0,0041 | 0,007484 |
| 2.800 | 1,12 | 0,0083 | 0,017759 |
| 3.000 | 1,20 | 0,0157 | 0,007691 |
| 3.200 | 1,28 | 0,0034 | 0,020749 |
| 3.400 | 1,36 | 0,0080 | 0,018896 |
| 3.600 | 1,44 | 0,0012 | 0,007307 |
| **3.800** | **1,52** | **0,0036** | **0,005631** |
| 4.000 | 1,60 | 0,0046 | 0,031852 |

- Mínimo train loss observado: `0,0002` en el bloque terminado en 1900.
- Mínimo validation loss observado: `0,005631` en 3800.
- El valor agregado de train loss de toda la corrida fue `0,017663`.
- La validation loss final fue `0,031852` (perplejidad `1,03237`).

## Precisión exacta por checkpoint

| Checkpoint | Ejemplos vistos | Fracción de época | Precisión | JSON válido |
|---:|---:|---:|---:|---:|
| 250 | 1.000 | 0,10 | 28/60 = 46,7% | 60/60 |
| 500 | 2.000 | 0,20 | 29/60 = 48,3% | 60/60 |
| 750 | 3.000 | 0,30 | 29/60 = 48,3% | 60/60 |
| 1.000 | 4.000 | 0,40 | 30/60 = 50,0% | 60/60 |
| 1.250 | 5.000 | 0,50 | 34/60 = 56,7% | 60/60 |
| 1.500 | 6.000 | 0,60 | 33/60 = 55,0% | 60/60 |
| 1.750 | 7.000 | 0,70 | 33/60 = 55,0% | 60/60 |
| 2.000 | 8.000 | 0,80 | 33/60 = 55,0% | 60/60 |
| 2.250 | 9.000 | 0,90 | 32/60 = 53,3% | 60/60 |
| 2.500 | 10.000 | 1,00 | 33/60 = 55,0% | 60/60 |
| 2.750 | 11.000 | 1,10 | 35/60 = 58,3% | 60/60 |
| 3.000 | 12.000 | 1,20 | 32/60 = 53,3% | 60/60 |
| 3.250 | 13.000 | 1,30 | 35/60 = 58,3% | 60/60 |
| 3.500 | 14.000 | 1,40 | 35/60 = 58,3% | 60/60 |
| 3.750 | 15.000 | 1,50 | 33/60 = 55,0% | 60/60 |
| **4.000** | **16.000** | **1,60** | **36/60 = 60,0%** | **60/60** |

Confusión del checkpoint 4000:

| Esperado | Negativo | Neutro | Positivo |
|---|---:|---:|---:|
| Negativo | 20 | 3 | 17 |
| Neutro | 0 | 9 | 1 |
| Positivo | 1 | 2 | 7 |

## Conclusiones

1. Los 4000 pasos solicitados terminaron correctamente con batch 4 y no se
   ejecutaron pasos adicionales.
2. Menor loss no implicó mayor precisión: 3800 tuvo la menor validation loss,
   pero 4000 tuvo la mejor precisión exacta.
3. El checkpoint recomendado por precisión es `checkpoint-4000` (60,0%). Si el
   criterio exclusivo es validation loss, corresponde usar el estado evaluado
   en 3800; como los checkpoints se guardaron cada 250, el más cercano es 3750.
4. Todos los checkpoints produjeron JSON válido en los 60 casos.
5. El resultado HF queda por debajo del mejor resultado MLX (57/60 en 1000),
   pero se usó un modelo base mucho menor por la restricción de memoria MPS.

Los logs con las 60 predicciones de cada checkpoint están en
`adapters-4000/evaluation-<paso>.log`; la telemetría completa está en
`adapters-4000/metrics.jsonl`.
