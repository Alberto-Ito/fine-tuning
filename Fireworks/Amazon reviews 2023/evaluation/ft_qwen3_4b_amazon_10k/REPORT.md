# Evaluación A/B del SFT `ft-qwen3-4b-amazon-10k`

Fecha: 2026-10-15

## Veredicto

El SFT produjo una mejora grande y consistente frente al Qwen3-4B base en este benchmark. Superó al base en los 200 casos tanto en Token F1 como en ROUGE-L, respondió de forma mucho más concisa y evitó truncamientos. El fine-tuning sí funcionó para aprender las cuatro tareas y el estilo del dataset.

El resultado se considera **aprobado para una siguiente etapa controlada**, con una limitación pendiente: el FT todavía sobrestima la recurrencia de problemas en algunos casos. No se recomienda tomar esta evaluación automática como aprobación final para producción sin una revisión humana ciega de groundedness y cumplimiento de reglas.

## Diseño del test

- 200 conversaciones, 50 por tarea.
- 200 productos (`parent_asin`) únicos.
- Exclusión estricta de los 2.337 productos presentes en los candidatos del training.
- Respuestas gold provenientes de ejemplos aprobados del corpus completo.
- Mismos prompts, dataset, límite de 500 tokens y `reasoning_effort="none"` para ambos modelos.
- Tareas: `pros_and_cons`, `use_cases`, `common_problems` y `purchase_recommendation`.

Se utilizó non-thinking mode porque el Qwen3-4B base consumía el límite de salida en razonamiento interno y no producía una respuesta visible comparable. Fireworks documenta `reasoning_effort="none"` como el modo no-thinking de Qwen3.

## Resultado global

| Métrica | Qwen3-4B base | FT Amazon 10k | Cambio FT |
|---|---:|---:|---:|
| Requests exitosos | 200/200 | 200/200 | — |
| Token F1 vs. gold | 0,2985 | **0,5272** | **+0,2287** |
| ROUGE-L F1 vs. gold | 0,1899 | **0,3964** | **+0,2065** |
| Longitud media | 215,6 palabras | **70,8 palabras** | −67,2% |
| Respuestas truncadas | 6 | **0** | −6 |
| Tokens de salida | 57.978 | **17.670** | −69,5% |
| Latencia media | 1,539 s | **1,316 s** | −14,5% |
| Latencia p50 | 1,520 s | **1,010 s** | −33,6% |
| Latencia p95 | **2,064 s** | 2,733 s | +32,4% |

El FT ganó los 200/200 pares en Token F1 y los 200/200 pares en ROUGE-L. La mejora no depende de unos pocos outliers.

## Resultado por tarea

### Token F1

| Tarea | Base | FT | Cambio |
|---|---:|---:|---:|
| `pros_and_cons` | 0,3981 | **0,5805** | +0,1824 |
| `use_cases` | 0,3331 | **0,5500** | +0,2169 |
| `common_problems` | 0,2313 | **0,4857** | +0,2544 |
| `purchase_recommendation` | 0,2316 | **0,4926** | +0,2610 |

### ROUGE-L F1

| Tarea | Base | FT | Cambio |
|---|---:|---:|---:|
| `pros_and_cons` | 0,2536 | **0,4412** | +0,1876 |
| `use_cases` | 0,2255 | **0,4252** | +0,1997 |
| `common_problems` | 0,1457 | **0,3616** | +0,2159 |
| `purchase_recommendation` | 0,1349 | **0,3574** | +0,2225 |

La mayor ganancia aparece en `purchase_recommendation` y `common_problems`, precisamente las tareas que requieren una estructura de salida más específica.

## Hallazgo de calidad pendiente

Entre los 50 casos de `common_problems`, 13 gold indicaban explícitamente que no había problemas recurrentes. En 5 de esos 13 casos (38,5%), el FT presentó una observación aislada o una característica como problema recurrente. Esto contradice la regla R003/R008 de no elevar un reporte aislado a defecto recurrente.

El detector textual no marcó ese patrón en el base, pero eso no implica que el base sea globalmente mejor calibrado: sus respuestas fueron mucho más largas y necesitan evaluación semántica humana. Este slice debe considerarse una prueba de regresión obligatoria para la próxima versión del SFT.

## Interpretación

- La mejora del FT es inequívoca en ajuste al formato gold, concisión y eficiencia de salida.
- La reducción de 40.308 tokens de salida en 200 casos es material para costo y latencia.
- El p95 del FT fue peor aunque su media y mediana fueron mejores; conviene medir nuevamente con más repeticiones antes de sacar conclusiones de infraestructura.
- La métrica exploratoria de tokens no soportados fue 0,4770 para el base y 0,5165 para el FT. No se usa como criterio de aceptación porque penaliza paráfrasis y vocabulario funcional; sí refuerza la necesidad de una revisión humana de factualidad.
- ROUGE y Token F1 miden proximidad con respuestas sintéticas gold, no verdad factual ni cumplimiento completo de R001–R008.

## Recomendación

1. Mantener este FT como candidato superior al base.
2. Agregar ejemplos dirigidos a distinguir “reporte aislado” de “problema recurrente”.
3. Reentrenar o continuar el SFT y exigir 0/13 errores en el slice de recurrencia.
4. Ejecutar una revisión humana ciega de al menos 50 pares base/FT, balanceados por tarea, evaluando groundedness, severidad, recomendación y afirmaciones inventadas.
5. Conservar `reasoning_effort="none"` en producción para esta clase de tareas cortas.

## Artefactos

- `test_heldout_200.jsonl`: dataset de test.
- `test_heldout_200.manifest.json`: método, conteos y hash reproducible.
- `predictions_base_nothink.jsonl`: respuestas del modelo base.
- `predictions_ft_nothink.jsonl`: respuestas del FT.
- `metrics_ab_nothink.json`: métricas completas A/B.
- `../scripts/evaluate_fireworks_sft.py`: runner reproducible con soporte de reanudación.

La API key se obtiene únicamente desde `FIREWORKS_API_KEY` y no se persiste en los artefactos.
