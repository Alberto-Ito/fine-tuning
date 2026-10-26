# Benchmark con GPT-5 — resultado final

Fecha: 2026-10-22

## Integridad

- Dataset: los mismos 103 casos de evaluación del benchmark anterior.
- Modelos: Qwen3-4B base, Qwen3-4B FT v2 y GPT-5.
- Inferencias: 103/103 válidas por modelo.
- Contrato de salida `Result`: 103/103 por modelo.
- Juez GPT-6 Sol: 103/103 juicios válidos, sin errores.
- GPT-5 se ejecutó sin `temperature`, porque el deployment rechaza ese parámetro.

## Decisión categórica contra la referencia

| Modelo | Accuracy | Macro-F1 | Weighted-F1 | Token F1 | ROUGE-L F1 |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B FT v2 | **54.37%** | **0.4607** | **0.5423** | **0.5200** | **0.3551** |
| GPT-5 | 37.86% | 0.3330 | 0.4529 | 0.3990 | 0.2595 |
| Qwen3-4B base | 27.18% | 0.2333 | 0.1689 | 0.3279 | 0.2221 |

Distribución de referencia: 51 `DO_NOT_BUY`, 24 `CONDITIONAL_BUY`, 24 `INSUFFICIENT_EVIDENCE` y 4 `BUY`.

| Modelo | BUY | CONDITIONAL_BUY | DO_NOT_BUY | INSUFFICIENT_EVIDENCE |
|---|---:|---:|---:|---:|
| Referencia | 4 | 24 | 51 | 24 |
| Qwen3-4B FT v2 | 7 | 28 | 49 | 19 |
| GPT-5 | **52** | 11 | 26 | 14 |
| Qwen3-4B base | 6 | 85 | 3 | 9 |

GPT-5 presenta un sesgo marcado hacia `BUY`: lo selecciona en 52 casos aunque la referencia lo hace en 4. Acertó los 4 casos `BUY`, pero tuvo precisión de solo 7.69% para esa clase. El FT v2 reproduce mucho mejor el balance esperado.

## Evaluación ciega con GPT-6 Sol

| Modelo | Score global | Grounding | Cobertura | Reglas | Calibración | Separación catálogo/usuarios | Accionabilidad | Preferido |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5 | **63.70** | **84.07** | **86.04** | **63.96** | **64.80** | **87.82** | **67.19** | **60/103 (58.25%)** |
| Qwen3-4B FT v2 | 60.40 | 74.65 | 68.44 | 62.36 | 60.69 | 79.99 | 62.47 | 35/103 (33.98%) |
| Qwen3-4B base | 43.12 | 62.51 | 44.08 | 41.15 | 45.21 | 62.87 | 39.90 | 8/103 (7.77%) |

| Modelo | Pass rate | Error crítico | Claim no sustentado |
|---|---:|---:|---:|
| GPT-5 | **35.92%** | 62.14% | **68.93%** |
| Qwen3-4B FT v2 | 33.01% | **58.25%** | 77.67% |
| Qwen3-4B base | 2.91% | 79.61% | 88.35% |

## Conclusión

GPT-5 produce la mejor explicación: más grounding, mejor cobertura de evidencia, mejor separación entre claims del vendedor y reportes de usuarios, y mayor preferencia global del juez. Sin embargo, el FT v2 toma decisiones categóricas mucho más alineadas con la referencia: 54.37% versus 37.86% de accuracy y 0.4607 versus 0.3330 de Macro-F1.

Para este copiloto, el FT v2 sigue siendo el mejor clasificador de la decisión de compra. GPT-5 sería mejor como redactor, verificador o segundo juez, pero su prompt/configuración actual necesita corregir el uso de `BUY` antes de considerarlo un reemplazo. Una arquitectura razonable sería usar FT v2 para la decisión y GPT-5 para revisar grounding y redactar la justificación, siempre con revisión humana para casos de alto riesgo.

GPT-6 Sol es un juez automático y sus scores no sustituyen las etiquetas de referencia. La métrica primaria para la decisión es Accuracy/Macro-F1; el juez aporta una segunda dimensión sobre la calidad de la explicación.
