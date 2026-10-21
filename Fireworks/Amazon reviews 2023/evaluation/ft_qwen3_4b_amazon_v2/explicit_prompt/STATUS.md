# Benchmark con prompt explícito — resultado final

Fecha: 2026-10-20

## Integridad de la corrida

- Dataset: 103 casos de `data/dataset/evaluation.jsonl`.
- Los tres modelos produjeron 103/103 respuestas válidas.
- Los tres modelos respetaron el formato explícito `Result` en 103/103 casos.
- GPT-6 Sol produjo 103/103 juicios válidos, sin errores.

## Métricas determinísticas

| Modelo | Accuracy | Macro-F1 | Weighted-F1 | Token F1 | ROUGE-L F1 | Estructura completa |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B FT v2 | **54.37%** | **0.4607** | **0.5423** | **0.5200** | **0.3551** | 100% |
| GPT-4.1 mini | 43.69% | 0.3749 | 0.3959 | 0.3767 | 0.2367 | 100% |
| Qwen3-4B base | 27.18% | 0.2333 | 0.1689 | 0.3279 | 0.2221 | 100% |

## Evaluación ciega con GPT-6 Sol

| Modelo | Score global | Grounding | Cobertura | Reglas | Calibración | Separación catálogo/usuarios | Accionabilidad | Preferido |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B FT v2 | **61.00** | **75.17** | 68.91 | **62.72** | **62.40** | **80.73** | **62.92** | **66/103 (64.08%)** |
| GPT-4.1 mini | 53.64 | 65.32 | **73.91** | 50.92 | 53.47 | 64.50 | 58.85 | 30/103 (29.13%) |
| Qwen3-4B base | 42.75 | 62.40 | 43.35 | 42.22 | 46.34 | 63.71 | 39.79 | 7/103 (6.80%) |

| Modelo | Pass rate | Error crítico | Claim no sustentado |
|---|---:|---:|---:|
| Qwen3-4B FT v2 | **34.95%** | **59.22%** | **75.73%** |
| GPT-4.1 mini | 10.68% | 69.90% | 99.03% |
| Qwen3-4B base | 1.94% | 81.55% | 86.41% |

## Conclusión

El fine-tuning v2 produjo una mejora sustancial y consistente: duplica la accuracy del Qwen base, supera a GPT-4.1 mini en accuracy, Macro-F1, Weighted-F1, similitud textual y score global del juez, y es la respuesta preferida en 66 de 103 casos.

Todavía no debe considerarse listo para compras autónomas: el juez detectó errores críticos en 59.22% de los casos y algún claim no sustentado en 75.73%. El resultado respalda usarlo como copiloto con revisión humana y salida estructurada, no como decisor final sin controles.

GPT-6 Sol es un juez automático, no verdad absoluta. Sus scores sirven para comparación relativa bajo la misma rúbrica. Accuracy y F1 contra las etiquetas humanas siguen siendo las métricas principales para la decisión categórica.
