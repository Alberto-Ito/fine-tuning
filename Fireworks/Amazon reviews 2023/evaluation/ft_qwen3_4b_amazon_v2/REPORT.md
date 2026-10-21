# Benchmark ft-qwen3-4b-amazon-v2

Fecha: 2026-10-20

## Diseño

- Dataset: `data/dataset/evaluation.jsonl`.
- Casos: 103 recomendaciones de compra.
- Entrada: mensajes `system` y `user`; la respuesta `assistant` se reservó como referencia.
- Modelos: Qwen3-4B base, ft-qwen3-4b-amazon-v2 y GPT-4.1 mini.
- Generación: temperatura 0; razonamiento desactivado para Qwen.
- Juez: GPT-6 Sol, presentación ciega A/B/C y orden aleatorio determinista por caso.
- Integridad: 103 respuestas válidas por modelo, 103 juicios válidos, cero IDs duplicados, cero rankings inválidos y cero respuestas vacías.

## Resultado principal

| Modelo | Score GPT-6 /100 | Accuracy de decisión | Macro-F1 | Weighted-F1 | Pass rate | Error crítico | Victorias |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-4B FT v2** | **59.24** | **45.63%** | **0.3136** | **0.4497** | **30.10%** | **66.02%** | **72.82%** |
| Qwen3-4B base | 41.61 | 27.18% | 0.2696 | 0.2807 | 0.97% | 80.58% | 15.53% |
| GPT-4.1 mini | 39.48 | 31.07% | 0.2100 | 0.3465 | 3.88% | 79.61% | 11.65% |

El FT v2 es claramente el mejor de este benchmark y mejora al Qwen base en 17.63 puntos de score, 18.45 puntos porcentuales de accuracy, 29.13 puntos porcentuales de aprobación y 14.56 puntos porcentuales menos de errores críticos. Sin embargo, sus valores absolutos todavía no son adecuados para decisiones autónomas de compra.

## Calidad semántica según GPT-6 Sol

| Modelo | Grounding | Cobertura | Reglas | Severidad | Separación catálogo/usuario | Actionability |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen3-4B FT v2** | **75.13** | **67.65** | **62.11** | **62.53** | **82.38** | **59.89** |
| Qwen3-4B base | 59.34 | 40.19 | 38.93 | 43.62 | 62.71 | 35.55 |
| GPT-4.1 mini | 52.39 | 47.28 | 36.99 | 41.16 | 54.69 | 39.19 |

## Distribución y calibración de decisiones

| Etiqueta | Referencia | Qwen FT v2 | Qwen base | GPT-4.1 mini |
|---|---:|---:|---:|---:|
| BUY | 4 | 37 | 18 | 67 |
| CONDITIONAL_BUY | 24 | 4 | 32 | 5 |
| DO_NOT_BUY | 51 | 51 | 19 | 31 |
| INSUFFICIENT_EVIDENCE | 24 | 11 | 34 | 0 |

Aunque el total de `DO_NOT_BUY` del FT coincide con la referencia, no siempre corresponde a los mismos casos. Su principal problema es la sobrepredicción de `BUY` y el uso insuficiente de `CONDITIONAL_BUY` e `INSUFFICIENT_EVIDENCE`.

### F1 por clase

| Modelo | BUY | CONDITIONAL_BUY | DO_NOT_BUY | INSUFFICIENT_EVIDENCE |
|---|---:|---:|---:|---:|
| Qwen FT v2 | 0.1951 | 0.1429 | **0.7451** | 0.1714 |
| Qwen base | **0.2727** | **0.2500** | 0.3143 | **0.2414** |
| GPT-4.1 mini | 0.1127 | 0.0690 | 0.6585 | 0.0000 |

El FT mejoró especialmente la clase mayoritaria `DO_NOT_BUY`, pero todavía aprende de forma insuficiente las distinciones entre compra directa, compra condicionada y evidencia insuficiente. Por eso su accuracy ponderada mejora mucho más que su macro-F1.

## Similitud con las referencias

| Modelo | Token F1 | ROUGE-L F1 | Longitud media (palabras) |
|---|---:|---:|---:|
| **Qwen3-4B FT v2** | **0.5124** | **0.3473** | 100.7 |
| GPT-4.1 mini | 0.2781 | 0.1502 | 86.9 |
| Qwen3-4B base | 0.2398 | 0.1442 | 64.2 |

El FT aprendió claramente la estructura, longitud y redacción del dataset. La diferencia entre esta alta similitud textual y su accuracy de 45.63% muestra que imitar el formato no garantiza seleccionar correctamente la etiqueta.

## Rendimiento operativo observado

| Modelo | Latencia media | p50 | p95 | Tokens salida totales | Tokens salida/caso |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B base | **1.19 s** | **1.01 s** | **2.59 s** | **9,315** | **90.4** |
| Qwen3-4B FT v2 | 2.22 s | 1.14 s | 9.87 s | 14,702 | 142.7 |
| GPT-4.1 mini | 2.80 s | 2.60 s | 4.93 s | 11,370 | 110.4 |

Las latencias pertenecen a proveedores y deployments distintos. Durante la primera pasada, ambos deployments Fireworks escalaron desde cero y produjeron errores 503 transitorios; esos casos se reintentaron y el conjunto final no contiene errores. Las cifras de latencia corresponden a las inferencias exitosas guardadas.

## Conclusión

El fine-tuning v2 produjo una mejora material y ganó 75 de 103 comparaciones ciegas. No obstante, un 66.02% de casos con al menos un error crítico y una macro-F1 de 0.3136 impiden recomendarlo como decisor autónomo.

Prioridades para la siguiente versión:

1. Incrementar y balancear ejemplos de `CONDITIONAL_BUY` e `INSUFFICIENT_EVIDENCE`.
2. Incluir contrastes cercanos donde solo cambie el dato que separa `BUY`, `CONDITIONAL_BUY` e `INSUFFICIENT_EVIDENCE`.
3. Mantener la etiqueta en un campo estructurado separado del texto explicativo.
4. Aplicar una regla de seguridad o clasificador determinista que bloquee `BUY` cuando falte evidencia para un requisito obligatorio.
5. Revisar manualmente los falsos `BUY`, por ser el error de mayor riesgo para un copiloto de compras.

## Limitaciones

- Los 103 ejemplos pertenecen a una única tarea; los resultados no miden pros/contras, casos de uso o síntesis de problemas.
- GPT-6 Sol es un único juez automatizado. Conviene calibrar una muestra con evaluadores humanos.
- Solo existen cuatro referencias `BUY`, por lo que el F1 de esa clase tiene alta varianza.
- Este conjunto es distinto del benchmark anterior de 200 casos; no deben compararse sus scores como si fueran el mismo test.
