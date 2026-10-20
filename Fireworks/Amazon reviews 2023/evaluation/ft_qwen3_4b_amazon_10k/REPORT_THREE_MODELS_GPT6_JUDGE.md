# Benchmark de 200 casos — GPT-6 Sol como juez

Fecha: 2026-10-15

## Resumen ejecutivo

GPT-4.1 mini obtuvo la mejor calidad global según el juez ciego GPT-6 Sol. El fine-tuning de Qwen3-4B produjo una mejora grande y consistente respecto del modelo base: +9.49 puntos de calidad global, +22.0 puntos porcentuales de aprobación y -23.5 puntos porcentuales de errores críticos. En la tarea central de recomendación de compra, el Qwen FT alcanzó 79.60/100 y 76% de aprobación, por encima de GPT-4.1 mini en aprobación para esa tarea (58%), aunque GPT-4.1 mini siguió ganando más comparaciones globales por cobertura y calidad de respuesta.

## Resultado global según GPT-6 Sol

| Modelo | Calidad /100 | Decisión correcta | Aprobación | Error crítico | Alguna afirmación no sustentada | Victorias | Grounding | Reglas | Cobertura |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-4.1 mini | **81.61** | **99.5%** | **65.0%** | **17.0%** | 93.5% | **67.0%** | **83.34** | **80.56** | **91.06** |
| Qwen3-4B FT | 74.58 | 99.0% | 47.5% | 23.0% | **77.5%** | 23.5% | 78.68 | 74.28 | 72.39 |
| Qwen3-4B base | 65.09 | 98.5% | 25.5% | 46.5% | 95.5% | 9.5% | 68.00 | 63.47 | 72.94 |

Notas:

- “Aprobación” exige score >=75, grounding >=80, cumplimiento de reglas >=80 y ningún error crítico.
- “Error crítico” incluye una decisión invertida, un problema de seguridad inventado u omitido, una restricción dura incumplida o una afirmación no sustentada capaz de cambiar la compra.
- “Alguna afirmación no sustentada” es una tasa binaria muy sensible: cuenta el caso completo como positivo si el juez detecta al menos una. No significa que ese porcentaje de todo el texto sea incorrecto.
- La exactitud de decisión está inflada por el diseño del conjunto: 104/200 referencias son `UNDETERMINED`, 94 `CONDITIONAL_BUY` y solo 2 `BUY`. Conviene reportarla junto con las demás métricas, no aisladamente.

## Resultado por tarea

| Tarea | GPT-4.1 mini: score / pass | Qwen FT: score / pass | Qwen base: score / pass |
|---|---:|---:|---:|
| Problemas comunes | **82.14 / 58%** | 70.16 / 32% | 55.26 / 4% |
| Pros y contras | **81.76 / 74%** | 75.62 / 40% | 74.26 / 38% |
| Recomendación de compra | **80.82** / 58% | 79.60 / **76%** | 71.04 / 44% |
| Casos de uso | **81.72 / 70%** | 72.92 / 42% | 59.80 / 16% |

## Métricas de similitud con la referencia

| Modelo | Token F1 | ROUGE-L F1 | Longitud media (palabras) |
|---|---:|---:|---:|
| Qwen3-4B FT | **0.5272** | **0.3964** | 70.8 |
| GPT-4.1 mini | 0.3174 | 0.2001 | 250.4 |
| Qwen3-4B base | 0.2985 | 0.1899 | 215.6 |

El FT lidera ampliamente estas métricas porque aprendió el formato, longitud y redacción de las respuestas de entrenamiento. Esto mide similitud textual, no superioridad semántica: GPT-4.1 mini puede dar una respuesta mejor con palabras distintas y por eso ganar para el juez mientras obtiene menor Token F1/ROUGE-L.

## Rendimiento operativo observado

| Modelo | Latencia media | p50 | p95 | Tokens de salida | Casos en límite de 500 tokens |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B FT | **1.32 s** | **1.01 s** | 2.83 s | **17,670** | **0** |
| Qwen3-4B base | 1.54 s | 1.53 s | **2.07 s** | 57,978 | 5 |
| GPT-4.1 mini | 9.93 s | 5.68 s | 22.82 s | 69,066 | 23 |

Estas latencias pertenecen a deployments y proveedores diferentes, por lo que reflejan la configuración observada, no una comparación pura de arquitectura.

## Interpretación

- **Mejor calidad general:** GPT-4.1 mini. Ganó 134 de 200 casos y obtuvo el mejor score en las cuatro familias de tareas.
- **Mejor relación entre especialización, brevedad y velocidad:** Qwen3-4B FT. Reduce mucho costo de salida y latencia, y demuestra que el SFT sí produjo una mejora sustancial frente al base.
- **Área fuerte del FT:** recomendaciones de compra. Su aprobación de 76% y grounding de 84.06 son buenos; la menor cobertura (69.86) sugiere que a menudo llega a una decisión correcta pero omite parte de la evidencia útil.
- **Área débil del FT:** problemas comunes. El 28% de error crítico y 32% de aprobación indican que todavía necesita trabajo en atribución, recurrencia, severidad y separación entre hechos de catálogo y reportes individuales.
- **El modelo base no es recomendable como copiloto sin controles:** 46.5% de error crítico global y solo 25.5% de aprobación.

## Validez y siguientes pasos

Este benchmark es evidencia útil de mejora relativa, pero no una certificación de producción. GPT-6 Sol fue un único juez automatizado y las referencias provienen del mismo estilo de datos usado para el SFT. Antes de decidir un despliegue definitivo conviene:

1. Revisar manualmente una muestra estratificada de desacuerdos y errores críticos.
2. Agregar más casos con decisiones `BUY`, `DO_NOT_BUY` e `INSUFFICIENT_EVIDENCE`; la distribución actual no permite evaluar bien macro-F1 por clase.
3. Usar una salida estructurada (`Result` + `details`) y medir la etiqueta de decisión de forma determinista.
4. Repetir el juez con un segundo prompt o juez y medir acuerdo inter-juez.
5. Evaluar costo, latencia y tasa de escalamiento humano con tráfico parecido al real.

## Integridad de la ejecución

- 200 casos: 50 por cada una de cuatro tareas.
- Tres respuestas por caso, presentadas al juez como A/B/C en orden aleatorio y sin nombre de modelo.
- 200 juicios exitosos, 200 IDs únicos, cero respuestas vacías, cero rankings inválidos.
- Temperatura 0 para GPT-4.1 mini; razonamiento bajo y salida JSON estructurada para GPT-6 Sol.
