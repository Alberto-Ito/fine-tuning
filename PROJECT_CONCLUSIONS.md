# Conclusiones del proyecto: fine-tuning de modelos pequeños para clasificación

**Fecha de corte:** 1 de octubre de 2026  
**Caso de estudio:** clasificación de intención sobre Banking77 (77 clases)

## Resumen ejecutivo

Las pruebas realizadas sostienen la hipótesis técnica central del proyecto: para una tarea acotada, estable y con etiquetas conocidas, un modelo pequeño especializado mediante fine-tuning puede igualar o superar a un modelo generalista mucho más grande.

Sobre el mismo conjunto de test de 3.080 ejemplos, **Qwen3.5-0.8B con LoRA alcanzó 86,88% de accuracy y 86,89% de macro-F1**, frente a **85,19% y 84,37% de GPT-5.6 Luna directo**. Es decir, el modelo de 0,8B obtuvo **+1,69 puntos porcentuales de accuracy y +2,52 puntos de macro-F1**. Qwen3-0.6B también quedó prácticamente a la par: 85,06% de accuracy y 85,09% de macro-F1.

La diferencia operativa es todavía más marcada. En el test completo, Qwen3.5 consumió 38.779 tokens de entrada y no generó tokens de salida, mientras que GPT-5.6 Luna directo reportó 1.368.780 tokens totales. Esto representa **35,3 veces más tokens para el modelo grande**. La versión de GPT-5.6 Luna detrás de un agente persistido reportó 15.101.827 tokens totales —96,7% de su entrada fue cacheada—, o **389,4 veces el volumen del clasificador pequeño**.

Por lo tanto, el resultado actual demuestra **paridad o superioridad de calidad y una ventaja estructural de costo por inferencia**. Lo que todavía no está cerrado es el ahorro exacto en dólares ni el volumen exacto de amortización: el repositorio no registra las tarifas efectivamente pagadas en Foundry ni el costo horario del hardware de serving. Esa conclusión económica debe expresarse con la fórmula de break-even incluida más abajo y completarse con precios reales del entorno de producción.

## Objetivo del proyecto

Validar si conviene reemplazar, para tareas repetitivas de clasificación, el uso continuo de un modelo generalista grande por un modelo pequeño especializado.

La tesis es la siguiente:

> Cuando el dominio y la salida están bien definidos, el costo inicial de preparar datos y ajustar un modelo pequeño se paga una sola vez. Después, cada inferencia es más simple y barata. Si la calidad se mantiene, el ahorro acumulado supera rápidamente la inversión inicial.

El proyecto no busca afirmar que un modelo pequeño reemplaza a uno grande en cualquier tarea. La conclusión aplica a flujos estrechos y evaluables, como clasificación de intención con un catálogo fijo de etiquetas.

## Metodología y comparabilidad

- Dataset: `PolyAI/banking77`.
- Entrenamiento: 9.002 ejemplos; validación: 1.001; test: 3.080.
- Salida: una de 77 clases.
- Modelos pequeños: Qwen3-0.6B y Qwen3.5-0.8B ajustados con LoRA durante dos épocas.
- Modelo grande directo: GPT-5.6 Luna, con instrucciones y lista completa de etiquetas en cada request.
- Agentes: un agente basado en GPT-4o y otro en GPT-5.6 Luna, ambos con instrucciones persistidas del lado del servidor.
- Todos los resultados de calidad consolidados corresponden al mismo split de test de 3.080 ejemplos.

## Resultados consolidados

| Modelo / modalidad | Accuracy | Macro-F1 | Tokens totales en test | Tokens por caso | Latencia media | Observación |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B LoRA, 2 épocas | 85,06% | 85,09% | 38.812 | 12,60 | No medida de forma comparable por request | Clasificador local |
| **Qwen3.5-0.8B LoRA, 2 épocas** | **86,88%** | **86,89%** | **38.779** | **12,59** | Benchmark por lotes | Mejor calidad global |
| GPT-5.6 Luna directo | 85,19% | 84,37% | 1.368.780 | 444,41 | 1,781 s | Prompt y 77 etiquetas por request |
| Agente GPT-4o v2 | 78,15% | 77,39% | 2.823.501 | 916,72 | 2,066 s | Una salida fuera del catálogo |
| Agente GPT-5.6 Luna v3 | 84,71% | 83,99% | 15.101.827 | 4.903,19 | 2,379 s | 96,7% de input cacheado; una salida inválida |

### Lectura de calidad

- Qwen3.5-0.8B fue el mejor modelo: supera a GPT-5.6 Luna directo en 1,69 pp de accuracy y 2,52 pp de macro-F1.
- Qwen3-0.6B logra esencialmente la misma accuracy que GPT-5.6 Luna directo (-0,13 pp) y un macro-F1 mayor (+0,72 pp).
- Encapsular el modelo grande en un agente no mejoró la tarea: el agente Luna quedó 2,18 pp por debajo de Qwen3.5 en accuracy; el agente GPT-4o quedó 8,73 pp por debajo.
- El modelo especializado además restringe la salida a logits sobre las 77 clases. Los agentes generativos produjeron al menos una respuesta fuera del catálogo, un riesgo operativo inexistente en el clasificador.

### Lectura de eficiencia

- GPT-5.6 Luna directo utilizó **35,3 veces** más tokens totales que Qwen3.5 para resolver exactamente los mismos casos.
- El agente GPT-5.6 Luna utilizó **389,4 veces** más tokens totales que Qwen3.5. El caching puede reducir el precio de gran parte de esos tokens, pero no elimina el procesamiento ni permite equiparar directamente token bruto con costo facturado.
- En Apple MPS, Qwen3.5 procesó entre 3,04 y 4,41 requests/s según la corrida de benchmark. GPT-5.6 Luna directo, ejecutado secuencialmente vía API, tuvo una latencia media de 1,781 s por request (aproximadamente 0,56 requests/s por flujo secuencial).
- Qwen3-0.6B fue aproximadamente dos veces más rápido que Qwen3.5 en el benchmark limpio (6,11 vs. 3,04 requests/s), a cambio de 1,82 pp de accuracy.

## Inversión inicial observada

El entrenamiento completo de Qwen3.5-0.8B por dos épocas tomó **14.883 segundos (4,13 horas)** en Apple MPS, incluidas las evaluaciones. Se usaron 18.004 exposiciones aproximadas a ejemplos de entrenamiento y 2.252 pasos de optimización. El pico observado fue 1,59 GB de memoria MPS asignada y 5,10 GB de memoria del driver.

Para Qwen3-0.6B, la segunda época tomó 77,1 minutos incluyendo nueve evaluaciones; el artefacto final de adaptador y tokenizer ocupó 20 MB. Como esa medición no incluye de forma consolidada la primera época, no debe usarse como costo total de entrenamiento.

La corrida de dos épocas es el punto válido para la comparación. Las pruebas de una tercera época sufrieron una degradación abrupta luego de la época 2 y terminaron con 38,47% de accuracy para Qwen3-0.6B y 32,31% para Qwen3.5-0.8B. La progresión de validación muestra que Qwen3.5 había alcanzado aproximadamente 86,9% cerca de la época 1,89 antes del colapso. Esto indica una inestabilidad de entrenamiento o reanudación que debe investigarse; **no indica que dos épocas sean insuficientes** y no invalida los checkpoints buenos ya evaluados.

## Modelo económico y punto de amortización

Definimos:

- `F`: inversión inicial del modelo pequeño (preparación de datos + entrenamiento + evaluación + despliegue).
- `c_s`: costo marginal por request del modelo pequeño en producción.
- `c_g`: costo por request del modelo grande vía API.
- `N`: cantidad acumulada de requests.

Entonces:

```text
Costo acumulado del modelo pequeño = F + N × c_s
Costo acumulado del modelo grande  =     N × c_g

Punto de equilibrio:
N_break-even = F / (c_g - c_s), siempre que c_g > c_s
```

Para expresar el costo por request con los datos disponibles:

```text
c_g_directo = (436,36 × P_input + 8,05 × P_output) / 1.000.000
```

donde `P_input` y `P_output` son los precios reales en USD por millón de tokens del deployment. Para el agente Luna deben separarse 162,42 tokens de entrada no cacheada, 4.731,94 tokens cacheados y 8,83 tokens de salida por request, aplicando la tarifa correspondiente a cada componente.

En un serving dedicado del modelo pequeño, si `H_s` es el costo horario de la instancia y se sostienen 3,04 requests/s:

```text
c_s ≈ H_s / (3,04 × 3.600) = H_s / 10.944
```

Esta aproximación supone utilización continua. Con poco tráfico, una instancia siempre encendida puede elevar el costo unitario; con batching, autoscaling o hardware CUDA optimizado, puede bajarlo. El entrenamiento de Qwen3.5 agrega aproximadamente `4,13 × H_train` al componente técnico de `F`, más el costo humano de preparación y validación.

### Escenarios para comunicar el retorno

| Escenario | Diferencia marginal `c_g - c_s` | Inversión inicial `F` | Break-even |
|---|---:|---:|---:|
| Conservador | USD 0,001 por request | USD 500 | 500.000 requests |
| Base | USD 0,005 por request | USD 500 | 100.000 requests |
| Alto ahorro | USD 0,010 por request | USD 500 | 50.000 requests |

Estos tres casos son **ilustrativos**, no costos medidos. Sirven para mostrar la sensibilidad del retorno: aun con una inversión inicial moderada, una diferencia pequeña pero recurrente en el costo unitario se amortiza con volumen. La tabla debe reemplazarse por tarifas de Foundry, costo horario de serving y horas reales de ingeniería antes de presentarla como business case financiero definitivo.

## Extrapolación a modelos y volúmenes mayores

Los resultados permiten extrapolar el patrón, no una cifra exacta:

1. **La especialización reduce la complejidad necesaria.** Para una salida cerrada de 77 clases, un modelo de 0,8B alcanzó mejor calidad que el modelo grande. Aumentar el tamaño del modelo generalista no garantiza una mejora cuando gran parte de su capacidad no es necesaria para la tarea.
2. **El overhead del prompt crece con el catálogo y las instrucciones.** El modelo grande directo vuelve a recibir reglas y etiquetas en cada request. El clasificador ajustado incorpora esa información en sus pesos y sólo necesita el texto de entrada.
3. **La ventaja acumulada crece linealmente con el tráfico.** El fine-tuning es un costo fijo; la diferencia de inferencia se repite en cada llamada. Por eso, cuanto mayor es el volumen y más estable es la tarea, más favorable es el modelo pequeño.
4. **Subir de 0,6B a 0,8B mostró una frontera calidad/costo medible.** Se ganaron 1,82 pp de accuracy, con aproximadamente la mitad del throughput y 42,5% más memoria MPS asignada. Esto sugiere seleccionar el modelo más pequeño que cumpla el SLA, no simplemente el más grande disponible.
5. **Para tareas más complejas puede hacer falta un modelo pequeño de mayor capacidad**, por ejemplo de varios miles de millones de parámetros. La hipótesis económica puede seguir siendo válida frente a un modelo frontier, pero debe revalidarse: calidad, hardware necesario, throughput, costo por hora y volumen de break-even cambian.

No sería válido extrapolar directamente que un modelo de 0,8B mantendrá esta ventaja en generación abierta, razonamiento general, clases que cambian constantemente o inputs fuera de dominio. En esos casos puede convenir un enfoque híbrido: modelo pequeño para el camino normal y escalamiento al modelo grande ante baja confianza o casos no reconocidos.

## Punto actual del proyecto

### Validado

- Pipeline reproducible y configurable para entrenamiento, evaluación y predicción con LoRA.
- Comparación sobre el mismo test entre dos modelos pequeños, GPT-5.6 Luna directo y dos configuraciones de agente.
- Paridad/superioridad de calidad del modelo pequeño especializado.
- Medición de tokens, tiempos de entrenamiento, memoria, throughput y latencia de las alternativas.
- Evidencia de que dos épocas son suficientes para el mejor resultado observado.

### Pendiente para cerrar el caso de negocio

- Incorporar el precio real facturado de Foundry, distinguiendo input, output y cached input.
- Medir costo y throughput del modelo pequeño en el hardware objetivo de producción, idealmente CUDA y con concurrencia/batching representativos.
- Incluir horas de ingeniería, etiquetado y mantenimiento en `F`.
- Repetir el entrenamiento con más de una semilla y reportar media y dispersión.
- Investigar la degradación de la tercera época y agregar alertas/early stopping para impedir que un checkpoint peor reemplace al mejor.
- Evaluar robustez ante datos fuera de dominio, cambios de distribución y clases ambiguas.
- Definir un umbral de confianza y un fallback al modelo grande para casos difíciles.

## Conclusión

El experimento ya prueba el punto técnico principal: **para Banking77, un modelo de menos de mil millones de parámetros ajustado específicamente no sólo iguala al modelo grande, sino que lo supera en las métricas de calidad observadas y elimina gran parte del overhead de inferencia**.

La recomendación es avanzar con **Qwen3.5-0.8B LoRA como candidato principal** cuando se prioriza calidad, y considerar **Qwen3-0.6B** cuando throughput y memoria sean más importantes. Para producción, el diseño más robusto es un clasificador pequeño como ruta principal y un modelo grande como fallback selectivo.

La afirmación financiera correcta, con la evidencia actual, es: **la arquitectura tiene una ventaja económica estructural y el costo inicial debería amortizarse a medida que crece el volumen**. Para convertir “debería” en un número auditado, sólo falta completar las tarifas reales en la fórmula de break-even y correr el benchmark en la infraestructura final.

## Fuentes internas

- `OLD_Qwen3.5-0.8B/reports/tables/qwen3_vs_qwen35_banking77.md`
- `OLD_Qwen3.5-0.8B/outputs/banking77/metrics/two_epochs/run_summary.json`
- `OLD_Qwen3-0.6B/outputs/banking77/metrics/two_epochs/run_summary.json`
- `MS Foundry/gpt-5.6-luna/outputs/test_metrics.json`
- `MS Foundry/gpt-4o_agent/outputs/test_metrics.json`
- `MS Foundry/gpt-5.6-luna_agent/outputs/test_metrics.json`
- `outputs/banking77/qwen3_06b/epoch3_metrics/run_summary.json`
- `outputs/banking77/qwen35_08b/epoch3_metrics/run_summary.json`
