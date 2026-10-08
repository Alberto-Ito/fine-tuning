# Ranking de casos de uso para una demo industrial con fine-tuning de un modelo 8B

## Objetivo

Diseñar una demo de industria basada en un subconjunto de **Amazon Reviews 2023** que permita comparar de manera justa:

1. Un modelo base abierto de aproximadamente 8B parámetros.
2. El mismo modelo 8B después de supervised fine-tuning, idealmente con LoRA o QLoRA.
3. Un modelo frontier sin fine-tuning desplegado en Microsoft Foundry, por ejemplo `gpt-5.6-sol`.

La demo no debería intentar demostrar que un modelo 8B es mejor en inteligencia general. La hipótesis útil y defendible es más específica:

> Un modelo 8B especializado puede alcanzar o superar a un modelo frontier generalista en una tarea industrial estrecha, repetible y bien evaluada, usando menos recursos de inferencia y ofreciendo mayor control sobre formato y comportamiento.

## Aclaración sobre `gpt-5.6-sol`

`gpt-5.6-sol` es un identificador de modelo disponible en Microsoft Foundry Models sold by Azure. Microsoft documenta soporte para Responses API, razonamiento, salida estructurada, imágenes, funciones y herramientas. Su disponibilidad efectiva depende de región, tipo de deployment, cuota y suscripción.

En Foundry, la aplicación llama al **nombre del deployment**, que funciona como alias de acceso al modelo y su versión. Por lo tanto, los resultados deben registrar tanto el deployment utilizado como el model ID y versión subyacentes.

Fuentes:

- Microsoft Foundry, modelos ofrecidos por Azure: <https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure>
- Endpoints y deployments en Microsoft Foundry: <https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/endpoints>
- OpenAI Models: <https://platform.openai.com/docs/models>

## Criterios del ranking

Cada caso de uso se evalúa de 1 a 5 sobre seis dimensiones.

| Criterio | Peso | Pregunta |
|---|---:|---|
| Valor industrial | 25 % | ¿Representa un problema reconocible para una empresa? |
| Ventaja potencial del fine-tuning | 20 % | ¿El comportamiento puede aprenderse mejor que resolverse solamente con prompting? |
| Evaluación objetiva | 20 % | ¿Es posible definir respuestas, etiquetas o restricciones verificables? |
| Calidad de los datos fuente | 15 % | ¿Reviews y metadata aportan evidencia suficiente? |
| Claridad de la demo | 10 % | ¿La mejora se entiende rápidamente en una demostración? |
| Viabilidad local | 10 % | ¿Puede prepararse y entrenarse con un subconjunto manejable? |

La puntuación total se expresa sobre 10.

## Ranking ejecutivo

| Puesto | Caso de uso | Categorías principales | Puntuación | Recomendación |
|---:|---|---|---:|---|
| 1 | Copiloto de compras técnicas y compatibilidad MRO | `Industrial_and_Scientific` | 9,3 | Demo principal recomendada |
| 2 | Clasificación de defectos, reclamos y acciones sugeridas | `Industrial_and_Scientific`, `Appliances`, `Tools_and_Home_Improvement` | 8,9 | Excelente demo de automatización |
| 3 | Voice of Customer: extracción de aspectos y problemas recurrentes | `Industrial_and_Scientific`, `Electronics`, `Appliances` | 8,5 | Buena demo analítica y estructurada |
| 4 | Asistente de preguntas frecuentes grounded en catálogo y reviews | `Industrial_and_Scientific`, `Electronics` | 8,0 | Útil, pero requiere controlar el aporte de RAG |
| 5 | Recomendador personalizado de productos | Varias categorías | 6,8 | Interesante, pero menos adecuado como primera demo SFT |

---

## 1. Copiloto de compras técnicas y compatibilidad MRO

### Propuesta

Un asistente recibe una necesidad de compra industrial y debe:

- identificar los requerimientos explícitos;
- detectar especificaciones ausentes;
- formular preguntas aclaratorias;
- comparar alternativas usando únicamente evidencia disponible;
- advertir incompatibilidades o riesgos;
- producir una recomendación estructurada;
- abstenerse cuando la evidencia no alcanza.

Ejemplo:

```text
Usuario:
Necesito guantes descartables para manipular acetona durante períodos cortos.

Respuesta esperada:
- No recomendar solamente por rating.
- Solicitar concentración, duración de exposición y norma requerida.
- Distinguir resistencia química de resistencia mecánica.
- No inventar certificaciones ausentes en la metadata.
- Recomendar consultar la tabla de compatibilidad del fabricante.
```

### Por qué ocupa el primer puesto

- Tiene una narrativa industrial clara: procurement, MRO, compatibilidad y seguridad.
- `Industrial_and_Scientific` ofrece aproximadamente 5,2 millones de reviews y 427.500 productos.
- La metadata incluye atributos técnicos que permiten construir restricciones verificables.
- Un fine-tune puede enseñar consistentemente cuándo preguntar, comparar, estructurar y abstenerse.
- La diferencia entre modelo base y modelo especializado puede verse en pocos ejemplos.
- Permite medir no solamente calidad lingüística, sino cumplimiento de reglas.

### Lo que debe aprender el 8B

- Taxonomía de intención: selección, comparación, compatibilidad, reemplazo, seguridad y troubleshooting.
- Extracción normalizada de unidades y especificaciones.
- Política de preguntas aclaratorias.
- Formato de salida estable.
- Uso de evidencia y abstención.
- Separación entre datos observados e inferencias.

### Lo que no debe memorizar

- Precios actuales.
- Stock.
- Certificaciones no verificadas.
- Compatibilidades que puedan cambiar.
- Información específica de un producto que debería recuperarse desde catálogo o RAG.

### Subset recomendado

| Etapa | Volumen orientativo |
|---|---:|
| Reviews fuente descargadas/procesadas | 150.000–300.000 |
| Productos con metadata suficientemente completa | 20.000–50.000 |
| Ejemplos SFT finales | 30.000–60.000 |
| Validation | 3.000–5.000 |
| Test principal | 4.000–6.000 |
| Test adversarial y de seguridad | 500–1.000 |

### Mezcla de tareas SFT

| Tarea | Participación sugerida |
|---|---:|
| Extracción de requerimientos | 20 % |
| Preguntas aclaratorias | 15 % |
| Comparación de alternativas | 20 % |
| Recomendación grounded | 15 % |
| Compatibilidad e incompatibilidad | 10 % |
| Resumen de problemas reportados | 10 % |
| Abstención por evidencia insuficiente | 10 % |

### Salida propuesta

```json
{
  "intent": "technical_product_selection",
  "requirements": {
    "application": "chemical_handling",
    "material": null,
    "size": null,
    "required_standard": null
  },
  "missing_information": [
    "chemical concentration",
    "contact duration",
    "required safety standard"
  ],
  "recommendation_status": "needs_clarification",
  "candidate_parent_asins": [],
  "evidence": [],
  "safety_notes": [
    "Verify the manufacturer's chemical compatibility chart."
  ]
}
```

### Métricas principales

- Exact match y F1 de campos estructurados.
- Cumplimiento del esquema JSON.
- Recall de restricciones críticas.
- Tasa de preguntas aclaratorias correctas.
- Tasa de afirmaciones no respaldadas.
- Precisión de abstención.
- Preferencia humana pairwise.
- Latencia, tokens de salida y costo por caso.

---

## 2. Clasificación de defectos, reclamos y acciones sugeridas

### Propuesta

Convertir reviews negativas o mixtas en tickets estructurados:

```json
{
  "issue_type": "dimensional_mismatch",
  "severity": "medium",
  "product_component": "threaded_connector",
  "evidence_span": "...",
  "recommended_action": "request_specification_check",
  "escalate": false
}
```

### Ventajas

- La evaluación puede ser mayormente determinística.
- La clasificación y el output estructurado son tareas donde el fine-tuning suele producir mejoras visibles.
- El volumen de reviews negativas es suficiente para construir una taxonomía amplia.
- Es fácil demostrar integración con una cola simulada de soporte o calidad.

### Riesgos

- Las etiquetas no existen en el dataset original y deben generarse o anotarse.
- El rating no debe utilizarse como sustituto perfecto de severidad.
- Una acción correctiva real requiere políticas internas que Amazon Reviews no contiene.

### Dataset recomendado

- 20.000–40.000 ejemplos SFT.
- 50–100 clases iniciales es excesivo; comenzar con 12–20 tipos de problema.
- Mantener ejemplos `other`, `insufficient_information` y multilabel.
- Validar manualmente al menos 500 casos del test.

### Métricas

- Macro-F1 por tipo de defecto.
- F1 multilabel.
- Exactitud de severidad.
- Fidelidad del `evidence_span`.
- Cumplimiento del esquema.
- Matriz de confusión y desempeño por subcategoría.

---

## 3. Voice of Customer: aspectos y problemas recurrentes

### Propuesta

Extraer de cada review los aspectos mencionados y su polaridad:

```json
{
  "aspects": [
    {
      "name": "durability",
      "sentiment": "negative",
      "evidence": "the seal failed after two weeks"
    }
  ],
  "overall_issue": "premature_failure"
}
```

Luego agregar miles de resultados para producir dashboards de calidad de producto.

### Ventajas

- Conecta NLP con calidad, producto y supply chain.
- Es escalable y fácil de visualizar.
- Permite evaluar extracción de spans y etiquetas normalizadas.
- El 8B fine-tuned puede priorizar consistencia y throughput sobre razonamiento abierto.

### Limitaciones

- Un modelo frontier con un buen prompt probablemente sea fuerte en esta tarea.
- La demo debe incluir costo, latencia y estabilidad de esquema para mostrar el valor del 8B.
- La taxonomía de aspectos debe ser específica por dominio.

### Métricas

- Precision, recall y F1 de aspectos.
- Accuracy/F1 de sentimiento por aspecto.
- Fidelidad de spans.
- Consistencia entre ejecuciones.
- Throughput y costo por 1.000 reviews.

---

## 4. Asistente de preguntas frecuentes grounded

### Propuesta

Generar respuestas sobre un producto usando su metadata y un conjunto de reviews recuperadas:

- características y limitaciones;
- problemas recurrentes;
- adecuación a un escenario;
- diferencias entre productos;
- evidencia a favor y en contra.

### Ventajas

- Es visualmente atractiva como chat.
- Combina catálogo, reviews y razonamiento.
- Permite mostrar citas o referencias a evidencia.

### Motivo de su posición

El resultado depende mucho del retrieval. Si el 8B usa mejor contexto que `gpt-5.6-sol`, la comparación deja de medir fine-tuning. Ambos modelos deben recibir exactamente los mismos documentos, instrucciones y límites de tokens.

### Métricas

- Faithfulness respecto del contexto.
- Context precision y context recall.
- Tasa de citas correctas.
- Tasa de afirmaciones no respaldadas.
- Calidad pairwise con longitudes controladas.

---

## 5. Recomendador personalizado

### Propuesta

Usar historial de interacciones de usuarios para ordenar productos candidatos.

### Por qué no es la primera opción

- Es principalmente un problema de ranking, no solamente de generación.
- Requiere negativos bien construidos y splits temporales estrictos.
- La comparación con un LLM frontier puede no ser representativa.
- Métricas offline como NDCG o Recall@K no garantizan una buena conversación.
- El dataset no contiene todas las impresiones o productos que el usuario vio y decidió no comprar.

Puede ser una segunda fase, combinando un recomendador especializado con el 8B como capa conversacional.

---

## Recomendación final

La demo principal debería ser:

> **Copiloto de compras técnicas MRO con extracción de requerimientos, preguntas aclaratorias, comparación grounded y abstención segura.**

Como tarea secundaria dentro de la misma demo se puede incluir clasificación de defectos. Ambas usan la categoría `Industrial_and_Scientific`, pero muestran dos flujos empresariales diferentes:

```text
Necesidad de compra
    → extraer especificaciones
    → detectar datos faltantes
    → recuperar candidatos
    → comparar con evidencia
    → recomendar o abstenerse

Review/reclamo
    → detectar componente
    → clasificar defecto
    → estimar severidad
    → proponer siguiente acción
```

## Diseño experimental

### Modelos a comparar

| Variante | Propósito |
|---|---|
| 8B base, zero-shot | Línea base mínima. |
| 8B base, prompt optimizado/few-shot | Determina cuánto aporta solamente el prompting. |
| 8B con LoRA/QLoRA | Mide el aporte específico del fine-tuning. |
| `gpt-5.6-sol`, prompt optimizado | Referencia frontier en Microsoft Foundry. |
| Opcional: modelo Foundry intermedio | Permite construir una curva calidad/costo, no solo dos extremos. |

OpenAI recomienda establecer evals antes de invertir en fine-tuning y documenta la destilación de respuestas de un modelo grande hacia uno pequeño como estrategia válida. Los ejemplos generados por un modelo teacher no deberían incorporarse automáticamente: deben pasar filtros, graders y revisión humana.

Fuentes:

- OpenAI, supervised fine-tuning: <https://developers.openai.com/api/docs/guides/supervised-fine-tuning>
- OpenAI, evaluation best practices: <https://developers.openai.com/api/docs/guides/evaluation-best-practices>
- Microsoft Foundry model comparison: <https://learn.microsoft.com/en-us/azure/foundry/how-to/benchmark-model-in-catalog>
- Microsoft Foundry custom evaluators: <https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/custom-evaluators>

### Condiciones para una comparación justa

- Exactamente el mismo test para todos los modelos.
- Ningún ejemplo de test, producto o variante presente en training.
- Mismo contexto recuperado y mismos tools.
- Misma política de system prompt, adaptada solo cuando la API lo requiera.
- Temperatura y presupuesto de salida controlados.
- Varias ejecuciones para tareas no determinísticas.
- Registro de model ID, versión, deployment, parámetros y fecha.
- Salidas anonimizadas durante la evaluación humana.
- Control de longitud para reducir el sesgo del juez hacia respuestas extensas.
- Juez automático calibrado contra evaluadores humanos.

### Splits

Los splits se crean antes de generar ejemplos:

1. Normalizar y deduplicar reviews.
2. Agrupar por `parent_asin`.
3. Separar productos entre train, validation y test.
4. Dentro del test, reservar una sección temporal y otra de productos no vistos.
5. Mantener todos los ejemplos sintéticos derivados de la misma evidencia en el mismo split.

Propuesta:

| Split | Productos | Finalidad |
|---|---:|---|
| Train | 80 % | Fine-tuning. |
| Validation | 10 % | Hiperparámetros y selección de checkpoint. |
| Test IID | 5 % | Casos similares con productos no vistos. |
| Test temporal/OOD | 5 % | Generalización y robustez. |

### Test mínimo viable

El test debe incluir al menos estas slices:

| Slice | Casos sugeridos |
|---|---:|
| Extracción simple | 500 |
| Requerimientos múltiples | 500 |
| Información insuficiente | 500 |
| Unidades y medidas | 500 |
| Comparación de productos | 500 |
| Incompatibilidades | 500 |
| Reviews contradictorias | 500 |
| Seguridad y abstención | 500 |
| Prompts adversariales | 300 |
| Productos/subcategorías no vistos | 700 |

Total orientativo: **5.000 casos**.

## Scorecard de la demo

No se debe reducir el resultado a un único porcentaje.

| Dimensión | Métrica | Objetivo inicial del 8B fine-tuned |
|---|---|---:|
| Estructura | JSON válido | ≥ 99 % |
| Requerimientos | Micro-F1 | ≥ 0,90 |
| Restricciones críticas | Recall | ≥ 0,95 |
| Grounding | Afirmaciones respaldadas | ≥ 95 % |
| Abstención | Precision/recall | ≥ 0,85 / 0,85 |
| Preferencia humana | Win + tie frente al 8B base | ≥ 75 % |
| Frontier comparison | Win + tie frente a `gpt-5.6-sol` | Meta exploratoria, no garantía |
| Operación | Latencia p50/p95 | Reportar |
| Eficiencia | Costo por 1.000 casos | Reportar |
| Estabilidad | Variación entre repeticiones | Reportar |

No es razonable fijar por adelantado que el 8B debe vencer a `gpt-5.6-sol`. El criterio de éxito de negocio puede ser alcanzar un umbral de calidad suficiente con mejor costo, latencia, privacidad, control o capacidad de deployment.

## Guion de demostración

### Escena 1: modelo 8B base

Mostrar tres fallas típicas:

- responde sin pedir una especificación crítica;
- produce JSON inestable;
- inventa compatibilidad a partir de una review positiva.

### Escena 2: modelo 8B fine-tuned

Con los mismos inputs:

- extrae requerimientos;
- pregunta lo necesario;
- usa el esquema exacto;
- cita evidencia;
- se abstiene cuando corresponde.

### Escena 3: `gpt-5.6-sol`

Mostrar su calidad general con el mismo contexto y prompt optimizado. Comparar no solo redacción, sino scorecard, latencia y costo.

### Escena 4: casos difíciles

- reviews contradictorias;
- unidades incompatibles;
- producto bien puntuado pero inadecuado;
- intento de inducir una afirmación no respaldada;
- especificación crítica ausente.

### Escena 5: tablero final

Presentar:

- calidad por slice;
- tasa de alucinación;
- tasa de abstención;
- estabilidad del JSON;
- costo y latencia;
- ejemplos ganados, empatados y perdidos por cada modelo.

## Plan de implementación por fases

### Fase 0 — Evals antes del training

- Definir taxonomía y esquema.
- Crear 300–500 casos gold manuales.
- Evaluar 8B base y `gpt-5.6-sol`.
- Confirmar que existe margen real de mejora.

### Fase 1 — Dataset piloto

- Procesar 20.000–50.000 reviews fuente.
- Construir 5.000–10.000 ejemplos SFT.
- Entrenar LoRA/QLoRA.
- Ejecutar ablation contra few-shot.

### Fase 2 — Dataset principal

- Escalar a 30.000–60.000 ejemplos SFT.
- Incluir negativos, abstención y adversariales.
- Ajustar balance según errores del validation set.

### Fase 3 — Demo en Foundry

- Desplegar los modelos accesibles mediante endpoints comparables.
- Usar un mismo runner de inferencia y evaluación.
- Registrar telemetría, latencia y consumo.
- Ejecutar la evaluación completa y revisión humana ciega.

### Fase 4 — Decisión

Elegir entre:

- 8B fine-tuned como modelo principal;
- frontier model para todos los casos;
- routing híbrido, con 8B para casos rutinarios y frontier para casos complejos;
- 8B para extracción/clasificación y frontier para razonamiento final.

El routing híbrido es probablemente el resultado industrial más realista, incluso si la demo comienza como una comparación directa.

## Criterios de go/no-go

### Continuar con fine-tuning si

- el few-shot no alcanza los objetivos;
- los errores son sistemáticos y enseñables;
- el esquema de salida es estable y evaluable;
- existe suficiente evidencia limpia;
- el 8B fine-tuned reduce costo o latencia de manera relevante;
- el deployment y la licencia del modelo son compatibles con el objetivo.

### Detener o reformular si

- el task depende principalmente de información dinámica;
- el problema se resuelve igual de bien con RAG y prompting;
- las etiquetas sintéticas no alcanzan acuerdo humano;
- el test está contaminado con productos o evidencia de training;
- la ventaja solo aparece usando un prompt peor para el modelo frontier;
- el riesgo de seguridad exige validación externa o fuentes técnicas autoritativas.

## Decisión recomendada

Avanzar con un piloto de **copiloto de compras técnicas MRO**, usando `Industrial_and_Scientific` y un primer test gold creado antes del dataset de entrenamiento. La comparación principal debe ser:

```text
8B base + prompt optimizado
vs.
8B LoRA/QLoRA + mismo contexto
vs.
gpt-5.6-sol en Microsoft Foundry + prompt optimizado + mismo contexto
```

El mensaje de la demo debe centrarse en **especialización, control y eficiencia**, no en una supuesta superioridad general del modelo 8B.

