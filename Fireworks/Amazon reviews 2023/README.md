# Amazon Reviews 2023

Documentación general para analizar y preparar **Amazon Reviews 2023**, un dataset público de reseñas y metadatos de productos recopilado por McAuley Lab. El objetivo de esta carpeta es registrar decisiones de exploración, muestreo, limpieza, particionado y transformación antes de usar los datos para fine-tuning o evaluación.

> Nota: el nombre correcto es **Amazon Reviews 2023**. No es un dataset de AWS.

## Fuentes oficiales

- Dataset en Hugging Face: <https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023>
- Sitio del proyecto: <https://amazon-reviews-2023.github.io/>
- Paper: *Bridging Language and Items for Retrieval and Recommendation* — <https://arxiv.org/abs/2403.03952>
- Repositorio de datos de UCSD: <https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/>

## Resumen

| Propiedad | Valor |
|---|---:|
| Período cubierto | Mayo de 1996 a septiembre de 2023 |
| Reviews | 571,54 millones |
| Usuarios anonimizados | 54,51 millones |
| Productos | 48,19 millones |
| Categorías | 33, más registros `Unknown` |
| Tokens en reviews | 30,14 mil millones |
| Tokens en metadata | 30,78 mil millones |
| Tamaño actual del repositorio HF | Aproximadamente 750 GB |
| Idioma declarado | Inglés |

El dataset contiene tres tipos principales de información:

1. Reseñas, ratings e indicadores de compra verificada.
2. Metadatos de producto, como título, descripción, atributos y precio.
3. Relaciones entre productos y usuarios, incluyendo identificadores y asociaciones `bought_together` cuando están disponibles.

La lista completa de dominios está documentada en [categories.md](categories.md). El ranking de demos industriales y el diseño experimental propuesto están en [use_case.md](use_case.md).

## Estructura de las reseñas

Cada registro de review puede contener:

| Campo | Tipo aproximado | Descripción |
|---|---|---|
| `rating` | `float` | Rating de 1 a 5 estrellas. |
| `title` | `string` | Título de la reseña. |
| `text` | `string` | Contenido textual de la reseña. |
| `images` | `list` | Imágenes adjuntadas por el usuario, si existen. |
| `asin` | `string` | Identificador de la variante concreta del producto. |
| `parent_asin` | `string` | Identificador común usado para agrupar variantes. |
| `user_id` | `string` | Identificador anonimizado del usuario. |
| `timestamp` | `integer` | Fecha y hora Unix, expresada en milisegundos. |
| `helpful_vote` | `integer` | Cantidad de votos que marcaron la review como útil. |
| `verified_purchase` | `boolean` | Indica si Amazon registró una compra verificada. |

## Estructura de la metadata

Cada producto puede contener:

| Campo | Tipo aproximado | Descripción |
|---|---|---|
| `main_category` | `string` | Categoría principal del producto. |
| `title` | `string` | Nombre publicado del producto. |
| `average_rating` | `float` | Rating promedio en el momento de la captura. |
| `rating_number` | `integer` | Número de ratings registrados. |
| `features` | `list[string]` | Características presentadas como viñetas. |
| `description` | `list[string]` | Descripción comercial del producto. |
| `price` | `string/float/null` | Precio capturado; puede faltar o requerir normalización. |
| `images` | `list/object` | URLs y variantes de imágenes. |
| `videos` | `list/object` | Información de videos asociados. |
| `store` | `string` | Tienda o marca mostrada. |
| `categories` | `list[string]` | Jerarquía de categorías cuando está disponible. |
| `details` | `string/object` | Detalles técnicos; puede estar serializado como texto. |
| `parent_asin` | `string` | Clave para relacionar metadata y reviews. |
| `bought_together` | `list/string/null` | Productos que suelen comprarse juntos. |
| `subtitle` | `string/null` | Subtítulo opcional. |
| `author` | `string/null` | Autor, principalmente relevante para publicaciones. |

## Relaciones importantes

- La unión recomendada entre reviews y metadata se realiza mediante `parent_asin`.
- Un `parent_asin` puede representar múltiples variantes identificadas por distintos `asin`.
- El mismo usuario puede publicar reviews en varias categorías y momentos.
- Algunos productos no tienen metadata completa.
- Una review puede repetirse entre variantes o productos relacionados; es necesario detectar duplicados antes de particionar.

## Carga por categoría

No se debe descargar el repositorio completo para las primeras iteraciones. La API de `datasets` permite seleccionar una configuración concreta.

```python
from datasets import load_dataset

reviews = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Industrial_and_Scientific",
    split="full",
    streaming=True,
    trust_remote_code=True,
)

metadata = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Industrial_and_Scientific",
    split="full",
    streaming=True,
    trust_remote_code=True,
)
```

Si la versión instalada de `datasets` ya no admite scripts remotos, se pueden leer los archivos oficiales directamente:

```python
from datasets import load_dataset

base_url = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/"
    "amazon_2023/raw"
)

reviews = load_dataset(
    "json",
    data_files=f"{base_url}/review_categories/Industrial_and_Scientific.jsonl.gz",
    split="train",
    streaming=True,
)

metadata = load_dataset(
    "json",
    data_files=f"{base_url}/meta_categories/meta_Industrial_and_Scientific.jsonl.gz",
    split="train",
    streaming=True,
)
```

`streaming=True` evita materializar la categoría completa antes de comenzar a procesarla. Si posteriormente se llama a `Dataset.from_list`, `save_to_disk` o `to_parquet`, los registros seleccionados sí ocuparán espacio local.

## Casos de uso

El dataset es adecuado como fuente para:

- análisis de sentimiento y satisfacción;
- clasificación de intención o tipo de problema;
- resumen de opiniones y extracción de aspectos;
- recomendación y comparación de productos;
- búsqueda semántica de catálogo;
- detección de problemas frecuentes;
- generación de preguntas y respuestas respaldadas por evidencia;
- construcción de escenarios sintéticos de soporte o compra;
- evaluación de respuestas grounded en metadata y reviews.

No contiene directamente procesos internos de una empresa, como inventario en tiempo real, tickets de soporte, devoluciones, órdenes de trabajo, reglas de negocio o trazas de herramientas. Esos flujos deben añadirse mediante datos propios o escenarios sintéticos controlados.

## Estrategia recomendada para fine-tuning

Las reviews no deben convertirse automáticamente en pares `usuario/asistente` sin definir antes la tarea. Se recomienda:

1. Seleccionar una o pocas categorías coherentes con el caso de uso.
2. Definir una taxonomía de tareas y criterios de calidad.
3. Filtrar spam, duplicados, texto vacío, PII y contenido no deseado.
4. Relacionar reviews con metadata mediante `parent_asin`.
5. Crear los splits por producto y tiempo antes de generar ejemplos sintéticos.
6. Generar ejemplos grounded conservando los IDs de evidencia.
7. Validar automáticamente fidelidad, formato y ausencia de filtraciones.
8. Revisar manualmente una muestra y todo el conjunto de evaluación.

Una primera iteración razonable puede usar entre 100.000 y 300.000 reviews como fuente, pero producir un conjunto SFT bastante menor y más curado, por ejemplo entre 30.000 y 80.000 conversaciones.

## Particionado y prevención de leakage

No se recomienda un split aleatorio por review. Reviews del mismo producto, variantes o textos duplicados podrían quedar en conjuntos diferentes.

Estrategia inicial:

- Agrupar por `parent_asin`.
- Asignar cada producto exclusivamente a `train`, `validation` o `test`.
- Usar `timestamp` para crear un test temporal cuando sea posible.
- Deduplicar texto exacto y casi duplicado antes del split.
- Mantener un test de productos o subcategorías no vistos.
- Separar los casos sintéticos derivados de una misma evidencia en el mismo split.

Distribución orientativa:

| Split | Proporción | Uso |
|---|---:|---|
| `train` | 80 % | Optimización del modelo. |
| `validation` | 10 % | Selección de hiperparámetros y checkpoints. |
| `test` | 10 % | Evaluación final sin exposición durante el desarrollo. |

## Controles de calidad sugeridos

- Longitud mínima y máxima del texto.
- Idioma detectado.
- Presencia de `parent_asin`.
- Consistencia entre `rating` y contenido.
- Eliminación de HTML, URLs rotas y caracteres anómalos.
- Deduplicación exacta, difusa y semántica.
- Detección y tratamiento de PII.
- Balance de ratings, productos, fechas y tipos de tarea.
- Verificación de que las respuestas sintéticas estén respaldadas por la evidencia.
- Registro de procedencia, transformaciones y versión de cada ejemplo.

## Riesgos y limitaciones

- Las reviews representan usuarios que decidieron publicar, no necesariamente a todos los compradores.
- Ratings y textos pueden incluir spam, fraude, incentivos o errores.
- Los precios, descripciones y atributos reflejan el momento de captura y pueden estar obsoletos.
- La distribución entre categorías, productos y ratings es desigual.
- Las reviews pueden contener información personal o sensible pese a la anonimización de IDs.
- `verified_purchase` no garantiza que una opinión sea correcta o representativa.
- El dataset es útil para retail y recomendación, pero no equivale a conversaciones reales de atención al cliente.
- La ficha del dataset no declara de forma clara una licencia estándar para todo el contenido. Antes de uso comercial se debe revisar la procedencia, las condiciones aplicables y los derechos sobre los textos individuales.

## Convenciones propuestas para datos derivados

Cada ejemplo generado debería conservar como mínimo:

```json
{
  "id": "example-id",
  "task": "product_comparison",
  "category": "Industrial_and_Scientific",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "evidence": {
    "parent_asin": ["..."],
    "review_ids": ["..."],
    "source_split": "train"
  },
  "provenance": {
    "source": "McAuley-Lab/Amazon-Reviews-2023",
    "transformation_version": "v1"
  }
}
```

Los IDs internos de las reviews deberán ser determinísticos y no incluir el texto completo. Conviene guardar hashes de contenido para deduplicación y trazabilidad.

## Próximos entregables

- Perfil estadístico por categoría.
- Esquema normalizado de reviews y metadata.
- Reglas de filtrado y deduplicación.
- Estrategia de muestreo.
- Definición de tareas SFT y evaluación.
- Script reproducible de descarga y preparación.
- Datasheet del conjunto derivado.
