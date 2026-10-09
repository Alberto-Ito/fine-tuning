# Datos locales

Esta carpeta contendrá los datos fuente seleccionados de `Industrial_and_Scientific`.

El script de preparación generará:

```text
industrial_and_scientific_150k/
├── reviews.jsonl.gz   # 150.000 reviews normalizadas
├── metadata.jsonl.gz  # metadata de los parent_asin seleccionados
├── enriched_reviews.jsonl.gz # reviews unidas con metadata de producto
├── product_plan.json  # productos, cohorte y cantidad objetivo
└── manifest.json      # procedencia, parámetros y estadísticas
```

`reviews.jsonl.gz` y `metadata.jsonl.gz` son la capa **raw curated**.
`enriched_reviews.jsonl.gz` es la capa intermedia unificada que deberá consumir
el generador de train, validation y test. Todavía no contiene conversaciones
SFT. El particionado posterior deberá agrupar por `source.parent_asin` para
impedir que reviews del mismo producto aparezcan en splits diferentes.

No deben versionarse archivos de datos grandes ni contenido derivado que pueda incluir información sensible. Antes de publicar o usar comercialmente el subset, se debe revisar la licencia, procedencia y tratamiento de PII.

## Estado actual

Subset generado y validado el 8 de octubre de 2026:

| Métrica | Resultado |
|---|---:|
| Reviews | 150.000 |
| Reviews únicas | 150.000 |
| Productos (`parent_asin`) | 14.000 |
| Productos con metadata | 14.000 |
| Duplicados de review | 0 |
| Metadata duplicada | 0 |
| Campos requeridos ausentes | 0 |
| Compra verificada | 138.185 |
| Tamaño local total | 43 MB |

Distribución observada de ratings:

| Rating | Reviews |
|---:|---:|
| 1 | 19.039 |
| 2 | 8.106 |
| 3 | 10.483 |
| 4 | 17.580 |
| 5 | 94.792 |
