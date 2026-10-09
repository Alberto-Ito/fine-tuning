# Scripts de preparación

Estos scripts preparan el subset fuente de `Industrial_and_Scientific`. No crean todavía los datasets SFT de train, validation y test.

`join_reviews_metadata.py` une cada review curada con la ficha de su producto
mediante `parent_asin`. Genera `enriched_reviews.jsonl.gz`, el dataset intermedio
canónico que deberá consumir el futuro generador de splits y ejemplos SFT. Los
campos legibles se organizan bajo `product` y `review`; los identificadores se
conservan bajo `source` para trazabilidad, deduplicación y particionado por
producto.

```bash
python3 scripts/join_reviews_metadata.py \
  --data-dir data/industrial_and_scientific_150k
```

## `build_industrial_subset.py`

Lee por streaming los archivos oficiales publicados en Hugging Face. No clona el repositorio ni guarda los archivos fuente completos.

Proceso:

1. Primera pasada completa: cuenta reviews válidas por `parent_asin` sin guardarlas.
2. Selecciona determinísticamente 4.000 productos core con al menos 20 reviews.
3. Asigna entre 20 y 40 reviews por producto core hasta sumar 120.000.
4. Selecciona 10.000 productos adicionales para long-tail, con 3 reviews cada uno.
5. Segunda pasada completa: conserva por producto las reviews con menor hash determinístico.
6. Recorre por streaming la metadata de la categoría.
7. Guarda únicamente metadata cuyo `parent_asin` fue seleccionado.
8. Escribe el plan de productos y un manifiesto reproducible.

Outputs previstos:

```text
data/industrial_and_scientific_150k/
├── reviews.jsonl.gz
├── metadata.jsonl.gz
├── product_plan.json
└── manifest.json
```

Comando previsto, todavía no ejecutado:

```bash
python3 scripts/build_industrial_subset.py
```

Vista de parámetros sin descargar datos:

```bash
python3 scripts/build_industrial_subset.py --help
```

## `validate_industrial_subset.py`

Lee solamente los outputs locales y comprueba:

- cantidad esperada de reviews;
- JSON válido;
- campos requeridos;
- IDs duplicados;
- distribución por rating;
- cobertura de metadata por `parent_asin`.

Comando previsto:

```bash
python3 scripts/validate_industrial_subset.py
```

## Distribución por producto

| Cohorte | Productos | Reviews por producto | Reviews totales |
|---|---:|---:|---:|
| Core | 4.000 | 20–40 | 120.000 |
| Long-tail | 10.000 | 3 | 30.000 |
| Total | 14.000 | — | 150.000 |

El subset conserva la distribución natural de ratings dentro de los productos elegidos. El balance de tareas y casos negativos se realizará al construir el dataset SFT, sin distorsionar esta capa fuente.

## Transferencia y almacenamiento

- El repositorio completo de aproximadamente 750 GB no se descarga.
- Solo se accede a los dos archivos oficiales de `Industrial_and_Scientific`.
- El archivo de reviews de la categoría se recorre dos veces: una para contar y otra para seleccionar. No se persiste completo.
- La metadata de la categoría sí se recorre completa para localizar todos los `parent_asin` seleccionados, pero solo las coincidencias se guardan.
- Los archivos temporales se escriben dentro del directorio de salida y se renombran al finalizar, evitando outputs finales incompletos.
- Una ejecución existente no se sobrescribe automáticamente.
