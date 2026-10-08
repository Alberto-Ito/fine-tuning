# Categorías de Amazon Reviews 2023

El dataset declara **33 categorías comerciales**, además de `Unknown` para registros que no pudieron asignarse a una categoría conocida. Los conteos corresponden a la ficha oficial y representan la cantidad aproximada de ratings/reviews.

| Categoría | Reviews | Descripción breve |
|---|---:|---|
| `All_Beauty` | 701,5 mil | Productos generales de belleza que no están limitados a una subcategoría específica, como fragancias, maquillaje, uñas y cuidado personal. |
| `Amazon_Fashion` | 2,5 millones | Selección transversal de moda de Amazon, incluyendo ropa, accesorios y productos de estilo personal. |
| `Appliances` | 2,1 millones | Electrodomésticos grandes y pequeños, repuestos y accesorios relacionados con su uso y mantenimiento. |
| `Arts_Crafts_and_Sewing` | 9,0 millones | Materiales para arte, manualidades, costura, tejido, pintura, decoración y fabricación artesanal. |
| `Automotive` | 20,0 millones | Repuestos, accesorios, herramientas y consumibles para automóviles, motocicletas y mantenimiento vehicular. |
| `Baby_Products` | 6,0 millones | Productos para bebés y cuidadores, como alimentación, transporte, higiene, seguridad y mobiliario infantil. |
| `Beauty_and_Personal_Care` | 23,9 millones | Cosmética, cuidado de piel y cabello, higiene personal, fragancias y dispositivos de belleza. |
| `Books` | 29,5 millones | Libros impresos de ficción, no ficción, educación, referencia y otras áreas editoriales. |
| `CDs_and_Vinyl` | 4,8 millones | Música publicada en CD, vinilo y otros formatos físicos. |
| `Cell_Phones_and_Accessories` | 20,8 millones | Teléfonos móviles, fundas, protectores, cargadores, cables, soportes y otros accesorios. |
| `Clothing_Shoes_and_Jewelry` | 66,0 millones | Ropa, calzado, joyería, relojes y accesorios para diferentes públicos y ocasiones. |
| `Digital_Music` | 130,4 mil | Álbumes, canciones y otros productos musicales distribuidos digitalmente. |
| `Electronics` | 43,9 millones | Electrónica de consumo, audio, video, computación, redes, cámaras, componentes y accesorios. |
| `Gift_Cards` | 152,4 mil | Tarjetas de regalo físicas o digitales y productos equivalentes de crédito prepago. |
| `Grocery_and_Gourmet_Food` | 14,3 millones | Alimentos, bebidas, ingredientes, snacks, productos gourmet y artículos de despensa. |
| `Handmade_Products` | 664,2 mil | Productos elaborados artesanalmente, personalizados o fabricados en pequeñas series. |
| `Health_and_Household` | 25,6 millones | Salud doméstica, limpieza, cuidado del hogar, suplementos, primeros auxilios y consumibles cotidianos. |
| `Health_and_Personal_Care` | 494,1 mil | Productos de salud y cuidado personal pertenecientes a una taxonomía histórica o más específica del catálogo. |
| `Home_and_Kitchen` | 67,4 millones | Cocina, hogar, muebles, organización, textiles, decoración y utensilios domésticos. |
| `Industrial_and_Scientific` | 5,2 millones | Suministros industriales, científicos y MRO: PPE, laboratorio, medición, fijaciones, adhesivos, limpieza, embalaje y componentes técnicos. |
| `Kindle_Store` | 25,6 millones | Libros electrónicos y publicaciones digitales distribuidas para dispositivos y aplicaciones Kindle. |
| `Magazine_Subscriptions` | 71,5 mil | Suscripciones a revistas y publicaciones periódicas. |
| `Movies_and_TV` | 17,3 millones | Películas, series y otros contenidos audiovisuales, principalmente en formatos físicos o ediciones comerciales. |
| `Musical_Instruments` | 3,0 millones | Instrumentos musicales, equipos de estudio, audio profesional, accesorios y repuestos. |
| `Office_Products` | 12,8 millones | Papelería, útiles, mobiliario, impresión, organización y suministros de oficina. |
| `Patio_Lawn_and_Garden` | 16,5 millones | Jardinería, patio, exteriores, riego, mobiliario exterior y mantenimiento de espacios verdes. |
| `Pet_Supplies` | 16,8 millones | Alimentación, higiene, salud, entrenamiento, transporte y accesorios para mascotas. |
| `Software` | 4,9 millones | Aplicaciones, sistemas, licencias y software distribuido física o digitalmente. |
| `Sports_and_Outdoors` | 19,6 millones | Deportes, fitness, camping, recreación, actividades al aire libre y equipamiento asociado. |
| `Subscription_Boxes` | 16,2 mil | Cajas de productos entregadas periódicamente bajo un modelo de suscripción. |
| `Tools_and_Home_Improvement` | 27,0 millones | Herramientas, ferretería, iluminación, electricidad, plomería, construcción y mejoras del hogar. |
| `Toys_and_Games` | 16,3 millones | Juguetes, juegos de mesa, rompecabezas, coleccionables y productos recreativos infantiles o familiares. |
| `Video_Games` | 4,6 millones | Videojuegos, consolas, periféricos, accesorios y ediciones físicas o digitales. |
| `Unknown` | 63,8 millones | Registros cuyo producto no quedó asociado de manera confiable con una de las 33 categorías declaradas. No debe tratarse como un dominio homogéneo. |

## Selección orientativa por caso de uso

### Retail general

- `Home_and_Kitchen`
- `Electronics`
- `Clothing_Shoes_and_Jewelry`
- `Beauty_and_Personal_Care`
- `Grocery_and_Gourmet_Food`

Estas categorías ofrecen gran volumen y diversidad, pero requieren muestreo para evitar que productos populares dominen la distribución.

### Catálogo y compras B2B

- `Industrial_and_Scientific`
- `Office_Products`
- `Tools_and_Home_Improvement`
- `Electronics`
- `Health_and_Household`

Son útiles para recomendación basada en atributos, compatibilidad, especificaciones y restricciones de compra.

### Primera prueba en una máquina local

- `All_Beauty`
- `Handmade_Products`
- `Digital_Music`
- `Gift_Cards`
- `Magazine_Subscriptions`
- `Subscription_Boxes`

Su menor cantidad de reviews reduce tiempo y almacenamiento, aunque algunas tienen poca variedad o pocos productos.

### Categorías que conviene tratar con precaución

- `Health_and_Household` y `Health_and_Personal_Care`: pueden generar afirmaciones médicas o de seguridad que requieren controles adicionales.
- `Industrial_and_Scientific`, `Automotive` y `Tools_and_Home_Improvement`: una recomendación incorrecta puede implicar incompatibilidad o riesgo físico.
- `Unknown`: mezcla dominios y dificulta establecer tareas, métricas y splits confiables.
- `Books`, `Kindle_Store`, `Movies_and_TV`, `Digital_Music` y `Software`: requieren atención adicional a copyright, licencias y naturaleza del contenido.

## Observaciones para el análisis

- Los nombres son categorías de catálogo, no etiquetas perfectas de dominio.
- Puede haber productos mal categorizados o presentes en taxonomías históricas similares.
- `All_Beauty` y `Beauty_and_Personal_Care` no deben fusionarse sin comparar sus distribuciones.
- `Health_and_Household` y `Health_and_Personal_Care` también deben analizarse por separado antes de una eventual unión.
- La cantidad de reviews no equivale a cantidad de productos ni a diversidad efectiva.
- Para prevenir leakage, los splits deben hacerse por `parent_asin`, no por categoría o review aislada.

## Fuente

- Ficha y estadísticas oficiales: <https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/README.md>

