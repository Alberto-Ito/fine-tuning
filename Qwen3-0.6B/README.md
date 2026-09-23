# Fine-tuning de Qwen3-0.6B

Este proyecto contiene un pipeline reproducible para ajustar y evaluar
`Qwen/Qwen3-0.6B` en dos tareas independientes de clasificación de intención:

- `PolyAI/banking77`
- `clinc/clinc_oos`

## Estructura

```text
Qwen3-0.6B/
├── configs/
│   ├── datasets/       # Configuración propia de cada dataset
│   └── experiments/    # Hiperparámetros de cada corrida
├── data/
│   ├── raw/            # Descargas originales e inmutables
│   └── processed/      # Datos transformados listos para entrenar
├── notebooks/          # Exploración y análisis interactivo
├── outputs/
│   ├── banking77/      # Checkpoints, logs, métricas y predicciones
│   └── clinc_oos/      # del experimento correspondiente
├── reports/
│   ├── figures/        # Gráficos comparativos
│   └── tables/         # Tablas de resultados
├── scripts/            # Comandos ejecutables del pipeline
├── src/
│   └── qwen3_finetuning/
│       ├── data/       # Descarga, validación y preprocesamiento
│       ├── training/   # Modelo, tokenización y entrenamiento
│       ├── evaluation/ # Métricas, inferencia y comparación
│       └── utils/      # Funciones compartidas
└── tests/              # Pruebas unitarias y de integración
```

## Flujo previsto

1. Descargar cada dataset en `data/raw/<dataset>/`.
2. Normalizarlo y guardarlo en `data/processed/<dataset>/`.
3. Definir parámetros en `configs/datasets/` y `configs/experiments/`.
4. Entrenar y guardar cada corrida en `outputs/<dataset>/`.
5. Evaluar el modelo y consolidar comparaciones en `reports/`.

Los datos, checkpoints y demás artefactos grandes son locales y no se
versionan. Los archivos `.gitkeep` conservan únicamente la estructura vacía.
